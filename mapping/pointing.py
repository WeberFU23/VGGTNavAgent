"""VLM pointing 定位（server 端，vggtslam 环境）。

链路：探索时 retrieve_captions 粗筛 → point 输出目标像素（point 优先，
bbox 只作交叉验证：point 落在 bbox 外则降置信；一次调用允许返回多个
实例点）→ sample_point_depth 在像素周围 patch 内过滤低置信点并取
中位数，得到 3D instance。到达后的判断由决策 VLM 直接完成；
verify_frame 仅保留给 ground_frame 诊断接口。

后端由 NAV_POINTING_BACKEND 选择：
- qwen（默认）：JSON 输出绝对像素坐标，模型路径 NAV_POINTING_MODEL_PATH。
- molmo：<point>/<points> XML 标签输出 0-100 归一化坐标，模型路径同为
  NAV_POINTING_MODEL_PATH；Molmo 不输出 bbox/confidence，bbox 交叉验证
  自动跳过，confidence 固定 1.0。
输出走校验，解析失败重试 1 次。本模块不 import torch/cv2，
sample_point_depth 为纯 numpy 函数，可脱离 GPU 单测。
"""

import math
import os
import re

import numpy as np

from mapping.vllm_client import Priority, VLLMError


class PointingBackendUnavailable(RuntimeError):
    """The configured pointing inference service cannot serve requests."""

VERIFY_PROMPT = """You are verifying a retrieval candidate for an embodied
navigation agent. Target description: {goal_text}

Look at the image and check, attribute by attribute (category, color,
material, shape, distinguishing details), whether an object matching the
FULL description is visible. The caption-based retriever is recall-oriented
and returns false positives; your job is to filter them out. Only report
match=true when every stated attribute is satisfied by a visible object.

Return exactly one JSON object:
{{
  "match": true or false,
  "checked_attributes": [{{"attribute": "...", "satisfied": true or false}}],
  "confidence": 0.0,
  "reason": "one short sentence"
}}"""

POINT_PROMPT = """You are localizing target objects for a robot. Target description is: {goal_text}

The image is {width}x{height} pixels. Point at every visible object instance
matching the FULL description. For each instance you should:
- Return one pixel at the center of the object's visible region, on the
  object surface, away from edges and occluders. This pixel will be used to
  sample the object's depth, so it must lie on the object itself, not on the
  background or a nearby object.
- Optionally add a tight bounding box for cross-checking.
- Use absolute pixel coordinates: x is horizontal (0 at left), y is vertical
  (0 at top).
If nothing matches, return an empty instances list.

Return exactly one JSON object:
{{
  "instances": [
    {{"pixel": [x, y], "bbox": [x0, y0, x1, y1] or null,
      "confidence": 0.0}}
  ]
}}"""

# point 落在 bbox 外时置信度乘的惩罚系数（bbox 仅交叉验证，point 优先）
_BBOX_MISMATCH_FACTOR = 0.5

# Molmo 的 pointing 输出为 <point x=".." y=".." alt=".."> 或
# <points x1=".." y1=".." x2=".." .../>，坐标是 0-100 归一化百分比。
MOLMO_POINT_PROMPT = (
    "Point to every visible object matching this description: {goal_text}")

_MOLMO_TAG_RE = re.compile(r"<points?\b([^>]*)/?>", re.IGNORECASE)


def _parse_molmo_points(text, w, h):
    """把 Molmo 的 0-100 归一化 <point>/<points> 标签解析为像素坐标。"""
    out = []
    for tag in _MOLMO_TAG_RE.findall(str(text)):
        xs = dict()
        ys = dict()
        for m in re.finditer(r'(x|y)(\d*)="([\d.]+)"', tag):
            axis, idx, value = m.group(1), m.group(2) or "0", m.group(3)
            (xs if axis == "x" else ys)[idx] = float(value)
        for idx, xv in xs.items():
            if idx not in ys:
                continue
            px = min(max(xv / 100.0 * w, 0), w - 1)
            py = min(max(ys[idx] / 100.0 * h, 0), h - 1)
            out.append({"pixel": (px, py), "confidence": 1.0, "bbox": None})
    return out


class PointingGrounder:
    """本地多模态 VLM pointing；model 为空时构造报错。"""

    def __init__(self, gateway, model, parse_retries=1, max_tokens=512,
                 backend=None):
        self.gateway = gateway
        self.model = str(model or "").strip()
        if not self.model:
            raise RuntimeError(
                "pointing 模型未配置：请设置 NAV_POINTING_MODEL_PATH"
                "（Qwen2.5-VL-7B-Instruct 或 Molmo 权重路径）")
        self.backend = str(
            backend or os.environ.get("NAV_POINTING_BACKEND", "qwen")
        ).strip().lower()
        self.parse_retries = int(parse_retries)
        self.max_tokens = int(max_tokens)

    def check_health(self, timeout=10.0):
        """Fail fast unless the endpoint is reachable and model is loaded."""
        try:
            return self.gateway.healthcheck(self.model, timeout=timeout)
        except VLLMError as exc:
            raise PointingBackendUnavailable(str(exc)) from exc

    def verify_frame(self, pil_img, goal_text, frame_key=None):
        """诊断用条件化复核；模型持续非法输出时保守返回 match=False。"""
        prompt = VERIFY_PROMPT.format(goal_text=str(goal_text))
        try:
            data = self._chat_checked(prompt, pil_img, frame_key, "verify")
        except PointingBackendUnavailable:
            # verify_frame is a legacy diagnostic API, not the production
            # pointing/instantiation path. Keep its conservative contract.
            data = None
        if data is None:
            return {"match": False, "checked_attributes": [],
                    "confidence": 0.0, "reason": "model output invalid"}
        attrs = data.get("checked_attributes") or []
        if not isinstance(attrs, list):
            attrs = []
        return {
            "match": bool(data.get("match")),
            "checked_attributes": attrs,
            "confidence": _clamp01(data.get("confidence", 0.0)),
            "reason": str(data.get("reason") or "")[:200],
        }

    def point(self, pil_img, goal_text, frame_key=None):
        """pointing：返回 [{pixel:(x,y), confidence, bbox}]，可多实例。

        point 优先；bbox 仅交叉验证——point 落在 bbox 外则降置信。
        像素坐标裁剪到图像范围内。模型持续非法输出时返回空列表。
        """
        w, h = pil_img.size
        if self.backend == "molmo":
            return self._point_molmo(pil_img, goal_text, frame_key, w, h)
        prompt = POINT_PROMPT.format(goal_text=str(goal_text),
                                     width=w, height=h)
        data = self._chat_checked(prompt, pil_img, frame_key, "point")
        if data is None:
            return []
        instances = data.get("instances") or []
        if not isinstance(instances, list):
            return []
        out = []
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            pixel = _parse_xy(inst.get("pixel"))
            if pixel is None:
                continue
            x = min(max(pixel[0], 0), w - 1)
            y = min(max(pixel[1], 0), h - 1)
            conf = _clamp01(inst.get("confidence", 0.0))
            bbox = _parse_bbox(inst.get("bbox"), w, h)
            if bbox is not None and not _inside((x, y), bbox):
                conf *= _BBOX_MISMATCH_FACTOR
            out.append({"pixel": (x, y), "confidence": conf, "bbox": bbox})
        return out

    def _point_molmo(self, pil_img, goal_text, frame_key, w, h):
        """Molmo pointing：纯文本 <point> 标签输出，无 bbox/confidence。"""
        prompt = MOLMO_POINT_PROMPT.format(goal_text=str(goal_text))
        last_error = None
        for _ in range(self.parse_retries + 1):
            try:
                text = self.gateway.chat(
                    self.model, prompt, [pil_img], kind="point",
                    cache_key=frame_key, priority=Priority.POINTING,
                    max_tokens=self.max_tokens)
            except VLLMError as exc:
                last_error = exc
                continue
            out = _parse_molmo_points(text, w, h)
            if out or not re.search(r"<points?\b", str(text)):
                # 解析到点，或模型明确没有输出任何 point 标签（即没有目标）
                return out
        if last_error is not None:
            raise PointingBackendUnavailable(str(last_error)) from last_error
        return []

    def _chat_checked(self, prompt, pil_img, frame_key, kind):
        """调模型并解析 JSON；失败时带错误提示重试 parse_retries 次。"""
        retry_note = ""
        last_error = None
        for _ in range(self.parse_retries + 1):
            try:
                return self.gateway.chat_json(
                    self.model, prompt + retry_note, [pil_img], kind=kind,
                    cache_key=frame_key if not retry_note else None,
                    priority=Priority.POINTING, max_tokens=self.max_tokens)
            except VLLMError as exc:
                last_error = exc
                retry_note = (
                    "\nYour previous output was not valid JSON. "
                    "Return exactly one JSON object and nothing else.")
        if last_error is not None:
            raise PointingBackendUnavailable(str(last_error)) from last_error
        return None


# ----------------------------------------------------------------------
# 深度采样（纯 numpy，供单测与 server._resolve_point 复用）
# ----------------------------------------------------------------------
def sample_point_depth(points_hw3, conf_mask_hw, pixel, patch=11,
                       min_points=5, cam_origin=None, bbox=None,
                       bbox_margin=0.15):
    """point 像素周围 patch 采样 3D 点。

    points_hw3: (H, W, 3) 世界系点（NaN = 无效）；
    conf_mask_hw: (H, W) bool，VGGT confidence 过滤掩码（True=可信）；
    pixel: (x, y) 点云网格坐标；patch: 采样窗口边长。

    bbox 非空时把采样窗口约束在 bbox 内部（四边各内缩 bbox_margin，
    避开边缘）：point 有像素级误差时 bbox 通常更可靠。point 远离
    bbox 导致窗口交集为空时，退化为在 bbox 内区中心开小窗采样。

    先按 conf 过滤低分点再取中位数。返回 {found, point, num_points,
    depth_std, spread}；depth_std 是 patch 内点到相机原点距离的标准差
    （cam_origin 缺省时用世界原点），作为实例的几何证据供 VLM 判断。
    """
    points = np.asarray(points_hw3, dtype=np.float64)
    conf = np.asarray(conf_mask_hw, dtype=bool)
    h, w = points.shape[:2]
    if conf.shape != (h, w):
        conf = np.ones((h, w), dtype=bool)
    half = max(int(patch), 1) // 2
    x, y = int(round(pixel[0])), int(round(pixel[1]))

    def _window(cx, cy):
        return (max(0, cx - half), min(w, cx + half + 1),
                max(0, cy - half), min(h, cy + half + 1))

    x0, x1, y0, y1 = _window(x, y)
    if bbox is not None:
        bx0, by0, bx1, by1 = (float(v) for v in bbox)
        mx = (bx1 - bx0) * float(bbox_margin)
        my = (by1 - by0) * float(bbox_margin)
        ix0, iy0 = bx0 + mx, by0 + my
        ix1, iy1 = max(bx1 - mx, ix0 + 1), max(by1 - my, iy0 + 1)
        cx0 = max(x0, int(math.ceil(min(ix0, w))))
        cy0 = max(y0, int(math.ceil(min(iy0, h))))
        cx1 = min(x1, int(math.floor(max(ix1, 0))) + 1)
        cy1 = min(y1, int(math.floor(max(iy1, 0))) + 1)
        if cx1 > cx0 and cy1 > cy0:
            x0, x1, y0, y1 = cx0, cx1, cy0, cy1
        else:
            # point 不在 bbox 附近：以 bbox 内区中心为准重新开窗
            bx = int(round((ix0 + ix1) / 2))
            by = int(round((iy0 + iy1) / 2))
            x0, x1, y0, y1 = _window(
                min(max(bx, 0), w - 1), min(max(by, 0), h - 1))
    patch_pts = points[y0:y1, x0:x1, :].reshape(-1, 3)
    patch_ok = conf[y0:y1, x0:x1].reshape(-1) & np.isfinite(patch_pts).all(axis=1)
    valid = patch_pts[patch_ok]
    if len(valid) < int(min_points):
        return {"found": False, "point": None, "num_points": int(len(valid)),
                "depth_std": None, "spread": None}
    origin = (np.asarray(cam_origin, dtype=np.float64)
              if cam_origin is not None else np.zeros(3))
    depths = np.linalg.norm(valid - origin, axis=1)
    return {
        "found": True,
        "point": np.median(valid, axis=0),
        "num_points": int(len(valid)),
        "depth_std": float(np.std(depths)),
        "spread": float(np.percentile(depths, 90) - np.percentile(depths, 10)),
    }


def _clamp01(value):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _parse_bbox(value, w, h):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
    y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [x0, y0, x1, y1]


def _inside(pixel, bbox):
    x0, y0, x1, y1 = bbox
    return x0 <= pixel[0] <= x1 and y0 <= pixel[1] <= y1
