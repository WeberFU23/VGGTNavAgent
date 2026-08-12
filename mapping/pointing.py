"""VLM pointing 定位：替代 SAM3 mask 投影（server 端，vggtslam 环境）。

链路：retrieve_captions 粗筛 → verify_frame 查询条件化复核（逐条核对
属性，滤假阳性帧）→ point 输出目标像素（point 优先，bbox 只作交叉
验证：point 落在 bbox 外则降置信；一次调用允许返回多个实例点）→
sample_point_depth 在像素周围 patch 内先按 VGGT confidence 过滤低分
点再取中位数，得 3D 点。

模型固定 Qwen2.5-VL-7B（4bit，经 VLLMGateway 调用）。输出走 JSON
schema 校验，解析失败重试 1 次。本模块不 import torch/cv2，
sample_point_depth 为纯 numpy 函数，可脱离 GPU 单测。
"""

import math

import numpy as np

from mapping.vllm_client import Priority, VLLMError

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

POINT_PROMPT = """You are localizing target objects for a robot. Target
description: {goal_text}

The image is {width}x{height} pixels. Point at every visible object instance
matching the FULL description (including color/material attributes). For
each instance return one pixel point on the object body (not the
background), and optionally a tight bounding box for cross-checking.
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


class PointingGrounder:
    """Qwen2.5-VL-7B pointing + 属性复核。model 为空时构造报错。"""

    def __init__(self, gateway, model, parse_retries=1, max_tokens=512):
        self.gateway = gateway
        self.model = str(model or "").strip()
        if not self.model:
            raise RuntimeError(
                "pointing 模型未配置：请设置 NAV_POINTING_MODEL_PATH"
                "（Qwen2.5-VL-7B-Instruct 权重路径）")
        self.parse_retries = int(parse_retries)
        self.max_tokens = int(max_tokens)

    # ------------------------------------------------------------------
    def verify_frame(self, pil_img, goal_text, frame_key=None):
        """查询条件化复核。返回 {match, checked_attributes, confidence, reason}；
        模型持续非法输出时返回 match=False（保守滤除）。"""
        prompt = VERIFY_PROMPT.format(goal_text=str(goal_text))
        data = self._chat_checked(prompt, pil_img, frame_key, "verify")
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

    # ------------------------------------------------------------------
    def _chat_checked(self, prompt, pil_img, frame_key, kind):
        """调模型并解析 JSON；失败时带错误提示重试 parse_retries 次。"""
        retry_note = ""
        for _ in range(self.parse_retries + 1):
            try:
                return self.gateway.chat_json(
                    self.model, prompt + retry_note, [pil_img], kind=kind,
                    cache_key=frame_key if not retry_note else None,
                    priority=Priority.POINTING, max_tokens=self.max_tokens)
            except VLLMError:
                retry_note = (
                    "\nYour previous output was not valid JSON. "
                    "Return exactly one JSON object and nothing else.")
        return None


# ----------------------------------------------------------------------
# 深度采样（纯 numpy，供单测与 server._resolve_point_patch 复用）
# ----------------------------------------------------------------------
def sample_point_depth(points_hw3, conf_mask_hw, pixel, patch=11,
                       min_points=5, cam_origin=None):
    """point 像素周围 patch 采样 3D 点。

    points_hw3: (H, W, 3) 世界系点（NaN = 无效）；
    conf_mask_hw: (H, W) bool，VGGT confidence 过滤掩码（True=可信）；
    pixel: (x, y) 点云网格坐标；patch: 采样窗口边长。

    先按 conf 过滤低分点再取中位数。返回 {found, point, num_points,
    depth_std, spread}；depth_std 是 patch 内点到相机原点距离的标准差
    （cam_origin 缺省时用世界原点），用于"深度方差过大 → 降级 belief"
    的小/远目标两段式判定。
    """
    points = np.asarray(points_hw3, dtype=np.float64)
    conf = np.asarray(conf_mask_hw, dtype=bool)
    h, w = points.shape[:2]
    if conf.shape != (h, w):
        conf = np.ones((h, w), dtype=bool)
    half = max(int(patch), 1) // 2
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
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


# ----------------------------------------------------------------------
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
