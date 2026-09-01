"""SAM 分割后端（server 端，vggtslam 环境）。

两个模式：
- 点提示精化（segment_at_point）：pointing 模型给粗落点，SAM 返回该点
  所在物体的完整 mask，用质心替代原始像素、用 mask 区域采样深度，
  消除"点在边缘混入背景"导致的 3D 漂移。
- 全景分割（segment_all，SoM）：automatic mask generator 输出整帧的
  物体级 mask 列表，渲染编号 overlay 交给决策 VLM 做多选——把"生成
  坐标"变成"选择题"。pointing 模型反复 REJECT（选错物体）时由决策
  VLM 自行升级到此模式。

segment_anything 惰性导入：未安装或权重缺失时 SAMRefiner.available
为 False，所有调用方自动退回无 SAM 的旧行为。本模块不在导入期
触碰 torch，本地（无 GPU/无 SAM 权重）可以安全 import 做单测。

环境变量：
- NAV_SAM_ENABLED: 0/1（默认 1；权重不存在时自动不可用）
- NAV_SAM_CKPT: sam_vit_h_4b8939.pth 等权重路径
- NAV_SAM_MODEL_TYPE: vit_h / vit_l / vit_b（默认 vit_h）
- NAV_SAM_DEVICE: cuda / cpu（默认 cuda，不可用自动回退 cpu）
"""

import hashlib
import os

import numpy as np

# SoM 过滤：mask 面积占比小于该值视为噪声（远处小点/碎片）
_SOM_MIN_AREA_FRAC = 0.002
# mask 面积占比超过该值视为背景区域（墙/地板/天花板），不作为候选
_SOM_MAX_AREA_FRAC = 0.55
# 点提示精化：mask 面积超过该值认为 SAM 选中了背景，不采纳
_REFINE_MAX_AREA_FRAC = 0.60


class SAMRefiner:
    """SAM 封装：set_image 按帧缓存，点提示与全分割共用。"""

    def __init__(self, ckpt=None, model_type=None, device=None,
                 max_masks=24):
        self.ckpt = str(
            ckpt or os.environ.get("NAV_SAM_CKPT", "")).strip()
        self.model_type = str(
            model_type or os.environ.get("NAV_SAM_MODEL_TYPE", "vit_h"))
        self.device = str(
            device or os.environ.get("NAV_SAM_DEVICE", "cuda"))
        self.max_masks = int(max_masks)
        self._predictor = None
        self._generator = None
        self._cached_key = None
        self.available = False
        self.disabled_reason = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls):
        enabled = os.environ.get("NAV_SAM_ENABLED", "1").strip().lower() in {
            "1", "true", "yes", "on"}
        refiner = cls()
        if not enabled:
            refiner.disabled_reason = "NAV_SAM_ENABLED=0"
            return refiner
        refiner.load()
        return refiner

    def load(self):
        """加载权重；失败只记录原因，不抛异常（调用方退回旧行为）。"""
        if not self.ckpt or not os.path.isfile(self.ckpt):
            self.disabled_reason = f"SAM 权重不存在: {self.ckpt or '<unset>'}"
            return False
        try:
            import torch
            from segment_anything import (SamAutomaticMaskGenerator,
                                          sam_model_registry, SamPredictor)
            device = self.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            sam = sam_model_registry[self.model_type](checkpoint=self.ckpt)
            sam.to(device)
            self._predictor = SamPredictor(sam)
            self._generator = SamAutomaticMaskGenerator(
                sam, points_per_side=32, pred_iou_thresh=0.86,
                stability_score_thresh=0.92, min_mask_region_area=100)
            self.available = True
            self.disabled_reason = None
            return True
        except Exception as exc:  # noqa: BLE001 - 不可用时降级而非崩服务
            self.disabled_reason = f"SAM 加载失败: {type(exc).__name__}: {exc}"
            self._predictor = None
            self._generator = None
            self.available = False
            return False

    # ------------------------------------------------------------------
    # 帧缓存（set_image 的 embedding 是点提示路径的主要开销）
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_key(rgb, cache_key=None):
        if cache_key:
            return str(cache_key)
        small = np.asarray(rgb)[::32, ::32].tobytes()
        return hashlib.md5(small).hexdigest()

    def _ensure_image(self, rgb, cache_key=None):
        key = self._frame_key(rgb, cache_key)
        if key != self._cached_key:
            self._predictor.set_image(np.asarray(rgb, dtype=np.uint8))
            self._cached_key = key

    # ------------------------------------------------------------------
    # 模式一：点提示精化
    # ------------------------------------------------------------------
    def segment_at_point(self, rgb, pixel, cache_key=None):
        """point 提示分割。返回 {mask, centroid, bbox, area_frac, iou} 或 None。

        multimask 输出中按"面积不超背景阈值 + 预测 IoU 最高"选择；
        全部超阈值时返回 None（视为选中了墙/地板等背景）。
        """
        if not self.available:
            return None
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        try:
            self._ensure_image(rgb, cache_key)
            masks, ious, _ = self._predictor.predict(
                point_coords=np.asarray([[float(pixel[0]),
                                          float(pixel[1])]]),
                point_labels=np.ones(1, dtype=np.int64),
                multimask_output=True)
        except Exception:  # noqa: BLE001 - 单次失败不影响后续请求
            return None
        if masks is None or len(masks) == 0:
            return None
        best = None
        for mask, iou in zip(masks, ious):
            area_frac = float(np.count_nonzero(mask)) / float(h * w)
            if area_frac < 1e-6 or area_frac > _REFINE_MAX_AREA_FRAC:
                continue
            if best is None or float(iou) > best[1]:
                best = (np.asarray(mask, dtype=bool), float(iou), area_frac)
        if best is None:
            return None
        mask, iou, area_frac = best
        return {"mask": mask, "centroid": mask_centroid(mask),
                "bbox": mask_bbox(mask), "area_frac": area_frac, "iou": iou}

    # ------------------------------------------------------------------
    # 模式二：全景分割（SoM）
    # ------------------------------------------------------------------
    def segment_all(self, rgb, max_masks=None):
        """整帧自动分割，返回过滤后的 mask 列表（mask_id 从 1 编号）。

        过滤面积过小/过大（背景）的 mask，按面积降序截断到 max_masks。
        每项：{mask_id, mask, centroid, bbox, area_frac, iou}。
        """
        if not self.available:
            return []
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        try:
            raw = self._generator.generate(rgb)
        except Exception:  # noqa: BLE001
            return []
        limit = int(max_masks or self.max_masks)
        kept = []
        for item in raw:
            mask = np.asarray(item.get("segmentation"), dtype=bool)
            if mask.shape != (h, w):
                continue
            area_frac = float(np.count_nonzero(mask)) / float(h * w)
            if not (_SOM_MIN_AREA_FRAC <= area_frac <= _SOM_MAX_AREA_FRAC):
                continue
            kept.append({"mask": mask, "area_frac": area_frac,
                         "iou": float(item.get("predicted_iou", 0.0)),
                         "centroid": mask_centroid(mask),
                         "bbox": mask_bbox(mask)})
        kept.sort(key=lambda row: row["area_frac"], reverse=True)
        kept = kept[:limit]
        for i, row in enumerate(kept, start=1):
            row["mask_id"] = i
        return kept


# ----------------------------------------------------------------------
# 纯 numpy 工具函数（可脱离 SAM 单测）
# ----------------------------------------------------------------------
def mask_centroid(mask):
    """mask 质心像素 (x, y)；空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_bbox(mask):
    """mask 外接框 [x0, y0, x1, y1]（x1/y1 为开区间）；空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()),
            float(xs.max()) + 1.0, float(ys.max()) + 1.0]


def render_som_overlay(rgb, masks, out_long_side=1024):
    """编号 mask overlay：每个 mask 半透明着色 + 质心处标注 mask_id。

    输出放大到长边 out_long_side 的 JPEG 字节，保证编号对 VLM 可读。
    需要 cv2；返回 None 表示编码失败。
    """
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    overlay = rgb.copy()
    h, w = overlay.shape[:2]
    for row in masks:
        color = _mask_color(row["mask_id"])
        region = row["mask"]
        overlay[region] = (0.45 * overlay[region] +
                           0.55 * np.asarray(color)).astype(np.uint8)
    scale = max(1.0, float(out_long_side) / float(max(h, w)))
    if scale > 1.0:
        overlay = cv2.resize(overlay, (int(round(w * scale)),
                                       int(round(h * scale))),
                             interpolation=cv2.INTER_LINEAR)
    font_scale = max(0.7, 1.1 * scale)
    for row in masks:
        centroid = row.get("centroid")
        if centroid is None:
            continue
        cx, cy = (int(round(centroid[0] * scale)),
                  int(round(centroid[1] * scale)))
        text = str(row["mask_id"])
        for thickness, color in ((6, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(overlay, text, (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        thickness, cv2.LINE_AA)
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes() if ok else None


def _mask_color(mask_id):
    """稳定的高区分度颜色（HSV 环均匀取值）。"""
    import cv2

    hue = int((int(mask_id) * 47) % 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))  # RGB 顺序
