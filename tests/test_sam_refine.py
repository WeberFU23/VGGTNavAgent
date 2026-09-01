"""SAM mask 精化与 SoM 的本地单测（不依赖 torch/SAM 权重）。"""
import numpy as np
import pytest

from mapping.pointing import sample_point_depth
from mapping.sam_backend import (SAMRefiner, mask_bbox, mask_centroid,
                                 _REFINE_MAX_AREA_FRAC)


def _grid(h=32, w=32, depth=5.0):
    """均匀点云网格：z=depth，x/y 随像素变化。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pts = np.stack([xx * 0.01, yy * 0.01,
                    np.full_like(xx, depth)], axis=-1)
    return pts


def test_sample_point_depth_prefers_mask():
    pts = _grid()
    conf = np.ones(pts.shape[:2], dtype=bool)
    # mask 区域深度 9m，其余 5m；采样结果应等于 mask 中位数（9m）
    pts[4:8, 4:8, 2] = 9.0
    mask = np.zeros(pts.shape[:2], dtype=bool)
    mask[4:8, 4:8] = True
    out = sample_point_depth(pts, conf, (1, 1), patch=5, mask_hw=mask)
    assert out["found"] and out["sampled_from"] == "mask"
    assert abs(out["point"][2] - 9.0) < 1e-6


def test_sample_point_depth_mask_too_small_falls_back():
    pts = _grid()
    conf = np.ones(pts.shape[:2], dtype=bool)
    mask = np.zeros(pts.shape[:2], dtype=bool)
    mask[10, 10] = True  # 只有 1 个点，不足 min_points
    out = sample_point_depth(pts, conf, (10, 10), patch=11, mask_hw=mask)
    assert out["found"] and out.get("sampled_from") == "patch"


def test_sample_point_depth_mask_shape_mismatch_ignored():
    pts = _grid()
    conf = np.ones(pts.shape[:2], dtype=bool)
    mask = np.zeros((16, 16), dtype=bool)  # 形状不符直接忽略
    out = sample_point_depth(pts, conf, (10, 10), patch=11, mask_hw=mask)
    assert out["found"] and out.get("sampled_from") == "patch"


def test_sample_point_depth_mask_subsamples_large_region():
    pts = _grid(h=64, w=64)
    conf = np.ones(pts.shape[:2], dtype=bool)
    mask = np.ones(pts.shape[:2], dtype=bool)  # 4096 点 > max_mask_points
    out = sample_point_depth(pts, conf, (32, 32), mask_hw=mask,
                             max_mask_points=100)
    assert out["found"] and out["num_points"] == 100


def test_mask_centroid_and_bbox():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 4:12] = True
    cx, cy = mask_centroid(mask)
    assert abs(cx - 7.5) < 1e-6 and abs(cy - 3.5) < 1e-6
    assert mask_bbox(mask) == [4.0, 2.0, 12.0, 6.0]
    assert mask_centroid(np.zeros((4, 4), bool)) is None
    assert mask_bbox(np.zeros((4, 4), bool)) is None


def test_refiner_unavailable_without_ckpt(monkeypatch):
    monkeypatch.setenv("NAV_SAM_CKPT", "/nonexistent/sam.pth")
    refiner = SAMRefiner.from_env()
    assert not refiner.available
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    assert refiner.segment_at_point(rgb, (5, 5)) is None
    assert refiner.segment_all(rgb) == []


def test_refiner_disabled_by_env(monkeypatch):
    monkeypatch.setenv("NAV_SAM_ENABLED", "0")
    refiner = SAMRefiner.from_env()
    assert not refiner.available
    assert refiner.disabled_reason == "NAV_SAM_ENABLED=0"
