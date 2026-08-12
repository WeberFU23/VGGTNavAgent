"""pointing 单元测试：JSON 解析、bbox 交叉验证、深度采样（mock gateway）。

    python tests/test_pointing.py
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.pointing import PointingGrounder, sample_point_depth
from mapping.vllm_client import VLLMError


class _MockGateway:
    """按队列返回 chat_json 结果；队列空则返回 {}。"""

    def __init__(self, replies=None, errors=None):
        self.replies = list(replies or [])
        self.errors = list(errors or [])
        self.calls = []

    def chat_json(self, model, prompt, images, *, kind, cache_key,
                  priority, max_tokens):
        self.calls.append({"kind": kind, "cache_key": cache_key,
                           "prompt": prompt})
        if self.errors and self.errors[0]:
            self.errors.pop(0)
            raise VLLMError("bad json")
        return self.replies.pop(0) if self.replies else {}


def _img(w=64, h=48):
    return Image.new("RGB", (w, h))


# ----------------------------------------------------------------------
# point
# ----------------------------------------------------------------------
def test_point_returns_instances():
    gw = _MockGateway([{"instances": [
        {"pixel": [10.2, 20.7], "bbox": [5, 15, 30, 40], "confidence": 0.9},
        {"pixel": [50, 5], "bbox": None, "confidence": 0.6},
    ]}])
    g = PointingGrounder(gw, model="qwen-7b")
    pts = g.point(_img(), "gray fabric sofa", frame_key="f1")
    assert len(pts) == 2
    assert pts[0]["pixel"] == (10.2, 20.7)
    assert pts[0]["confidence"] == pytest.approx(0.9)
    assert pts[1]["confidence"] == pytest.approx(0.6)


def test_point_bbox_mismatch_penalized():
    gw = _MockGateway([{"instances": [
        {"pixel": [60, 5], "bbox": [0, 0, 10, 10], "confidence": 0.8},
    ]}])
    g = PointingGrounder(gw, model="qwen-7b")
    pts = g.point(_img(), "basket")
    # point 落在 bbox 外 -> 置信度乘 0.5
    assert pts[0]["confidence"] == pytest.approx(0.4)


def test_point_clamps_to_image_and_drops_invalid():
    gw = _MockGateway([{"instances": [
        {"pixel": [999, -3], "confidence": 0.7},
        {"pixel": "nonsense", "confidence": 0.9},
        {"pixel": [1, 2], "confidence": 5.0},
    ]}])
    g = PointingGrounder(gw, model="qwen-7b")
    pts = g.point(_img(w=64, h=48), "basket")
    assert len(pts) == 2
    assert pts[0]["pixel"] == (63, 0)
    assert pts[1]["confidence"] == 1.0     # clamp 到 [0,1]


def test_point_retries_on_invalid_json():
    gw = _MockGateway(
        replies=[{"instances": [{"pixel": [3, 4], "confidence": 0.5}]}],
        errors=[True])
    g = PointingGrounder(gw, model="qwen-7b", parse_retries=1)
    pts = g.point(_img(), "basket")
    assert len(pts) == 1
    assert len(gw.calls) == 2
    assert "not valid JSON" in gw.calls[1]["prompt"]


def test_point_gives_up_after_retries():
    gw = _MockGateway(errors=[True, True])
    g = PointingGrounder(gw, model="qwen-7b", parse_retries=1)
    assert g.point(_img(), "basket") == []


def test_requires_model():
    with pytest.raises(RuntimeError, match="NAV_POINTING_MODEL_PATH"):
        PointingGrounder(_MockGateway(), model="")


# ----------------------------------------------------------------------
# verify_frame
# ----------------------------------------------------------------------
def test_verify_frame_match():
    gw = _MockGateway([{
        "match": True,
        "checked_attributes": [
            {"attribute": "gray", "satisfied": True},
            {"attribute": "fabric", "satisfied": True}],
        "confidence": 0.85, "reason": "all attributes visible"}])
    g = PointingGrounder(gw, model="qwen-7b")
    out = g.verify_frame(_img(), "gray fabric sofa")
    assert out["match"] is True
    assert out["confidence"] == pytest.approx(0.85)
    assert len(out["checked_attributes"]) == 2


def test_verify_frame_conservative_on_invalid():
    gw = _MockGateway(errors=[True, True])
    g = PointingGrounder(gw, model="qwen-7b", parse_retries=1)
    out = g.verify_frame(_img(), "basket")
    assert out["match"] is False
    assert out["confidence"] == 0.0


# ----------------------------------------------------------------------
# sample_point_depth
# ----------------------------------------------------------------------
def _grid(h=20, w=20, z=2.0):
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.astype(float), ys.astype(float),
                    np.full((h, w), z)], axis=-1)
    return pts


def test_sample_median_point():
    pts = _grid()
    conf = np.ones(pts.shape[:2], dtype=bool)
    out = sample_point_depth(pts, conf, pixel=(10, 10), patch=11)
    assert out["found"] is True
    assert out["num_points"] == 121
    np.testing.assert_allclose(out["point"], [10, 10, 2.0], atol=0.6)


def test_sample_filters_low_confidence():
    pts = _grid()
    conf = np.ones(pts.shape[:2], dtype=bool)
    conf[:, :] = False
    conf[10, 10] = True
    out = sample_point_depth(pts, conf, pixel=(10, 10), patch=11,
                             min_points=1)
    assert out["found"] is True
    assert out["num_points"] == 1
    np.testing.assert_allclose(out["point"], [10, 10, 2.0])


def test_sample_too_few_points():
    pts = _grid()
    conf = np.zeros(pts.shape[:2], dtype=bool)
    out = sample_point_depth(pts, conf, pixel=(5, 5), patch=11)
    assert out["found"] is False
    assert out["point"] is None


def test_sample_nan_points_ignored():
    pts = _grid()
    pts[4:17, 4:17] = np.nan           # 覆盖整个 11x11 patch
    conf = np.ones(pts.shape[:2], dtype=bool)
    out = sample_point_depth(pts, conf, pixel=(10, 10), patch=11,
                             min_points=5)
    assert out["found"] is False        # 中心 patch 全 NaN


def test_sample_depth_std_uses_cam_origin():
    pts = _grid(z=3.0)
    conf = np.ones(pts.shape[:2], dtype=bool)
    out = sample_point_depth(pts, conf, pixel=(10, 10), patch=11,
                             cam_origin=[0, 0, 0])
    assert out["found"] is True
    assert out["depth_std"] is not None and out["depth_std"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
