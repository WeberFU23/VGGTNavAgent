"""semantic_memory 后端 ground_object 端到端测试（假点云 + mock 模型）。

server.py 依赖 cv2/torch/torchvision（vggtslam 环境），本地单测用
轻量 stub 注入 sys.modules 后导入，绕开 GPU/模型依赖。

    python tests/test_semantic_ground.py
"""

import os
import sys
import threading
import types

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stubs():
    """为 cv2/torch/torchvision 注入最小 stub（import 用，不调用 GPU）。"""
    for name in ("cv2", "torch"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.ModuleType(name)
    try:
        import torchvision  # noqa: F401
    except ImportError:
        tv = types.ModuleType("torchvision")
        transforms = types.ModuleType("torchvision.transforms")
        functional = types.ModuleType("torchvision.transforms.functional")
        functional.to_pil_image = lambda x: Image.fromarray(
            np.asarray(x, dtype=np.uint8))
        transforms.functional = functional
        tv.transforms = transforms
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.functional"] = functional


_install_stubs()

from mapping.server import MappingServer  # noqa: E402


# ----------------------------------------------------------------------
# 假 SLAM 世界：两个子图共 2 帧，点云 = 平面网格
# ----------------------------------------------------------------------
class _FakeSubmap:
    def __init__(self, sid, frame_ids, z=2.0, h=20, w=20):
        self._sid = sid
        self._frame_ids = frame_ids
        ys, xs = np.mgrid[0:h, 0:w]
        pts = np.stack([xs.astype(float), ys.astype(float),
                        np.full((h, w), z)], axis=-1)
        self.pointclouds = [pts.copy() for _ in frame_ids]
        self.conf_masks = [np.ones((h, w), dtype=float)
                           for _ in frame_ids]
        self._h, self._w = h, w

    def get_id(self):
        return self._sid

    def get_frame_ids(self):
        return self._frame_ids

    def get_lc_status(self):
        return False

    def get_conf_threshold(self):
        return 0.5

    def get_frame_at_index(self, index):
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def get_all_poses_world(self, graph):
        return np.stack([np.eye(4) for _ in self._frame_ids])


class _FakeGraph:
    def get_homography(self, _index):
        return np.eye(4)


class _FakeMap:
    def __init__(self, submaps):
        self._submaps = {s.get_id(): s for s in submaps}

    def ordered_submaps_by_key(self):
        return [self._submaps[k] for k in sorted(self._submaps)]

    def get_submap(self, sid):
        return self._submaps[sid]


class _FakeSolver:
    def __init__(self, submaps):
        self.map = _FakeMap(submaps)
        self.graph = _FakeGraph()


class _MockEmbedder:
    def encode(self, texts):
        return np.stack([np.ones(8, dtype=np.float32) for _ in texts])


class _MockStore:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, _emb, k=10):
        return self._hits[:k]


class _MockPointer:
    """探索 grounding 对每个召回帧直接 point；到达复核仍可用 verify。"""

    def __init__(self):
        self.verify_calls = []

    def verify_frame(self, pil, text, frame_key=None):
        self.verify_calls.append(frame_key)
        ok = frame_key == "frame_2"
        return {"match": ok, "checked_attributes": [],
                "confidence": 0.9 if ok else 0.1, "reason": ""}

    def point(self, pil, text, frame_key=None):
        return [{"pixel": (10.0, 10.0), "confidence": 0.8,
                 "bbox": [5, 5, 15, 15]}]


def _make_server():
    srv = MappingServer.__new__(MappingServer)
    srv.retrieve_top_k = 10
    srv.point_patch = 11
    srv.embedder = _MockEmbedder()
    srv.pointer = _MockPointer()
    srv.caption_store = _MockStore([
        {"frame_id": 1, "caption": "a kitchen with sink", "score": 0.7,
         "pose": np.eye(4).tolist()},
        {"frame_id": 2, "caption": "a gray fabric sofa", "score": 0.6,
         "pose": np.eye(4).tolist()},
    ])
    srv.solver = _FakeSolver([_FakeSubmap(0, [1, 2])])
    srv.solver_lock = threading.Lock()
    srv.data_lock = threading.Lock()
    srv.gpu_lock = threading.Lock()
    srv.diag_lock = threading.Lock()
    srv._current_episode = None
    srv._diag_fp = None
    srv._diag_frame_dir = None
    srv._candidate_seq = 0
    srv._ground_candidates = {}
    return srv


# ----------------------------------------------------------------------
def test_semantic_ground_end_to_end():
    srv = _make_server()
    out = srv._ground_object_semantic("gray fabric sofa", top_k=5)
    results = out["results"]
    # K 被 NAV_RETRIEVE_TOP_K 召回优先放大，但 store 只有 2 条
    found = [r for r in results if r.get("found")]
    assert len(found) == 2                         # 探索阶段不预先 VQA 拒绝
    assert srv.pointer.verify_calls == []
    hit = found[1]
    assert hit["frame_id"] == 2
    # patch 深度采样：先 conf 过滤再取中位数 -> 平面网格 (10, 10, 2)
    np.testing.assert_allclose(hit["point"], [10.0, 10.0, 2.0], atol=0.6)
    for key in ("candidate_id", "point_score", "bbox", "num_points", "point"):
        assert key in hit
    assert hit["text"] == "a gray fabric sofa"
    assert hit["point_score"] == pytest.approx(0.8)
    assert srv._ground_candidates[hit["candidate_id"]]["point_score"] == \
        pytest.approx(0.8)


def test_semantic_ground_disabled_without_models():
    srv = _make_server()
    srv.pointer = None
    out = srv._ground_object_semantic("sofa", top_k=5)
    assert out["results"] == []
    assert "error" in out


def test_resolve_point_candidate_refreshes():
    srv = _make_server()
    out = srv._ground_object_semantic("sofa", top_k=5)
    cid = [r for r in out["results"] if r.get("found")][0]["candidate_id"]
    n_before = len(srv._ground_candidates)
    resolved = srv.resolve_candidate(cid)
    assert resolved["found"] is True
    np.testing.assert_allclose(resolved["point"], [10.0, 10.0, 2.0],
                               atol=0.6)
    # 重解析不重复注册候选
    assert len(srv._ground_candidates) == n_before


def test_retrieve_captions_refreshes_pose():
    srv = _make_server()
    out = srv.retrieve_captions("sofa", top_k=5)
    assert len(out["results"]) == 2
    assert out["results"][0]["frame_id"] == 1       # score 降序
    assert "caption" in out["results"][0]
    # 位姿由当前图优化刷新（假世界 = 单位阵）
    np.testing.assert_allclose(out["results"][0]["pose"], np.eye(4))


def test_locate_frame():
    srv = _make_server()
    assert srv._locate_frame(2) == (0, 1)
    assert srv._locate_frame(999) is None


def test_ground_frame_semantic():
    srv = _make_server()
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    # _MockPointer 只对 frame_key=frame_2 通过；当前帧无 frame_key -> 拒绝
    out = srv._ground_frame_semantic(rgb, "sofa")
    assert out["found"] is False
    assert "verify" in out

    class _YesPointer(_MockPointer):
        def verify_frame(self, pil, text, frame_key=None):
            return {"match": True, "checked_attributes": [],
                    "confidence": 0.9, "reason": ""}

    srv.pointer = _YesPointer()
    out = srv._ground_frame_semantic(rgb, "sofa")
    assert out["found"] is True
    assert out["score"] == pytest.approx(0.8)
    assert out["target_pixels"] == pytest.approx(10.0)
    assert out["points"][0]["pixel"] == [10.0, 10.0]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
