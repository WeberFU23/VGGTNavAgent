"""caption_store 单元测试：mock embedder 与 gateway，不依赖真实权重。

    python tests/test_caption_store.py
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.caption_store import (BGEM3Embedder, CaptionStore, CaptionWorker)


def _vec(seed, dim=8):
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _MockEmbedder:
    def encode(self, texts):
        return np.stack([_vec(abs(hash(t)) % 1000) for t in texts])


class _MockGateway:
    def __init__(self, reply="a gray fabric sofa next to a wooden table"):
        self.reply = reply
        self.calls = []

    def chat(self, model, prompt, images, *, kind, cache_key, priority,
             max_tokens):
        self.calls.append((model, kind, cache_key, priority))
        return self.reply


# ----------------------------------------------------------------------
# CaptionStore
# ----------------------------------------------------------------------
def test_add_and_retrieve_order():
    store = CaptionStore()
    store.add(1, None, "red chair", _vec(1))
    store.add(2, None, "blue sofa", _vec(2))
    store.add(3, None, "wooden table", _vec(3))
    results = store.retrieve(_vec(2), k=2)
    assert [r["frame_id"] for r in results][:1] == [2]
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["caption"] == "blue sofa"


def test_retrieve_empty_store():
    assert CaptionStore().retrieve(_vec(1), k=5) == []


def test_retrieve_k_capped_by_size():
    store = CaptionStore()
    store.add(1, None, "only frame", _vec(1))
    assert len(store.retrieve(_vec(1), k=10)) == 1


def test_persistence_roundtrip(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))
    store.set_episode("ep1")
    pose = np.eye(4, dtype=np.float32)
    store.add(10, pose, "caption a", _vec(1))
    store.add(11, pose, "caption b", _vec(2))
    store.save()

    restored = CaptionStore(persist_dir=str(tmp_path))
    assert restored.load("ep1") == 2
    assert restored.has(10) and restored.has(11)
    rec = [r for r in restored.records if r["frame_id"] == 10][0]
    assert np.allclose(rec["pose"], pose)
    # 检索结果与原始 store 一致
    assert (restored.retrieve(_vec(1), 1)[0]["frame_id"]
            == store.retrieve(_vec(1), 1)[0]["frame_id"])


def test_set_episode_saves_and_clears(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))
    store.set_episode("ep1")
    store.add(1, None, "ep1 caption", _vec(1))
    store.set_episode("ep2")           # 触发 ep1 落盘 + 清空
    assert len(store) == 0
    assert os.path.exists(
        tmp_path / "ep1" / "captions.jsonl")
    store.add(2, None, "ep2 caption", _vec(2))
    store.set_episode("ep1")           # 再切回：ep2 落盘
    assert os.path.exists(tmp_path / "ep2" / "captions.jsonl")


def test_embedder_requires_path(tmp_path):
    with pytest.raises(RuntimeError, match="NAV_EMBED_MODEL_PATH"):
        BGEM3Embedder("")
    with pytest.raises(RuntimeError, match="不存在"):
        BGEM3Embedder(str(tmp_path / "missing"))


# ----------------------------------------------------------------------
# CaptionWorker
# ----------------------------------------------------------------------
def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_worker_generates_caption(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))
    gw = _MockGateway()
    worker = CaptionWorker(gw, _MockEmbedder(), store, model="qwen-3b",
                           busy_fn=lambda: False)
    from PIL import Image
    worker.enqueue(42, Image.new("RGB", (8, 8)), np.eye(4))
    assert _wait_for(lambda: store.has(42))
    rec = store.records[0]
    assert "sofa" in rec["caption"]
    assert rec["pose"] is not None
    assert gw.calls[0][1] == "caption"
    worker.close()


def test_worker_yields_when_gpu_busy(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))
    gw = _MockGateway()
    busy = {"flag": True}
    worker = CaptionWorker(gw, _MockEmbedder(), store, model="qwen-3b",
                           busy_fn=lambda: busy["flag"])
    from PIL import Image
    worker.enqueue(7, Image.new("RGB", (8, 8)))
    time.sleep(0.2)
    assert not store.has(7)          # GPU 忙，让路
    busy["flag"] = False
    assert _wait_for(lambda: store.has(7))
    worker.close()


def test_worker_skips_duplicate_and_empty_model(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))
    gw = _MockGateway()
    worker = CaptionWorker(gw, _MockEmbedder(), store, model="qwen-3b",
                           busy_fn=lambda: False)
    from PIL import Image
    img = Image.new("RGB", (8, 8))
    worker.enqueue(5, img)
    assert _wait_for(lambda: store.has(5))
    worker.enqueue(5, img)           # 已有 caption，不再调用
    time.sleep(0.1)
    assert len(gw.calls) == 1
    worker.close()

    # model 为空：完全不工作（构造不报错，enqueue 静默忽略）
    worker2 = CaptionWorker(gw, _MockEmbedder(), store, model="",
                            busy_fn=lambda: False)
    worker2.enqueue(9, img)
    time.sleep(0.1)
    assert not store.has(9)
    worker2.close()


def test_worker_survives_gateway_error(tmp_path):
    store = CaptionStore(persist_dir=str(tmp_path))

    class _BadGateway:
        def chat(self, *a, **k):
            raise RuntimeError("vllm down")

    worker = CaptionWorker(_BadGateway(), _MockEmbedder(), store,
                           model="qwen-3b", busy_fn=lambda: False)
    from PIL import Image
    worker.enqueue(1, Image.new("RGB", (8, 8)))
    assert _wait_for(lambda: worker.errors >= 1)
    assert not store.has(1)
    worker.close()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
