"""vllm_client 单元测试：mock HTTP，不依赖真实 vLLM 服务。

    python tests/test_vllm_client.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from mapping.vllm_client import Priority, VLLMError, VLLMGateway


class _FakeResponse:
    def __init__(self, text="ok", status_code=200, payload=None):
        self._text = text
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is not None:
            return self._payload
        return {"choices": [{"message": {"content": self._text}}]}


def test_healthcheck_requires_requested_model():
    gw = _gateway(lambda *a, **k: _FakeResponse(), start_worker=False)
    get = lambda *a, **k: _FakeResponse(
        payload={"data": [{"id": "molmo"}]})
    assert gw.healthcheck("molmo", get_fn=get)["models"] == ["molmo"]
    with pytest.raises(VLLMError, match="not loaded"):
        gw.healthcheck("qwen", get_fn=get)


def test_live_chat_probe_detects_quota_error_without_retries():
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        response = _FakeResponse(status_code=400)
        response.text = "insufficient balance"
        return response

    gw = _gateway(post, start_worker=False, max_retries=5)
    with pytest.raises(VLLMError, match="insufficient balance"):
        gw.probe_chat("caption-model", timeout=1)
    assert len(calls) == 1


def test_live_chat_probe_accepts_nonempty_generation():
    gw = _gateway(
        lambda *args, **kwargs: _FakeResponse("OK"), start_worker=False)
    result = gw.probe_chat("caption-model", timeout=1)
    assert result["model"] == "caption-model"
    assert result["output"] == "OK"


def _gateway(post_fn, **kwargs):
    kwargs.setdefault("backoff_base", 0.0)
    return VLLMGateway("http://fake/v1", post_fn=post_fn, **kwargs)


def test_priority_queue_order():
    gw = _gateway(lambda *a, **k: _FakeResponse(), start_worker=False)
    gw.chat_async("m", "caption task", kind="caption", priority=Priority.CAPTION)
    gw.chat_async("m", "decision task", kind="decision", priority=Priority.DECISION)
    gw.chat_async("m", "pointing task", kind="pointing", priority=Priority.POINTING)
    order = []
    while not gw._queue.empty():
        prio, _seq, item = gw._queue.get()
        order.append((prio, item[0]["messages"][0]["content"][0]["text"]))
    assert [p for p, _ in order] == [
        Priority.DECISION, Priority.POINTING, Priority.CAPTION]
    assert order[0][1] == "decision task"


def test_cache_hit_skips_http():
    calls = []

    def post(*a, **k):
        calls.append(a)
        return _FakeResponse("cached")

    gw = _gateway(post)
    r1 = gw.chat("m", "p", kind="caption", cache_key="frame_1")
    r2 = gw.chat("m", "p", kind="caption", cache_key="frame_1")
    assert r1 == r2 == "cached"
    assert len(calls) == 1


def test_trace_records_raw_output_and_cache_hit():
    traces = []
    gw = _gateway(lambda *a, **k: _FakeResponse("raw caption"),
                  trace_fn=traces.append)
    assert gw.chat("caption-model", "describe", kind="caption",
                   cache_key="frame_7") == "raw caption"
    assert gw.chat("caption-model", "describe", kind="caption",
                   cache_key="frame_7") == "raw caption"
    assert len(traces) == 2
    assert traces[0]["raw_output"] == "raw caption"
    assert traces[0]["kind"] == "caption" and not traces[0]["cache_hit"]
    assert traces[1]["cache_hit"] is True


def test_cache_key_distinguishes_kind_and_frame():
    calls = []

    def post(*a, **k):
        calls.append(a)
        return _FakeResponse("x")

    gw = _gateway(post)
    gw.chat("m", "p", kind="caption", cache_key="frame_1")
    gw.chat("m", "p", kind="pointing", cache_key="frame_1")
    gw.chat("m", "p", kind="caption", cache_key="frame_2")
    assert len(calls) == 3


def test_clear_cache_forces_new_http_request():
    calls = []

    def post(*a, **k):
        calls.append(a)
        return _FakeResponse("x")

    gw = _gateway(post)
    gw.chat("m", "p", cache_key="frame_1")
    gw.clear_cache()
    gw.chat("m", "p", cache_key="frame_1")
    assert len(calls) == 2


def test_retry_then_success():
    calls = []

    def post(*a, **k):
        calls.append(a)
        if len(calls) < 3:
            raise ConnectionError("refused")
        return _FakeResponse("recovered")

    gw = _gateway(post, max_retries=3)
    assert gw.chat("m", "p") == "recovered"
    assert len(calls) == 3


def test_retry_exhausted_raises():
    def post(*a, **k):
        raise ConnectionError("down")

    gw = _gateway(post, max_retries=2)
    with pytest.raises(VLLMError):
        gw.chat("m", "p")


def test_404_model_not_found_no_retry():
    calls = []

    def post(*a, **k):
        calls.append(a)
        return _FakeResponse(status_code=404)

    gw = _gateway(post, max_retries=3)
    with pytest.raises(VLLMError, match="404"):
        gw.chat("m", "p")
    assert len(calls) == 1


def test_other_4xx_does_not_retry():
    calls = []

    def post(*a, **k):
        calls.append(a)
        return _FakeResponse(status_code=400)

    gw = _gateway(post, max_retries=3)
    with pytest.raises(VLLMError, match="400"):
        gw.chat("m", "p")
    assert len(calls) == 1


def test_empty_model_raises():
    gw = _gateway(lambda *a, **k: _FakeResponse())
    with pytest.raises(VLLMError):
        gw.chat("", "p")


def test_parse_json_strips_fence_and_prose():
    data = VLLMGateway.parse_json('prefix ```json\n{"a": 1}\n``` suffix')
    assert data == {"a": 1}
    with pytest.raises(VLLMError):
        VLLMGateway.parse_json("not json at all")


def test_async_result():
    gw = _gateway(lambda *a, **k: _FakeResponse("async-ok"))
    handle = gw.chat_async("m", "p")
    assert handle.result(timeout=10) == "async-ok"
    assert handle.done()


def test_closed_gateway_rejects_new_requests():
    gw = _gateway(lambda *a, **k: _FakeResponse())
    gw.close()
    gw.close()  # 幂等
    with pytest.raises(VLLMError, match="已关闭"):
        gw.chat_async("m", "p")


def test_image_payload_encoding():
    from PIL import Image
    captured = []

    def post(url, headers=None, json=None, timeout=None):
        captured.append(json)
        return _FakeResponse()

    gw = _gateway(post)
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    gw.chat("m", "p", [img, b"\xff\xd8rawjpeg"])
    content = captured[0]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    imgs = [c for c in content if c["type"] == "image_url"]
    assert len(imgs) == 2
    assert all(c["image_url"]["url"].startswith("data:image/jpeg;base64,")
               for c in imgs)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
