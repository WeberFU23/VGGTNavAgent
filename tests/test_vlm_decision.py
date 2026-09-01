"""Dependency-light tests for the unified VLM transport."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.vlm import VLMDecisionClient


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakePost:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        output = self.outputs.pop(0)
        content = "```json\n" + json.dumps(output) + "\n```"
        return _Response({
            "choices": [{"message": {"content": content}}],
        })


def test_openai_compatible_unified_call():
    fake = _FakePost([{"action": "EXPLORE", "reason": "keep mapping"}])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake)
    decision = client.agentic_chat(
        "Event: world_state_updated", [("candidate_instance_1", b"jpeg-one")])
    assert decision["action"] == "EXPLORE"

    url, kwargs = fake.calls[-1]
    assert url == "http://vlm.local/v1/chat/completions"
    content = kwargs["json"]["messages"][1]["content"]
    labels = [x.get("text") for x in content if x.get("type") == "text"]
    assert "Image label: candidate_instance_1" in labels
    assert sum(x.get("type") == "image_url" for x in content) == 1


def test_disabled_client_is_network_free():
    called = []

    def fail_if_called(*args, **kwargs):
        called.append(True)
        raise AssertionError("network should not be called")

    client = VLMDecisionClient(
        api_url="http://unused", model="unused", enabled=False,
        post_fn=fail_if_called)
    assert client.agentic_chat("event", []) is None
    assert client.chat_text("describe") is None
    assert called == []


def test_decision_live_probe_requires_valid_json_generation():
    fake = _FakePost([{"ok": True}])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake)
    assert client.probe() is True
    assert fake.calls[0][1]["json"]["max_tokens"] == 16


def test_png_topdown_map_uses_png_data_uri():
    fake = _FakePost([{"action": "END_ADJUST"}])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake)
    png = b"\x89PNG\r\n\x1a\nmap-bytes"
    client.agentic_chat("Event: adjustment", [
        ("current_observation", b"jpeg"), ("topdown_map", png)])
    content = fake.calls[-1][1]["json"]["messages"][1]["content"]
    urls = [part["image_url"]["url"] for part in content
            if part.get("type") == "image_url"]
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert urls[1].startswith("data:image/png;base64,")


def test_image_budget_keeps_core_context_and_newest_tool_image():
    previous = os.environ.get("NAV_VLM_MAX_IMAGES")
    os.environ["NAV_VLM_MAX_IMAGES"] = "4"
    try:
        parts = VLMDecisionClient._image_parts([
            ("current_observation", b"current"),
            ("topdown_map", b"map"),
            ("selected_candidate", b"candidate"),
            ("tool_frame_10_rgb", b"old-tool"),
            ("tool_frame_11_rgb", b"new-tool"),
        ])
    finally:
        if previous is None:
            os.environ.pop("NAV_VLM_MAX_IMAGES", None)
        else:
            os.environ["NAV_VLM_MAX_IMAGES"] = previous
    assert [part[0] for part in parts] == [
        "current_observation", "topdown_map", "selected_candidate",
        "tool_frame_11_rgb"]


def test_entity_image_budget_never_drops_new_observation():
    previous = os.environ.get("NAV_VLM_MAX_IMAGES")
    os.environ["NAV_VLM_MAX_IMAGES"] = "2"
    try:
        parts = VLMDecisionClient._image_parts([
            ("new_observation", b"new"),
            ("candidate_instance_1", b"old-1"),
            ("candidate_instance_2", b"old-2"),
        ])
    finally:
        if previous is None:
            os.environ.pop("NAV_VLM_MAX_IMAGES", None)
        else:
            os.environ["NAV_VLM_MAX_IMAGES"] = previous
    assert [part[0] for part in parts] == [
        "new_observation", "candidate_instance_2"]


def test_chat_text_returns_plain_text_without_json_mode():
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return _Response({"choices": [{"message": {
            "content": "a red ceramic cup beside the sink"}}]})

    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=post)
    text = client.chat_text("describe the marked object",
                            [("pointing_overlay", b"jpeg")])
    assert text == "a red ceramic cup beside the sink"
    payload = calls[0]["json"]
    assert "response_format" not in payload      # 自由文本不开 JSON mode
    labels = [x.get("text") for x in payload["messages"][1]["content"]
              if x.get("type") == "text"]
    assert "Image label: pointing_overlay" in labels


def test_vlm_trace_contains_raw_and_parsed_outputs(tmp_path):
    path = tmp_path / "vlm_calls.jsonl"
    fake = _FakePost([{"action": "EXPLORE", "reason": "continue"}])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake, trace_path=str(path))
    client.set_trace_context(episode="ep1", step=12)
    assert client.agentic_chat("event", [("current", b"jpeg")])["action"] \
        == "EXPLORE"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["parsed_output"]["action"] == "EXPLORE"
    assert "choices" in record["raw_response"]
    assert record["context"] == {"episode": "ep1", "step": 12}
    assert record["images"][0]["label"] == "current"


def test_vlm_trace_saves_exact_transmitted_images(tmp_path):
    trace_path = tmp_path / "vlm_calls.jsonl"
    image_dir = tmp_path / "vlm_inputs"
    fake = _FakePost([{"action": "END_ADJUST"}])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake, trace_path=str(trace_path), image_dir=str(image_dir))
    client.set_trace_context(episode="scene/ep1", step=7)
    topdown = b"\x89PNG\r\n\x1a\nexact-map-bytes"
    client.agentic_chat("event", [
        ("current_observation", b"jpeg-bytes"),
        ("topdown_map", topdown),
    ])

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    topdown_meta = next(
        item for item in record["images"] if item["label"] == "topdown_map")
    saved = topdown_meta["saved_path"]
    assert os.path.basename(saved).endswith("topdown_map.png")
    with open(saved, "rb") as fp:
        assert fp.read() == topdown
    assert topdown_meta["sha1"] == __import__("hashlib").sha1(topdown).hexdigest()


if __name__ == "__main__":
    test_openai_compatible_unified_call()
    test_disabled_client_is_network_free()
    test_png_topdown_map_uses_png_data_uri()
    test_chat_text_returns_plain_text_without_json_mode()
    print("VLM decision tests passed")
