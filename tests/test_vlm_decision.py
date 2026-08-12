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
    fake = _FakePost([{"action": "EXPLORE", "confidence": 0.85}])
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
    assert called == []


if __name__ == "__main__":
    test_openai_compatible_unified_call()
    test_disabled_client_is_network_free()
    print("VLM decision tests passed")
