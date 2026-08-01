"""Dependency-light tests for VLM schemas, prompts, images, and fallbacks."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision import prompts
from decision.types import TargetSpec
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


def test_prompt_contract():
    system = prompts.SYSTEM_PROMPT.lower()
    assert "rgb" in system
    assert "no ground-truth depth" in system
    assert "never invent coordinates" in system
    assert "strict json" in system
    candidate = prompts.candidate_prompt(
        "Find the red chair", {"grounding_query": "red chair"}, {}, [])
    assert "navigate" in candidate.lower()
    assert "explore" in candidate.lower()
    assert "red-mask" in candidate.lower()


def test_openai_compatible_event_calls():
    fake = _FakePost([
        {
            "grounding_query": "red fabric chair",
            "target_description": "a red chair with fabric upholstery",
            "confidence": 0.9,
        },
        {
            "decision": "navigate",
            "candidate_id": "c2",
            "rejected_candidate_ids": ["c1"],
            "exploration_hint": "none",
            "confidence": 0.85,
            "reason": "candidate c2 has the requested red upholstery",
        },
    ])
    client = VLMDecisionClient(
        api_url="http://vlm.local/v1", model="test-vlm", enabled=True,
        post_fn=fake)
    spec = client.parse_instruction("Find the red chair", "any", None)
    assert spec.grounding_query == "red fabric chair"

    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    decision = client.choose_candidate(
        "Find the red chair", spec, {"step": 40}, rgb,
        [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
        {"c1": b"jpeg-one", "c2": b"jpeg-two"})
    assert decision.decision == "navigate"
    assert decision.candidate_id == "c2"
    assert decision.rejected_candidate_ids == ["c1"]

    url, kwargs = fake.calls[-1]
    assert url == "http://vlm.local/v1/chat/completions"
    content = kwargs["json"]["messages"][1]["content"]
    labels = [x.get("text") for x in content if x.get("type") == "text"]
    assert "Image label: current_observation" in labels
    assert "Image label: candidate_0" in labels
    assert "Image label: candidate_1" in labels
    assert sum(x.get("type") == "image_url" for x in content) == 3


def test_disabled_client_is_network_free():
    called = []

    def fail_if_called(*args, **kwargs):
        called.append(True)
        raise AssertionError("network should not be called")

    client = VLMDecisionClient(
        api_url="http://unused", model="unused", enabled=False,
        post_fn=fail_if_called)
    assert client.parse_instruction("find a chair", "any", None) is None
    assert called == []


if __name__ == "__main__":
    test_prompt_contract()
    test_openai_compatible_event_calls()
    test_disabled_client_is_network_free()
    print("VLM decision tests passed")
