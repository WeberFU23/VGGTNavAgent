"""图像目标（image-goal）支持回归测试。

覆盖：模式检测与收图、冷启动描述与失败回退、决策附件只含未找到目标、
goal_index 记账链（实例化 -> 实例 -> report 撤下）、description 模式不受影响。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent
from decision import prompts


class _FakeVLM:
    def __init__(self, description_reply=None):
        self.description_reply = description_reply
        self.chat_calls = []

    def encode_rgb(self, rgb):
        return b"jpeg-bytes"

    def chat_text(self, prompt, images=None, max_tokens=None):
        self.chat_calls.append((prompt, images))
        return self.description_reply


def _obs(step=0, goal_type="description", goal_images=None):
    return SimpleNamespace(
        step_count=step, goal_text="Find the target",
        target_mode="all", target_count=None,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=None,
        goal_type=goal_type, goal_images=goal_images)


def _make_agent(vlm=None):
    agent = NavAgent()
    agent.vlm = vlm or _FakeVLM()
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    return agent


def _goal_rgb(seed):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = seed
    return img


def test_image_mode_detection_and_capture():
    vlm = _FakeVLM("goal_image_0: a round wall clock\n"
                   "goal_image_1: a purple armchair")
    agent = _make_agent(vlm)
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    assert agent._image_goal_mode is True
    assert len(agent._goal_images) == 2
    assert agent._goal_descriptions == ["a round wall clock",
                                        "a purple armchair"]
    assert len(vlm.chat_calls) == 1
    # 幂等：第二次 capture 不再收图也不再调 VLM
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(3)]))
    assert len(agent._goal_images) == 2
    assert len(vlm.chat_calls) == 1


def test_description_mode_untouched():
    vlm = _FakeVLM("should not be called")
    agent = _make_agent(vlm)
    agent._capture_goal_images(_obs(goal_type="description",
                                    goal_images=[_goal_rgb(1)]))
    assert agent._image_goal_mode is False
    assert agent._goal_images == []
    assert vlm.chat_calls == []
    assert agent._goal_images_payload() == []


def test_cold_start_description_fallback():
    vlm = _FakeVLM(None)  # API 失败
    agent = _make_agent(vlm)
    obs = _obs(goal_type="image", goal_images=[_goal_rgb(1)])
    agent._last_observation = obs
    agent._capture_goal_images(obs)
    assert agent._goal_descriptions == ["Find the target"]


def test_payload_excludes_found_goals():
    agent = _make_agent(_FakeVLM("goal_image_0: clock\ngoal_image_1: chair"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    payload = agent._goal_images_payload()
    assert [label for label, _ in payload] == ["goal_image_0",
                                               "goal_image_1"]
    agent._goal_found.add(0)
    payload = agent._goal_images_payload()
    assert [label for label, _ in payload] == ["goal_image_1"]


def _hit(point, candidate_id, goal_index=None):
    return {"point": list(point), "found": True, "frame_id": 10,
            "candidate_id": candidate_id, "pixel": [100.0, 100.0],
            "bbox": None, "point_score": 1.0, "text": "target",
            "goal_index": goal_index}


def test_goal_index_bookkeeping_and_report():
    agent = _make_agent(_FakeVLM("goal_image_0: clock"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1)]))
    obs = _obs(step=100)
    agent._last_observation = obs
    changed = agent._ingest_semantic_hits(
        obs, [_hit([1.0, 2.0, 0.0], "c1", goal_index=0)], select=False)
    iid = changed[0]["instance_id"]
    assert agent._instance_goal_index[iid] == 0
    # report 后目标撤下
    agent.target_instance_id = iid
    action = agent._report_found(iid)
    from benchmark_api import Action
    assert action == int(Action.TARGET_FOUND)
    assert agent._goal_found == {0}
    assert agent._goal_images_payload() == []


def test_goal_index_ignored_in_description_mode():
    agent = _make_agent()
    agent._last_observation = _obs(step=100)
    changed = agent._ingest_semantic_hits(
        _obs(step=100), [_hit([1.0, 2.0, 0.0], "c1", goal_index=0)],
        select=False)
    assert changed[0]["instance_id"] not in agent._instance_goal_index


def test_merge_migrates_goal_index():
    agent = _make_agent(_FakeVLM("goal_image_0: clock"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1)]))
    obs = _obs(step=100)
    agent._last_observation = obs
    keep = agent.memory.add([0.0, 0.0, 0.0], "old clock")
    changed = agent._ingest_semantic_hits(
        obs, [_hit([5.0, 5.0, 0.0], "c1", goal_index=0)], select=False)
    drop_id = changed[0]["instance_id"]
    out = agent._tool_merge_instances(keep.iid, drop_id)
    assert out["into"] == keep.iid
    assert agent._instance_goal_index[keep.iid] == 0
    assert drop_id not in agent._instance_goal_index


def test_prompt_image_section_only_in_image_mode():
    state = {"task": {"goal": "g", "mode": "all", "found": 0,
                      "expected": None}}
    text = prompts.build_decision_prompt("world_state_updated", state, 15)
    assert "Image-goal mode" not in text
    state_img = {"task": {"goal": "g", "mode": "all", "found": 0,
                          "expected": None, "goal_type": "image",
                          "goal_descriptions": ["a clock"],
                          "goals_unfound": [0], "goals_total": 1}}
    text = prompts.build_decision_prompt("world_state_updated", state_img, 15)
    assert "Image-goal mode" in text
    assert "goal_index=N" in text
