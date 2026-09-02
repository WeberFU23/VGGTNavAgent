"""实例化去重（3m 预筛 + resolve_duplicate 裁决）回归测试。"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent


def _obs(step=100):
    return SimpleNamespace(
        step_count=step, goal_text="Find leather chairs",
        target_mode="many", target_count=3,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=None)


def _make_agent():
    agent = NavAgent()
    agent.target_text = "leather chair"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    agent._last_observation = _obs()
    return agent


def _hit(point, candidate_id, frame_id=10):
    return {"point": list(point), "found": True, "frame_id": frame_id,
            "candidate_id": candidate_id, "pixel": [100.0, 100.0],
            "bbox": None, "point_score": 1.0, "text": "leather chair"}


def test_instantiation_without_neighbors_creates_instance():
    agent = _make_agent()
    changed = agent._ingest_semantic_hits(
        _obs(), [_hit([1.0, 2.0, 0.0], "c1")], select=False)
    assert len(changed) == 1 and changed[0]["is_new"] is True
    assert len(agent.memory.nodes) == 1
    assert agent._last_dup_reviews == []


def test_instantiation_near_existing_suspends_for_review():
    agent = _make_agent()
    agent.memory.add([1.0, 2.0, 0.0], "dark leather chair")
    changed = agent._ingest_semantic_hits(
        _obs(), [_hit([1.4, 2.0, 0.0], "c2")], select=False)
    assert changed == []
    assert len(agent.memory.nodes) == 1  # 没有新建实例
    assert len(agent._last_dup_reviews) == 1
    review = agent._last_dup_reviews[0]
    assert review["neighbors"][0]["dist_m"] == 0.4
    # 挂起的 observation 不可导航
    assert agent.memory.instance_for_observation(
        review["observation_id"]) is None


def test_resolve_duplicate_new_creates_instance():
    agent = _make_agent()
    agent.memory.add([1.0, 2.0, 0.0], "dark leather chair")
    agent._ingest_semantic_hits(_obs(), [_hit([1.4, 2.0, 0.0], "c2")],
                                select=False)
    oid = agent._last_dup_reviews[0]["observation_id"]
    out = agent._tool_resolve_duplicate(oid, "NEW", text="red leather chair")
    assert out["resolved"] == "new"
    assert len(agent.memory.nodes) == 2
    # 重复裁决被拒
    again = agent._tool_resolve_duplicate(oid, "NEW")
    assert "error" in again


def test_resolve_duplicate_merges_into_existing():
    agent = _make_agent()
    node = agent.memory.add([1.0, 2.0, 0.0], "dark leather chair")
    agent._ingest_semantic_hits(_obs(), [_hit([1.4, 2.0, 0.0], "c2")],
                                select=False)
    oid = agent._last_dup_reviews[0]["observation_id"]
    out = agent._tool_resolve_duplicate(oid, "DUPLICATE", duplicate_of=node.iid)
    assert out["resolved"] == "duplicate" and out["instance_id"] == node.iid
    assert len(agent.memory.nodes) == 1
    assert len(node.observation_ids) == 2


def test_resolve_duplicate_rejects_unknown_target():
    agent = _make_agent()
    node = agent.memory.add([1.0, 2.0, 0.0], "dark leather chair")
    agent._ingest_semantic_hits(_obs(), [_hit([1.4, 2.0, 0.0], "c2")],
                                select=False)
    oid = agent._last_dup_reviews[0]["observation_id"]
    out = agent._tool_resolve_duplicate(oid, "DUPLICATE", duplicate_of=999)
    assert "error" in out
    assert agent.memory.instance_for_observation(oid) is None
    bad = agent._tool_resolve_duplicate(oid, "MAYBE")
    assert "error" in bad
