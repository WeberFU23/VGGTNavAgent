"""缺陷修复回归：adjustment 区域预算 key、GOTO_INSTANCE 规划失败计数与
重选冷却、挂起提案刷屏抑制。"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent
from decision import DecisionResult


def _obs(step=100):
    return SimpleNamespace(
        step_count=step, goal_text="Find a gray fabric sofa",
        target_mode="any", target_count=None,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=None)


def _make_agent():
    agent = NavAgent()
    agent.target_text = "gray fabric sofa"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def _hit():
    return {"found": True, "point": [1.05, 2.02, 0.0],
            "text": "a gray fabric sofa", "frame_id": 6,
            "candidate_id": "c6", "bbox": [10, 10, 50, 50],
            "point_score": 0.9}


# --- 缺陷 2：跨 session 累计预算已移除，同一区域允许多次调整 -------------

def test_start_adjustment_not_limited_by_cross_session_budget():
    agent = _make_agent()
    # 不再存在任何跨 session 预算状态
    assert not hasattr(agent, "_adjustment_budgets")
    # 转向连击防护退化为 per-session 计数
    assert agent._adjust_turn_streak == 0
    agent._adjust_turn_streak = 2
    agent._adjust_last_turn = "TURN_LEFT"
    agent._build_decider_input = lambda obs, **kwargs: ({}, None)
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"rgb")
    agent.decision_loop = SimpleNamespace(
        decide=lambda *a, **k: DecisionResult("TURN_LEFT"), logger=None)
    abandoned = []
    agent._abandon_adjustment_target = lambda obs, reason: (
        abandoned.append(reason) or None)
    agent._adjustment_action(_obs())
    # per-session 转向连击防护仍然生效
    assert abandoned == ["repeated_same_turn"]


# --- 缺陷 3：GOTO_INSTANCE 规划失败计数 + 冷却，不再一次失败永久丢弃 -----

def test_goto_instance_plan_failure_counts_and_cools_down():
    agent = _make_agent()
    node = agent.instance_store.add([3, 4, 0], "gray fabric sofa")
    agent._plan_to_target = lambda obs: False

    ok = agent._apply_decider_steering(
        _obs(step=100), DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert ok is False
    assert agent.mode == "explore"
    assert agent.target_instance_id is None
    # 实例保留在池中，未进 unreachable
    assert node.iid not in agent._unreachable_instance_ids
    assert agent._instance_plan_failures[node.iid]["count"] == 1

    # 冷却期内重选被拒绝，计数不变
    ok = agent._apply_decider_steering(
        _obs(step=110), DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert ok is False
    assert agent._instance_plan_failures[node.iid]["count"] == 1
    assert any("ignored" in e for e in agent._events)

    # 冷却结束后可重试；达到上限才标记 unreachable
    limit = agent.instance_plan_fail_limit
    for i in range(limit - 1):
        agent._apply_decider_steering(
            _obs(step=200 + i * 100),
            DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert agent._instance_plan_failures[node.iid]["count"] == limit
    assert node.iid in agent._unreachable_instance_ids


def test_goto_instance_successful_plan_resets_failures():
    agent = _make_agent()
    node = agent.instance_store.add([3, 4, 0], "gray fabric sofa")
    agent._plan_to_target = lambda obs: False
    agent._apply_decider_steering(
        _obs(step=100), DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert agent._instance_plan_failures[node.iid]["count"] == 1
    agent._plan_to_target = lambda obs: True
    ok = agent._apply_decider_steering(
        _obs(step=200), DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert ok is True
    assert agent.mode == "nav"
    assert node.iid not in agent._instance_plan_failures


# --- 缺陷 5：同一挂起观测重复命中不再刷屏 ---------------------------------

def test_pending_proposal_reingest_is_silent():
    agent = _make_agent()
    agent.instance_store.add(
        [1.0, 2.0, 0.0], "nearby chair", frame_id=4, candidate_id="c4")
    obs = SimpleNamespace(step_count=50)
    hit = _hit()
    agent._ingest_semantic_hits(obs, [hit], select=False)
    assert agent._proposal_queue["c6"]["status"] == "duplicate_review"
    first_logs = [e for e in agent._events if "retained as" in e]
    assert len(first_logs) == 1
    # 同一证据反复命中：静默刷新，不再产生日志、不再重复挂起
    for step in (60, 70, 80):
        agent._ingest_semantic_hits(
            SimpleNamespace(step_count=step), [dict(hit)], select=False)
    logs = [e for e in agent._events if "retained as" in e]
    assert len(logs) == 1
    assert agent._proposal_queue["c6"]["step"] == 80
    assert agent._last_dup_reviews == []
    assert len(agent.instance_store.nodes) == 1
    assert len(agent.instance_store.observations) == 2
