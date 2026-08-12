"""Phase 4 单元测试：决策循环（schema 校验/工具循环/FINISH 硬条件/
规则回退/trace）+ 俯视地图渲染 + 决策状态组装。

    python tests/test_decider.py
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import navigator as nav
from agents.decision_state import build_world_state
from agents.map_render import render_topdown
from agents.nav_agent import NavAgent
from benchmark_api import Action
from decision import DecisionLoop, DecisionTraceLogger


def _state(mode="all", found=1, expected=None, unexplored=0.05,
           unresolved=0, instances=None, frontiers=None, anchors=None):
    return {
        "task": {"goal": "Find all baskets", "mode": mode,
                 "found": found, "expected": expected},
        "step": 400, "max_steps": 500,
        "instances": instances if instances is not None else [
            {"id": 1, "category": "basket", "status": "visited",
             "confidence": 0.9, "n_obs": 2, "dist_m": 1.5,
             "path_cost_m": 2.0}],
        "belief_anchors": anchors if anchors is not None else [],
        "frontiers": frontiers if frontiers is not None else [
            {"id": "f0", "dist_m": 3.0, "size": 12, "semantic_hint": 0.1}],
        "recent_events": ["reported TARGET_FOUND 'basket' (total 1)"],
        "older_events_total": 0,
        "termination": {"unexplored_ratio": unexplored,
                        "unresolved_anchor_count": unresolved,
                        "frontier_count": 1,
                        "recent_queries_without_new_candidate": 6},
    }


class _ScriptedChat:
    """按队列返回决策模型输出。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, images):
        self.calls.append((prompt, images))
        return self.replies.pop(0) if self.replies else None


# ----------------------------------------------------------------------
# schema 校验与重试
# ----------------------------------------------------------------------
def test_valid_goto_instance():
    chat = _ScriptedChat([{"action": "GOTO_INSTANCE", "target_id": "1",
                           "reason": "unvisited", "confidence": 0.8}])
    loop = DecisionLoop(chat)
    state = _state(instances=[{"id": 1, "status": "confirmed"}])
    result = loop.decide("instance_confirmed", state)
    assert result.action == "GOTO_INSTANCE"
    assert result.target_id == "1"
    assert result.validation == "ok"


def test_arrival_uses_unified_schema_and_event_actions():
    chat = _ScriptedChat([{"action": "REPORT_FOUND", "confidence": 0.9}])
    result = DecisionLoop(chat).decide("arrival", _state(), images=[b"rgb"])
    assert result.action == "REPORT_FOUND"
    assert chat.calls[0][1] == [b"rgb"]


def test_arrival_rejects_navigation_action():
    chat = _ScriptedChat([
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
        {"action": "SCAN", "confidence": 0.5},
    ])
    result = DecisionLoop(chat).decide("arrival", _state())
    assert result.action == "SCAN"
    assert "invalid for event" in chat.calls[1][0]


def test_visited_instance_rejected_then_retry():
    chat = _ScriptedChat([
        {"action": "GOTO_INSTANCE", "target_id": "1"},      # visited -> 拒绝
        {"action": "GOTO_FRONTIER", "target_id": "f0", "confidence": 0.5}])
    loop = DecisionLoop(chat)
    state = _state(instances=[{"id": 1, "status": "visited"}])
    result = loop.decide("ev", state)
    assert result.action == "GOTO_FRONTIER"
    assert len(chat.calls) == 2
    assert "rejected" in chat.calls[1][0]


def test_invalid_twice_falls_back():
    chat = _ScriptedChat([
        {"action": "TELEPORT"}, {"action": "GOTO_INSTANCE", "target_id": "zz"}])
    loop = DecisionLoop(chat)
    assert loop.decide("ev", _state()) is None


def test_model_unavailable_falls_back():
    loop = DecisionLoop(_ScriptedChat([]))       # chat 返回 None
    assert loop.decide("ev", _state()) is None


# ----------------------------------------------------------------------
# 工具循环
# ----------------------------------------------------------------------
def test_tool_loop_query_memory():
    tools_seen = []

    def query_memory(text):
        tools_seen.append(text)
        return [{"frame_id": 7, "caption": "a basket on the floor"}]

    chat = _ScriptedChat([
        {"tool_call": {"name": "query_memory", "text": "basket"}},
        {"action": "FINISH", "confidence": 0.9}])
    loop = DecisionLoop(chat, tools={"query_memory": query_memory})
    result = loop.decide("finish_check", _state())
    assert tools_seen == ["basket"]
    assert result.action == "FINISH"
    assert result.tool_calls == 1


def test_tool_loop_look_at_image():
    def look_at(frame_id):
        return b"\xff\xd8jpeg-bytes" if int(frame_id) == 7 else None

    chat = _ScriptedChat([
        {"tool_call": {"name": "look_at", "frame_id": 7}},
        {"action": "FINISH", "confidence": 0.9}])
    loop = DecisionLoop(chat, tools={"look_at": look_at})
    result = loop.decide("finish_check", _state())
    assert result.action == "FINISH"
    # 第二轮带上了工具返回的图像
    assert any(img == ("tool_keyframe", b"\xff\xd8jpeg-bytes")
               for img in chat.calls[1][1])


def test_tool_rounds_exhausted():
    chat = _ScriptedChat(
        [{"tool_call": {"name": "query_memory", "text": "x"}}] * 10)
    loop = DecisionLoop(chat, tools={"query_memory": lambda t: []},
                        max_tool_rounds=3)
    assert loop.decide("ev", _state()) is None


# ----------------------------------------------------------------------
# FINISH 硬条件
# ----------------------------------------------------------------------
def test_finish_hard_condition_pass():
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    result = DecisionLoop(chat).decide("finish_check", _state())
    assert result.action == "FINISH"


def test_finish_downgraded_with_unresolved_anchor():
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    result = DecisionLoop(chat).decide(
        "finish_check", _state(unresolved=2))
    assert result.action == "GOTO_FRONTIER"
    assert result.target_id == "f0"
    assert result.validation == "finish_downgraded"


def test_finish_downgraded_many_count_short():
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    state = _state(mode="many", found=1, expected=3)
    result = DecisionLoop(chat).decide("finish_check", state)
    assert result.action == "GOTO_FRONTIER"


def test_finish_downgraded_no_frontier():
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    state = _state(unexplored=0.5, frontiers=[])
    state["termination"]["frontier_count"] = 0
    result = DecisionLoop(chat).decide("finish_check", state)
    assert result.validation == "finish_downgraded_no_frontier"


def test_many_counting_hint_in_prompt():
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    DecisionLoop(chat).decide(
        "finish_check", _state(mode="many", found=1, expected=3))
    assert "Counting hint" in chat.calls[0][0]


# ----------------------------------------------------------------------
# trace 日志
# ----------------------------------------------------------------------
def test_trace_log_written(tmp_path):
    logger = DecisionTraceLogger(str(tmp_path / "trace.jsonl"))
    chat = _ScriptedChat([{"action": "FINISH", "confidence": 0.9}])
    loop = DecisionLoop(chat, logger=logger)
    loop.decide("finish_check", _state())
    lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "finish_check"
    assert rec["output"]["action"] == "FINISH"
    assert rec["validation"] == "ok"
    assert "input_summary" in rec and "step" in rec


# ----------------------------------------------------------------------
# 俯视地图渲染
# ----------------------------------------------------------------------
def _grid(h=30, w=40):
    free = np.zeros((h, w), dtype=bool)
    free[3:-3, 3:-3] = True
    obstacle = ~free                          # 边界全障碍 -> 无 unknown
    return nav.OccupancyGrid(1.0, np.array([0.0, 0.0]), free, obstacle)


def test_render_topdown_png():
    png = render_topdown(
        _grid(),
        trajectory=[(5, 5), (10, 10), (20, 15)],
        pose=(20, 15, 0.5),
        instances=[{"id": 1, "xy": (25, 20), "visited": False}],
        anchors=[{"id": "b0", "xy": (8, 22)}],
        frontiers=[{"id": "f0", "xy": (35, 25)}])
    assert png[:4] == b"\x89PNG"
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(png))
    assert img.size == (40 * 4, 30 * 4)


# ----------------------------------------------------------------------
# 决策状态组装
# ----------------------------------------------------------------------
def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0,
                                       actions=[])
    _poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (_poses, [0, 1, 2]))
    return agent


def test_build_world_state_ids_and_precompute():
    agent = _make_agent()
    node, _ = agent.memory.add_or_merge("basket", [3, 4, 0], 0.9,
                                        merge_dist=0.75)
    agent.ledger.add_observation("basket", [8, 8, 0], 0.6, merge_dist=0.75)
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    frontiers = [{"world": np.array([10.0, 2.0]), "size": 7}]
    state = build_world_state(agent, obs, grid=_grid(),
                              frontiers=frontiers)
    assert state["task"]["mode"] == "all"
    assert state["instances"][0]["id"] == node.iid
    assert state["instances"][0]["dist_m"] is not None
    assert state["instances"][0]["path_cost_m"] is not None   # A* 预计算
    assert state["belief_anchors"][0]["id"] == "b0"
    assert state["frontiers"][0]["id"] == "f0"
    assert state["termination"]["unexplored_ratio"] is not None


# ----------------------------------------------------------------------
# NavAgent 接线：四模式 mock 决策
# ----------------------------------------------------------------------
def _wire_decider(agent, replies):
    chat = _ScriptedChat(replies)
    agent.decision_loop = DecisionLoop(chat)
    agent.decider_mode = "vlm"
    return chat


def test_navagent_all_mode_finish_via_decider():
    agent = _make_agent()
    agent._reported_count = 1
    agent._no_hit_queries = 6
    agent.query_interval = 20
    agent.finish_frontier_patience = 3
    agent._frontier_empty_streak = 5
    agent.memory.add_or_merge("basket", [3, 4, 0], 0.9, merge_dist=0.75,
                              status="visited")
    agent.grid = _grid()                        # 全已知 -> unexplored 0
    _wire_decider(agent, [{"action": "FINISH", "confidence": 0.9}])
    obs = SimpleNamespace(step_count=400, max_steps=500,
                          goal_text="Find all baskets")
    assert agent._should_finish(obs) is True


def test_navagent_all_mode_finish_downgraded_by_anchor():
    agent = _make_agent()
    agent._reported_count = 1
    agent.query_interval = 20
    agent.finish_frontier_patience = 3
    agent._frontier_empty_streak = 5
    agent.ledger.add_observation("basket", [8, 8, 0], 0.7, merge_dist=0.75)
    agent.grid = _grid()
    chat = _wire_decider(agent, [{"action": "FINISH", "confidence": 0.9}])
    obs = SimpleNamespace(step_count=400, max_steps=500,
                          goal_text="Find all baskets")
    assert agent._should_finish(obs) is False   # 有未复核锚点 -> 降级


def test_navagent_many_mode_finish_is_deterministic():
    agent = _make_agent()
    agent._target_mode = "many"
    agent._target_count = 2
    agent._reported_count = 2
    _wire_decider(agent, [])                    # 决策层不应被调用
    obs = SimpleNamespace(step_count=100, max_steps=500,
                          goal_text="Find two baskets")
    assert agent._should_finish(obs) is True


def test_navagent_decider_next_goto_instance():
    agent = _make_agent()
    agent._target_mode = "any"
    agent._reported_count = 1
    node, _ = agent.memory.add_or_merge("basket", [6, 6, 0], 0.9,
                                        merge_dist=0.75)
    _wire_decider(agent, [
        {"action": "GOTO_INSTANCE", "target_id": str(node.iid),
         "confidence": 0.8}])
    obs = SimpleNamespace(step_count=200, max_steps=500,
                          goal_text="Find a basket",
                          rgb=np.zeros((48, 64, 3), dtype=np.uint8),
                          episode_id="ep", previous_action=None)
    action = agent._decider_next(obs, "instance_confirmed")
    assert action is not None
    # 目标点已被设置（规划因无 SLAM 位姿失败则保持 explore，均合法）
    assert agent.target_point is not None
    assert any("GOTO_INSTANCE" in e for e in agent._events)


def test_navagent_decider_none_falls_back_to_rules():
    agent = _make_agent()
    agent._target_mode = "any"
    _wire_decider(agent, [])                    # 模型不可用
    obs = SimpleNamespace(step_count=200, max_steps=500,
                          goal_text="Find a basket",
                          rgb=np.zeros((48, 64, 3), dtype=np.uint8),
                          episode_id="ep", previous_action=None)
    assert agent._decider_next(obs, "instance_confirmed") is None


def test_navagent_rules_mode_has_no_decider():
    agent = NavAgent()                          # 默认 NAV_DECIDER=rules
    assert agent.decision_loop is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
