"""decision_window（最近 N 轮决策原始窗口）测试。

只依赖 numpy + mock client，不需要建图服务端或真实 VLM。
"""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent
from decision import DecisionLoop
from decision import prompts


def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def _exchange(reply='{"action": "GOTO_INSTANCE", "target_id": "1"}',
              tool_calls=None, tool_results=None,
              image_labels=("current_observation", "topdown_map")):
    return {"tool_calls": list(tool_calls or []),
            "tool_results": list(tool_results or []),
            "image_labels": list(image_labels),
            "reply": reply}


def _append(agent, event, step, exchange=None):
    agent.decision_loop = SimpleNamespace(
        last_exchange=exchange if exchange is not None else _exchange())
    agent._append_decision_window(event, step)


class _ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, images, system_prompt=None):
        self.calls.append((prompt, images))
        return self.replies.pop(0) if self.replies else None


def _state():
    return {"task": {"goal": "Find baskets", "mode": "all", "found": 0},
            "step": 10, "max_steps": 500,
            "instances": [{"id": 1, "text": "basket", "reported": False}],
            "frontiers": [{"id": "f0", "path_cost_m": 1.0}],
            "adjustment": {"max_forward_steps": 8,
                           "pitch_offset_steps": 0,
                           "max_pitch_offset_steps": 1}}


# ------------------------------------------------------------ 追加与滚动
def test_window_appends_and_rolls_at_maxlen(monkeypatch):
    monkeypatch.setenv("NAV_DECISION_WINDOW", "2")
    agent = _make_agent()
    agent._nav_reset_state()
    for step in (1, 2, 3):
        _append(agent, "world_state_updated", step)
    assert [e["step"] for e in agent._decision_window] == [2, 3]
    entry = agent._decision_window[-1]
    assert entry["event"] == "world_state_updated"
    assert entry["outcome"] is None
    assert entry["image_labels"] == ["current_observation", "topdown_map"]
    assert json.loads(entry["reply"])["action"] == "GOTO_INSTANCE"


def test_window_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("NAV_DECISION_WINDOW", "0")
    agent = _make_agent()
    agent._nav_reset_state()
    assert agent._decision_window is None
    _append(agent, "arrival", 7)          # no-op，不报错
    assert agent._decision_window is None
    assert agent._decision_window_entries() is None


def test_window_cleared_on_reset():
    agent = _make_agent()
    _append(agent, "arrival", 7)
    assert len(agent._decision_window) == 1
    agent._nav_reset_state()
    assert len(agent._decision_window) == 0


# ------------------------------------------------------- tool_results 截断
def test_loop_exchange_truncates_tool_results():
    chat = _ScriptedChat([
        {"tool_call": {"name": "big", "frame_id": 3}},
        {"action": "SCAN", "reason": "look around"},
    ])
    loop = DecisionLoop(
        chat_fn=chat,
        tools={"big": lambda frame_id: {"payload": "x" * 9000}})
    result = loop.decide("world_state_updated", _state())
    assert result is not None and result.action == "SCAN"
    exchange = loop.last_exchange
    assert exchange["tool_calls"] == [
        {"name": "big", "args": {"frame_id": 3}}]
    assert len(exchange["tool_results"]) == 1
    assert len(exchange["tool_results"][0]) <= 2000
    assert json.loads(exchange["reply"])["action"] == "SCAN"


def test_loop_exchange_skips_rejected_retries():
    chat = _ScriptedChat([
        {"action": "EXPLORE"},            # 该事件不允许，进校验重试
        {"action": "SCAN", "reason": "ok"},
    ])
    loop = DecisionLoop(chat_fn=chat, tools={})
    result = loop.decide("world_state_updated", _state())
    assert result.action == "SCAN"
    assert json.loads(loop.last_exchange["reply"])["action"] == "SCAN"


# --------------------------------------------------------- adjustment 条目
def test_adjustment_exchange_has_no_tool_calls():
    chat = _ScriptedChat([
        {"tool_call": {"name": "big"}},   # takeover 禁用工具 → 校验重试
        {"action": "TURN_LEFT", "reason": "look left"},
    ])
    loop = DecisionLoop(chat_fn=chat, tools={"big": lambda: {"ok": True}})
    result = loop.decide("adjustment", _state())
    assert result is not None and result.action == "TURN_LEFT"
    exchange = loop.last_exchange
    assert exchange["tool_calls"] == []
    assert json.loads(exchange["reply"])["action"] == "TURN_LEFT"

    agent = _make_agent()
    agent.decision_loop = loop
    agent._append_decision_window("adjustment", 12)
    entry = agent._decision_window[-1]
    assert entry["event"] == "adjustment"
    assert entry["tool_calls"] == []
    assert "TURN_LEFT" in entry["reply"]


# ----------------------------------------------------------- outcome 回填
def test_window_outcome_settles_ok_and_collision():
    agent = _make_agent()
    _append(agent, "world_state_updated", 5)
    agent._last_observation = SimpleNamespace(step_count=5)
    agent._record_action("GOTO_INSTANCE", 1)
    agent._last_motion_failed = False
    agent._settle_action_outcomes()
    assert agent._decision_window[-1]["outcome"] == "ok"

    _append(agent, "world_state_updated", 6)
    agent._record_action("GOTO_FRONTIER", "f0")
    agent._last_motion_failed = True
    agent._settle_action_outcomes()
    assert agent._decision_window[-1]["outcome"] == "collision"


def test_window_outcome_marks_arrived():
    agent = _make_agent()
    _append(agent, "world_state_updated", 5,
            _exchange(reply='{"action": "GOTO_FRONTIER", "target_id": "f0"}'))
    _append(agent, "world_state_updated", 6)   # 默认 GOTO_INSTANCE
    agent._mark_goto_arrived()
    assert agent._decision_window[-1]["outcome"] == "arrived"
    assert agent._decision_window[0]["outcome"] is None


# ------------------------------------------------------------- prompt 渲染
def test_prompt_renders_window_section():
    entries = [{
        "step": 41, "event": "world_state_updated",
        "tool_calls": [{"name": "instantiate_points",
                        "args": {"frame_id": 12, "label": "basket"}}],
        "tool_results": ['{"ok": true, "result": {"instances": []}}'],
        "image_labels": ["current_observation", "topdown_map"],
        "reply": '{"action": "GOTO_INSTANCE", "target_id": "1"}',
        "outcome": "arrived",
    }]
    text = prompts.build_decision_prompt(
        "arrival", _state(), 15, decision_window=entries)
    assert "Recent decision window (last 1 rounds" in text
    assert "[step 41 | world_state_updated]" in text
    assert "instantiate_points(frame_id=12" in text
    assert '-> {"ok": true' in text
    assert 'reply: {"action": "GOTO_INSTANCE", "target_id": "1"}' in text
    assert "outcome: arrived" in text
    assert "[image attached: current_observation]" in text


def test_prompt_omits_window_section_when_empty():
    for window in (None, []):
        text = prompts.build_decision_prompt(
            "arrival", _state(), 15, decision_window=window)
        assert "Recent decision window" not in text


def test_prompt_window_pending_outcome():
    entries = [{"step": 3, "event": "adjustment", "tool_calls": [],
                "tool_results": [], "image_labels": [],
                "reply": '{"action": "TURN_LEFT"}', "outcome": None}]
    text = prompts.build_decision_prompt(
        "adjustment", _state(), 15, decision_window=entries)
    assert "[step 3 | adjustment]" in text
    assert "outcome: pending" in text


# ------------------------------------------------------------ trace 字段
def test_decide_records_window_size_in_trace():
    chat = _ScriptedChat([{"action": "SCAN", "reason": "ok"}])
    loop = DecisionLoop(chat_fn=chat, tools={})
    records = []
    loop.decide("world_state_updated", _state(),
                decision_window=[{"step": 1}, {"step": 2}],
                trace_sink=records.append)
    assert records and records[0]["window_size"] == 2

    records.clear()
    loop.decide("world_state_updated", _state(), trace_sink=records.append)
    assert records[0]["window_size"] == 0
