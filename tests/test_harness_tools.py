"""harness 化新增能力测试：决策工具、动作流水、白名单放宽、get_captions。

只依赖 numpy + mock client，不需要建图服务端或真实 VLM。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.decision_state import build_world_state
from agents.nav_agent import NavAgent
from decision import DecisionLoop
from mapping.caption_store import CaptionStore
from mapping.client import MappingClient


def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def _state(mode="all", found=1, expected=None):
    return {
        "task": {"goal": "Find all baskets", "mode": mode,
                 "found": found, "expected": expected},
        "step": 400, "max_steps": 500,
        "instances": [{"id": 1, "text": "basket near a shelf",
                       "reported": False}],
        "frontiers": [{"id": "f0", "path_cost_m": 3.0}],
        "navigation": {"active_target": {"type": "instance", "id": 1}},
    }


class _ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, images):
        self.calls.append((prompt, images))
        return self.replies.pop(0) if self.replies else None


def _hit():
    return {"found": True, "point": [1.0, 2.0, 0.0], "text": "a basket",
            "frame_id": 5, "candidate_id": "c5", "point_score": 0.9}


# ---------------------------------------------------------------- 工具：notes
def test_set_notes_tool_truncates_and_roundtrips():
    agent = _make_agent()
    out = agent._tool_set_notes("plan: check frame 7")
    assert out == {"notes": "plan: check frame 7"}
    assert agent._notes == "plan: check frame 7"
    out = agent._tool_set_notes("x" * 600)
    assert len(agent._notes) == 500
    assert out["notes"] == agent._notes


# -------------------------------------------------------- 工具：action history
def test_action_history_excludes_pending_and_pages():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=10)
    agent._record_action("GOTO_INSTANCE", 3)
    agent._last_observation = SimpleNamespace(step_count=11)
    agent._record_action("EXPLORE", None)
    # 第一条已结算，第二条 outcome=None（进行中）不出现在历史里
    agent._action_log[0]["outcome"] = "ok"
    rows = agent._tool_get_action_history()
    assert rows == [{"step": 10, "action": "GOTO_INSTANCE",
                     "target_id": "3", "outcome": "ok"}]
    agent._action_log[1]["outcome"] = "collision"
    rows = agent._tool_get_action_history(before_step=11)
    assert [r["step"] for r in rows] == [10]
    rows = agent._tool_get_action_history(limit=1)
    assert [r["step"] for r in rows] == [11]


def test_action_history_limit_returns_latest():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=0)
    for i in range(5):
        agent._last_observation = SimpleNamespace(step_count=i)
        agent._record_action("EXPLORE", None)
        agent._action_log[-1]["outcome"] = "ok"
    rows = agent._tool_get_action_history(limit=2)
    assert [r["step"] for r in rows] == [3, 4]


def test_record_action_caps_log_at_500():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=1)
    for _ in range(600):
        agent._record_action("EXPLORE", None)
    assert len(agent._action_log) == 500


def test_settle_action_outcomes_marks_collision_and_ok():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=5)
    agent._record_action("GOTO_FRONTIER", "f0")
    agent._last_motion_failed = True
    agent._settle_action_outcomes()
    assert agent._action_log[0]["outcome"] == "collision"
    agent._record_action("TURN_LEFT", None)
    agent._last_motion_failed = False
    agent._settle_action_outcomes()
    assert agent._action_log[1]["outcome"] == "ok"


def test_mark_goto_arrived_settles_latest_goto():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=5)
    agent._record_action("EXPLORE", None)
    agent._record_action("GOTO_INSTANCE", 3)
    agent._mark_goto_arrived()
    assert agent._action_log[1]["outcome"] == "arrived"
    assert agent._action_log[0]["outcome"] is None


# --------------------------------------------------------- 工具：map status
def test_get_agent_status_aggregates_server_and_agent_state():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=100, max_steps=500)
    agent.memory.add([1, 0, 0], "basket")
    reported = agent.memory.add([2, 0, 0], "reported basket")
    agent.memory.mark_reported(reported)
    agent.client = SimpleNamespace(
        get_state=lambda: {"num_frames": 42, "num_submaps": 3,
                           "num_loop_closures": 1, "caption_pending": 2,
                           "semantic": {"caption_enabled": True}},
        get_captioned_frame_ids=lambda: (True, [3, 5, 7, 9, 11, 13, 15]))
    status = agent._tool_get_agent_status()
    assert status["num_frames"] == 42
    assert status["caption_pending"] == 2
    assert status["latest_captioned_frame_ids"] == [7, 9, 11, 13, 15]
    assert status["instances_total"] == 2
    assert status["unreported_instances"] == 1
    assert status["steps_remaining"] == 400


def test_get_agent_status_swallows_server_errors():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_state=lambda: (_ for _ in ()).throw(RuntimeError("rpc down")))
    out = agent._tool_get_agent_status()
    assert "error" in out


# --------------------------------------------------------- 工具：view_frame
def test_view_frame_returns_jpeg_payload():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_frame_image=lambda fid: (({"found": True}, b"jpeg-7")
                                     if fid == 7 else ({"found": False}, b"")))
    assert agent._tool_view_frame(7) == b"jpeg-7"
    assert agent._tool_view_frame(8) is None
    agent.client = SimpleNamespace()      # 旧服务端无此方法
    assert agent._tool_view_frame(7) is None


# ------------------------------- 工具：ground_target / instantiate_points
def test_ground_target_without_frame_retrieves_and_ingests_hits():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace(
        ground_object=lambda text, top_k: [_hit(), {"found": False}])
    rows = agent._tool_ground_target("basket")
    assert len(agent.memory.nodes) == 1
    node = agent.memory.nodes[0]
    assert rows == [{"instance_id": node.iid, "observation_id": 1,
                     "frame_id": 5, "confidence": 0.9,
                     "association": "new_without_visual_relation",
                     "reported": False}]


def test_ground_target_with_frame_skips_retrieval_and_uses_exact_frame():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    seen = []

    def point_frame(frame_id, query):
        seen.append((frame_id, query))
        return {"results": [_hit()]}

    agent.client = SimpleNamespace(
        point_frame=point_frame,
        ground_object=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ground_object must not run for a fixed frame")))
    rows = agent._tool_ground_target("basket by the sink", frame_id=5)
    assert seen == [(5, "basket by the sink")]
    assert rows[0]["frame_id"] == 5


def test_ground_target_requires_observation_and_handles_errors():
    agent = _make_agent()
    assert "error" in agent._tool_ground_target("basket")   # 尚无观测
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace(
        ground_object=lambda text, top_k: (_ for _ in ()).throw(
            RuntimeError("rpc down")))
    assert "error" in agent._tool_ground_target("basket")
    agent.client = SimpleNamespace(
        ground_object=lambda text, top_k: [{"found": False}])
    assert agent._tool_ground_target("basket") == []


def test_instantiate_points_uses_normalized_pixels_and_ingests():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    seen = []

    def instantiate_pixels(frame_id, pixels, normalized=True):
        seen.append((frame_id, pixels, normalized))
        return {"results": [_hit()]}

    agent.client = SimpleNamespace(instantiate_pixels=instantiate_pixels)
    rows = agent._tool_instantiate_points(5, [[500, 500]], "basket")
    assert seen == [(5, [[500, 500]], True)]
    assert rows[0]["instance_id"] == agent.memory.nodes[0].iid
    assert rows[0]["observation_id"] == 1
    assert rows[0]["association"] == "new_without_visual_relation"
    assert rows[0]["frame_id"] == 5
    assert agent.memory.nodes[0].text == "a basket"


def test_instantiate_points_label_becomes_initial_text():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    hit = _hit()
    del hit["text"]
    agent.client = SimpleNamespace(
        instantiate_pixels=lambda fid, pixels, normalized=True:
            {"results": [hit]})
    agent._tool_instantiate_points(
        5, [[500, 500]], "wooden chair by the table")
    assert agent.memory.nodes[0].text == "wooden chair by the table"


def test_instantiate_points_requires_pixels():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace()
    out = agent._tool_instantiate_points(5, [], "basket")
    assert "error" in out


def test_instantiate_points_propagates_server_error():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace(
        instantiate_pixels=lambda fid, pixels, normalized=True:
            {"results": [], "error": "unknown frame_id 99"})
    out = agent._tool_instantiate_points(99, [[500, 500]], "basket")
    assert "error" in out


def test_point_frame_returns_normalized_pixels():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        point_pixels=lambda fid, text: {
            "width": 1280, "height": 720,
            "points": [{"pixel": [640.0, 360.0], "confidence": 1.0,
                        "bbox": None}]})
    out = agent._tool_point_frame(7, "wooden chair")
    assert out == {"points": [{"pixel": [500.0, 500.0]}]}


def test_point_frame_propagates_server_error():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        point_pixels=lambda fid, text: {"points": [],
                                        "error": "unknown frame_id 99"})
    out = agent._tool_point_frame(99, "wooden chair")
    assert "error" in out


# -------------------------------------------- world-state：notes / recent_actions
def test_world_state_includes_notes_and_recent_actions():
    agent = _make_agent()
    agent._notes = "working on frame 7"
    agent._last_observation = SimpleNamespace(step_count=9)
    agent._record_action("GOTO_INSTANCE", 1)
    agent._action_log[0]["outcome"] = "arrived"
    agent._record_action("EXPLORE", None)      # 进行中，不注入
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(agent, obs)
    assert state["notes"] == "working on frame 7"
    assert state["recent_actions"] == [
        {"step": 9, "action": "GOTO_INSTANCE", "target_id": "1",
         "outcome": "arrived"}]


def test_world_state_recent_actions_caps_at_three():
    agent = _make_agent()
    for i in range(5):
        agent._last_observation = SimpleNamespace(step_count=i)
        agent._record_action("EXPLORE", None)
        agent._action_log[-1]["outcome"] = "ok"
    obs = SimpleNamespace(step_count=50, max_steps=500, goal_text="x")
    state = build_world_state(agent, obs)
    assert [r["step"] for r in state["recent_actions"]] == [2, 3, 4]


# -------------------------------------------- _build_decider_input：新关键帧通知
def test_build_decider_input_attaches_new_keyframes_once():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0},
        get_captioned_frame_ids=lambda: (True, [3, 5]),
        get_captions=lambda ids: {"captions": [
            {"frame_id": 3, "caption": "a kitchen"},
            {"frame_id": 5, "caption": "c" * 300}]})
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state, _map = agent._build_decider_input(obs)
    assert state["new_keyframes"] == [
        {"frame_id": 3, "caption": "a kitchen"},
        {"frame_id": 5, "caption": "c" * 200}]
    assert agent._last_notified_frame_id == 5
    # 第二次决策无新帧：不再携带该字段
    state, _map = agent._build_decider_input(obs)
    assert "new_keyframes" not in state


def test_build_decider_input_skips_keyframes_when_server_is_old():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0})
    obs = SimpleNamespace(step_count=50, max_steps=500, goal_text="x")
    state, _map = agent._build_decider_input(obs)
    assert "new_keyframes" not in state


# ------------------------------------------------------ agent_loop：view_frame
def test_view_frame_tool_attaches_frame_image():
    chat = _ScriptedChat([
        {"tool_call": {"name": "view_frame", "frame_id": 7}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"view_frame": lambda frame_id: b"frame-jpeg"}).decide(
            "world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert ("tool_frame_7_rgb", b"frame-jpeg") in chat.calls[1][1]
    assert '"image_ref": "tool_frame_7_rgb"' in chat.calls[1][0]


def test_view_frame_missing_image_returns_error():
    chat = _ScriptedChat([
        {"tool_call": {"name": "view_frame", "frame_id": 7}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"view_frame": lambda frame_id: None}).decide(
            "world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert "frame image not found" in chat.calls[1][0]


# ------------------------------------------- agent_loop：adjustment 禁工具
def test_adjustment_disables_tool_calls():
    called = []
    chat = _ScriptedChat([
        {"tool_call": {"name": "search_frames", "query": "x"}},
        {"action": "END_ADJUST"},
    ])
    result = DecisionLoop(
        chat,
        tools={"search_frames": lambda query: called.append(query) or []}
    ).decide("adjustment", _state())
    assert result.action == "END_ADJUST"
    assert called == []
    assert result.tool_calls == 0
    assert "tools are disabled during adjustment" in chat.calls[1][0]


def test_new_write_tools_refresh_world_state():
    fresh = _state()

    def set_notes(text):
        return {"notes": text}

    chat = _ScriptedChat([
        {"tool_call": {"name": "set_notes", "text": "go to frame 7"}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    calls = []

    def state_fn():
        calls.append(1)
        return fresh

    result = DecisionLoop(
        chat, tools={"set_notes": set_notes}).decide(
            "world_state_updated", _state(), state_fn=state_fn)
    assert result.action == "GOTO_FRONTIER"
    assert calls == [1]          # set_notes 触发 world-state 刷新
    assert "World state after your write" in chat.calls[1][0]


# ------------------------------------------------------- 放宽后的事件白名单
def test_relaxed_event_whitelists():
    # arrival 允许 GOTO_FRONTIER
    result = DecisionLoop(_ScriptedChat([
        {"action": "GOTO_FRONTIER", "target_id": "f0"}])).decide(
            "arrival", _state())
    assert result.action == "GOTO_FRONTIER" and result.target_id == "f0"
    # world_state_updated 允许 FINISH / REPORT_FOUND / SCAN
    for action in ("FINISH", "REPORT_FOUND", "SCAN"):
        target_id = "1" if action == "REPORT_FOUND" else None
        result = DecisionLoop(_ScriptedChat([{
            "action": action, "target_id": target_id}])).decide(
                "world_state_updated", _state())
        assert result.action == action
    # finish_check 仍不允许 SCAN / REPORT_FOUND
    for rejected in ("SCAN", "REPORT_FOUND"):
        chat = _ScriptedChat([
            {"action": rejected}, {"action": "FINISH"}])
        assert DecisionLoop(chat).decide(
            "finish_check", _state()).action == "FINISH"
    # EXPLORE 在所有事件中都被拒绝（finish_check 也不例外）
    chat = _ScriptedChat([
        {"action": "EXPLORE"}, {"action": "FINISH"}])
    assert DecisionLoop(chat).decide(
        "finish_check", _state()).action == "FINISH"
    # adjustment 白名单不变
    chat = _ScriptedChat([
        {"action": "EXPLORE"}, {"action": "END_ADJUST"}])
    assert DecisionLoop(chat).decide(
        "adjustment", _state()).action == "END_ADJUST"


# ------------------------------------------------------------ get_captions
def test_caption_store_get_captions_skips_missing():
    store = CaptionStore()
    store.add(3, None, "a kitchen", np.ones(8, dtype=np.float32))
    store.add(5, None, "a hallway", np.ones(8, dtype=np.float32))
    assert store.get_captions([3, 99, 5]) == [
        {"frame_id": 3, "caption": "a kitchen"},
        {"frame_id": 5, "caption": "a hallway"}]
    assert store.get_captions([]) == []


def test_client_get_captions_rpc_shape():
    client = MappingClient.__new__(MappingClient)
    seen = []

    def fake_request(header, payload=b"", retries=1):
        seen.append(header)
        return {"ok": True,
                "captions": [{"frame_id": 3, "caption": "x"}]}, b""

    client._request = fake_request
    out = client.get_captions([3, "5"])
    assert seen[0]["cmd"] == "get_captions"
    assert seen[0]["frame_ids"] == [3, 5]
    assert out["captions"] == [{"frame_id": 3, "caption": "x"}]
