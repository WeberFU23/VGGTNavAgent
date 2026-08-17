"""具身 VLM harness：动作协议、记忆工具、地图与状态回归测试。"""

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from benchmark_api import Action

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import navigator as nav
from agents.decision_state import build_world_state
from agents.map_render import render_topdown
from agents.nav_agent import NavAgent
from decision import DecisionLoop, DecisionTraceLogger


def _state(mode="all", found=1, expected=None, instances=None, frontiers=None):
    return {
        "task": {"goal": "Find all baskets", "mode": mode,
                 "found": found, "expected": expected},
        "step": 400, "max_steps": 500,
        "instances": instances if instances is not None else [
            {"id": 1, "text": "basket near a shelf", "reported": False}],
        "frontiers": frontiers if frontiers is not None else [
            {"id": "f0", "dist_m": 3.0, "size": 12}],
        "recent_events": [],
        "termination": {"unexplored_ratio": 0.05,
                        "unreported_instance_count": 1,
                        "frontier_count": 1},
    }


class _ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, images):
        self.calls.append((prompt, images))
        return self.replies.pop(0) if self.replies else None


def test_prompt_documents_action_effects_tool_returns_and_no_confidence():
    prompt = DecisionLoop(_ScriptedChat([]))._build_prompt(
        "arrival", _state())
    assert "plans an A* path" in prompt
    assert "12 left turns, four sampled views" in prompt
    assert "Returns a JSON array of {frame_id, score, caption}" in prompt
    assert "returns the surviving full instance" in prompt
    assert '"name": "search_captions"' in prompt
    assert '"name": "search_instances"' in prompt
    assert '"name": "look_instance"' in prompt
    assert "search_instances -> inspect_instance and/or" in prompt
    assert "Returns the full object {id, point, text," in prompt
    assert "look_at" not in prompt
    assert "query_memory" not in prompt
    assert "list_instances" not in prompt
    assert '"confidence"' not in prompt


def test_search_captions_tool_call():
    seen = []

    def search_captions(text):
        seen.append(text)
        return [{"frame_id": 7, "score": 0.8, "caption": "red cup"}]

    chat = _ScriptedChat([
        {"tool_call": {"name": "search_captions", "text": "red cup"}},
        {"action": "EXPLORE"},
    ])
    result = DecisionLoop(
        chat, tools={"search_captions": search_captions}).decide(
            "world_state_updated", _state())
    assert seen == ["red cup"]
    assert result.action == "EXPLORE" and result.tool_calls == 1


def test_look_instance_attaches_instance_evidence_image():
    chat = _ScriptedChat([
        {"tool_call": {"name": "look_instance", "instance_id": 3}},
        {"action": "EXPLORE"},
    ])
    result = DecisionLoop(
        chat, tools={"look_instance": lambda instance_id: b"instance-jpeg"}) \
        .decide("world_state_updated", _state())
    assert result.action == "EXPLORE"
    assert ("tool_instance_evidence", b"instance-jpeg") in chat.calls[1][1]


def test_goto_accepts_any_unreported_instance():
    chat = _ScriptedChat([{"action": "GOTO_INSTANCE", "target_id": "1"}])
    result = DecisionLoop(chat).decide("world_state_updated", _state())
    assert result.action == "GOTO_INSTANCE" and result.target_id == "1"
    assert "confidence" not in result.as_dict()


def test_reported_instance_is_not_a_navigation_target():
    chat = _ScriptedChat([
        {"action": "GOTO_INSTANCE", "target_id": "1"},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    state = _state(instances=[{"id": 1, "text": "basket", "reported": True}])
    result = DecisionLoop(chat).decide("world_state_updated", state)
    assert result.action == "GOTO_FRONTIER"


def test_verify_and_reject_are_not_actions():
    for removed in ("VERIFY", "REJECT"):
        chat = _ScriptedChat([
            {"action": removed},
            {"action": "EXPLORE"},
        ])
        result = DecisionLoop(chat).decide("world_state_updated", _state())
        assert result.action == "EXPLORE"


def test_arrival_actions_are_report_scan_or_explore():
    for action in ("REPORT_FOUND", "SCAN", "EXPLORE"):
        result = DecisionLoop(_ScriptedChat([
            {"action": action}])).decide("arrival", _state())
        assert result.action == action


def test_scan_complete_reselects_globally_instead_of_reporting():
    for action, target_id in (("GOTO_INSTANCE", "1"),
                              ("GOTO_FRONTIER", "f0"),
                              ("EXPLORE", None)):
        result = DecisionLoop(_ScriptedChat([{
            "action": action, "target_id": target_id
        }])).decide("scan_complete", _state())
        assert result.action == action
    chat = _ScriptedChat([
        {"action": "REPORT_FOUND"}, {"action": "EXPLORE"}])
    assert DecisionLoop(chat).decide(
        "scan_complete", _state()).action == "EXPLORE"


def test_generic_memory_tool_call():
    seen = []

    def update_instance(instance_id, text):
        seen.append((instance_id, text))
        return {"id": instance_id, "text": text}

    chat = _ScriptedChat([
        {"tool_call": {"name": "update_instance", "instance_id": 1,
                       "text": "red cup, uncertain handle"}},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    result = DecisionLoop(chat, tools={"update_instance": update_instance}) \
        .decide("world_state_updated", _state())
    assert seen == [(1, "red cup, uncertain handle")]
    assert result.tool_calls == 1


def test_write_tool_refreshes_world_state_before_validation():
    stale = _state(instances=[
        {"id": 1, "text": "red cup on table", "reported": False},
        {"id": 2, "text": "same red cup, other view", "reported": False},
    ])
    fresh = _state(instances=[
        {"id": 1, "text": "merged red cup", "reported": False},
    ])

    def merge_instances(instance_ids, text=""):
        return {"id": 1, "text": text, "merged": list(instance_ids)}

    chat = _ScriptedChat([
        {"tool_call": {"name": "merge_instances", "instance_ids": [1, 2],
                       "text": "merged red cup"}},
        # 合并后实例 2 已删除；基于旧状态的目标必须被拒绝
        {"action": "GOTO_INSTANCE", "target_id": "2"},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    result = DecisionLoop(
        chat, tools={"merge_instances": merge_instances}).decide(
            "world_state_updated", stale, state_fn=lambda: fresh)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "1"
    assert result.tool_calls == 1
    # 重试 prompt 中应包含刷新后的 world-state（不再有实例 2 的文本）
    retry_prompt = chat.calls[2][0]
    assert "World state after your write" in retry_prompt
    assert "merged red cup" in retry_prompt
    assert "same red cup, other view" not in retry_prompt.split(
        "World state after your write")[-1]


def test_write_tool_refreshes_topdown_map_image():
    old_map = b"old-map"
    new_map = b"new-map"
    chat = _ScriptedChat([
        {"tool_call": {"name": "update_instance", "instance_id": 1,
                       "text": "updated basket"}},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    loop = DecisionLoop(chat, tools={
        "update_instance": lambda instance_id, text: {
            "id": instance_id, "text": text}})
    result = loop.decide(
        "world_state_updated", _state(), map_png=old_map,
        state_fn=lambda: (_state(), new_map))
    assert result.action == "GOTO_INSTANCE"
    second_images = chat.calls[1][1]
    assert ("topdown_map", new_map) in second_images
    assert ("topdown_map", old_map) not in second_images


def test_failed_write_tool_does_not_refresh_state():
    calls = []

    def merge_instances(instance_ids, text=""):
        return {"error": "merge requires at least two existing instances"}

    chat = _ScriptedChat([
        {"tool_call": {"name": "merge_instances", "instance_ids": [1, 9]}},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    result = DecisionLoop(
        chat, tools={"merge_instances": merge_instances}).decide(
            "world_state_updated", _state(),
            state_fn=lambda: calls.append(1) or _state())
    assert result.action == "GOTO_INSTANCE" and result.target_id == "1"
    assert calls == []


def test_broken_state_fn_falls_back_to_pre_write_state():
    def update_instance(instance_id, text):
        return {"id": instance_id, "text": text}

    def bad_state_fn():
        raise RuntimeError("state rebuild failed")

    chat = _ScriptedChat([
        {"tool_call": {"name": "update_instance", "instance_id": 1,
                       "text": "red cup"}},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    result = DecisionLoop(
        chat, tools={"update_instance": update_instance}).decide(
            "world_state_updated", _state(), state_fn=bad_state_fn)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "1"


def test_finish_only_enforces_explicit_many_count():
    finish = [{"action": "FINISH"}]
    assert DecisionLoop(_ScriptedChat(finish)).decide(
        "finish_check", _state()).action == "FINISH"
    short = DecisionLoop(_ScriptedChat(finish)).decide(
        "finish_check", _state(mode="many", found=1, expected=3))
    assert short.action == "GOTO_INSTANCE"
    assert short.validation == "finish_downgraded"


def test_trace_log_written():
    path = Path(".decision_trace_test.jsonl")
    if path.exists():
        path.unlink()
    loop = DecisionLoop(
        _ScriptedChat([{"action": "FINISH"}]),
        logger=DecisionTraceLogger(path))
    loop.decide("finish_check", _state())
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["output"]["action"] == "FINISH"
    path.unlink()


def _grid(h=30, w=40):
    free = np.zeros((h, w), dtype=bool)
    free[3:-3, 3:-3] = True
    return nav.OccupancyGrid(
        1.0, np.array([0.0, 0.0]), free, ~free)


def test_render_topdown_uses_single_instance_layer():
    png = render_topdown(
        _grid(), trajectory=[(5, 5), (10, 10)], pose=(20, 15, 0.5),
        instances=[{"id": 1, "xy": (25, 20), "reported": True}],
        frontiers=[{"id": "f0", "xy": (35, 25)}])
    assert png[:4] == b"\x89PNG"
    assert Image.open(io.BytesIO(png)).size == (160, 120)


def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def test_world_state_contains_text_evidence_and_no_anchor_table():
    agent = _make_agent()
    node, _ = agent.memory.remember(
        [3, 4, 0], "possible woven basket",
        evidence=[{"frame_id": 7}], candidate_id="c7")
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(
        agent, obs, grid=_grid(),
        frontiers=[{"world": np.array([10.0, 2.0]), "size": 7}])
    row = state["instances"][0]
    assert row["id"] == node.iid
    assert row["text"] == "possible woven basket"
    assert row["evidence_count"] == 1
    assert "belief_anchors" not in state
    assert state["termination"]["unreported_instance_count"] == 1
    assert state["instances_total"] == 1
    assert state["instances_omitted_ids"] == []
    assert state["reported_instance_ids"] == []


def _build_state(agent, step=50):
    obs = SimpleNamespace(step_count=step, max_steps=500,
                          goal_text="Find all baskets")
    return build_world_state(agent, obs)


def test_world_state_summarizes_instances_beyond_k():
    agent = _make_agent()
    for i in range(35):
        agent.memory.add([float(i + 1), 0, 0], f"instance number {i}",
                         step=i)
    reported = agent.memory.add([100, 0, 0], "already reported")
    agent.memory.mark_reported(reported)
    state = _build_state(agent)
    assert len(state["instances"]) == 30          # K=30 硬上限
    assert state["instances_total"] == 36
    assert len(state["instances_omitted_ids"]) == 5
    assert state["reported_instance_ids"] == [reported.iid]
    # 摘要按距离排序，最近的一定入选；reported 不在表中
    ids = [row["id"] for row in state["instances"]]
    assert 1 in ids and reported.iid not in ids
    assert all(not row["reported"] for row in state["instances"])
    dists = [row["dist_m"] for row in state["instances"]]
    assert dists == sorted(dists)


def test_world_state_summary_prefers_nearest_newest_relevant():
    agent = _make_agent()                          # target_text="basket"
    nearest = agent.memory.add([1, 0, 0], "wooden chair", step=1)
    newest = agent.memory.add([9, 0, 0], "small table", step=100)
    relevant = agent.memory.add([5, 0, 0], "woven basket with handle",
                                step=2)
    other = agent.memory.add([2, 0, 0], "desk lamp", step=3)
    os.environ["NAV_STATE_MAX_INSTANCES"] = "3"    # K=3 -> 每路取 1
    try:
        state = _build_state(agent)
    finally:
        del os.environ["NAV_STATE_MAX_INSTANCES"]
    ids = [row["id"] for row in state["instances"]]
    assert set(ids) == {nearest.iid, newest.iid, relevant.iid}
    assert state["instances_omitted_ids"] == [other.iid]
    assert ids == sorted(ids, key=lambda i: {nearest.iid: 1.0,
                                             relevant.iid: 5.0,
                                             newest.iid: 9.0}[i])


def test_world_state_truncates_instance_text():
    agent = _make_agent()
    agent.memory.add([1, 0, 0], "x" * 300)
    row = _build_state(agent)["instances"][0]
    assert len(row["text"]) == 120
    assert row["text"].endswith("...")


def test_omitted_instance_is_valid_goto_target():
    state = _state(instances=[{"id": 1, "text": "cup", "reported": False}])
    state["instances_omitted_ids"] = [42]
    chat = _ScriptedChat([{"action": "GOTO_INSTANCE", "target_id": "42"}])
    result = DecisionLoop(chat).decide("world_state_updated", state)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "42"


def test_navagent_memory_tools_update_and_merge():
    agent = _make_agent()
    a = agent.memory.add([0, 0, 0], "view A")
    b = agent.memory.add([2, 0, 0], "view B")
    updated = agent._tool_update_instance(a.iid, "same basket, front view")
    assert updated["text"] == "same basket, front view"
    merged = agent._tool_merge_instances([a.iid, b.iid], "one basket")
    assert merged["id"] == a.iid and len(agent.memory.nodes) == 1


def test_navagent_undo_merge_restores_originals():
    agent = _make_agent()
    a = agent.memory.add([0, 0, 0], "view A", evidence=[{"frame_id": 1}])
    b = agent.memory.add([2, 0, 0], "view B", evidence=[{"frame_id": 2}])
    agent._tool_merge_instances([a.iid, b.iid], "one basket")
    assert len(agent.memory.nodes) == 1
    out = agent._tool_undo_merge()
    assert out["kept"]["text"] == "view A"
    assert out["restored"][0]["text"] == "view B"
    assert [n.iid for n in agent.memory.nodes] == [a.iid, b.iid]
    assert list(agent.memory.get(a.iid).point) == [0, 0, 0]
    assert agent.memory.get(b.iid).evidence == [{"frame_id": 2}]
    assert "error" in agent._tool_undo_merge()


def test_navagent_undo_merge_restores_removed_navigation_target():
    agent = _make_agent()
    a = agent.memory.add([0, 0, 0], "view A", candidate_id="ca")
    b = agent.memory.add([2, 0, 0], "view B", candidate_id="cb")
    agent.target_instance_id = b.iid
    agent.target_point = np.asarray(b.point)
    agent.target_candidate_id = b.candidate_id
    agent._tool_merge_instances([a.iid, b.iid], "one basket")
    assert agent.target_instance_id == a.iid
    agent._tool_undo_merge()
    assert agent.target_instance_id == b.iid
    assert np.allclose(agent.target_point, b.point)
    assert agent.target_candidate_id == "cb"


def test_report_found_does_not_emit_duplicate_target_found():
    agent = _make_agent()
    node = agent.memory.add([0, 0, 0], "basket")
    agent.memory.mark_reported(node)
    agent.target_instance_id = node.iid
    agent.target_point = np.asarray(node.point)
    before = agent._reported_count
    action = agent._report_found()
    assert action != int(Action.TARGET_FOUND)
    assert agent._reported_count == before
    assert agent.target_instance_id is None


def test_undo_merge_never_revokes_report():
    agent = _make_agent()
    a = agent.memory.add([0, 0, 0], "A")
    b = agent.memory.add([2, 0, 0], "B")
    agent.memory.merge([a.iid, b.iid], "merged")
    agent.memory.mark_reported(agent.memory.get(a.iid))
    agent.memory.undo_merge()
    assert agent.memory.get(a.iid).reported      # 已发生的报告不撤销
    assert not agent.memory.get(b.iid).reported


def test_undo_merge_tool_refreshes_state():
    stale = _state(instances=[
        {"id": 1, "text": "merged cup", "reported": False}])
    fresh = _state(instances=[
        {"id": 1, "text": "cup view A", "reported": False},
        {"id": 2, "text": "cup view B", "reported": False}])
    calls = []

    def undo_merge():
        calls.append(1)
        return {"kept": {"id": 1}, "restored": [{"id": 2}]}

    chat = _ScriptedChat([
        {"tool_call": {"name": "undo_merge"}},
        # 撤销后实例 2 才存在；校验必须基于刷新后的状态
        {"action": "GOTO_INSTANCE", "target_id": "2"},
    ])
    result = DecisionLoop(chat, tools={"undo_merge": undo_merge}).decide(
        "world_state_updated", stale, state_fn=lambda: fresh)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "2"
    assert calls == [1]


def _jpeg_bytes(size=(100, 100)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeTextVLM:
    enabled = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat_text(self, prompt, images):
        self.calls.append((prompt, images))
        return self.reply


def _ingest_hit():
    return {"found": True, "point": [1.0, 2.0, 0.0],
            "text": "a kitchen counter with several objects on it",
            "frame_id": 5, "candidate_id": "c5",
            "bbox": [10, 10, 50, 50], "point_score": 0.9}


def test_ingest_generates_instance_level_text():
    agent = _make_agent()
    agent.vlm = _FakeTextVLM("a red ceramic cup beside the sink")
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"overlay-jpeg"),
        get_frame_image=lambda fid: ({"found": True}, _jpeg_bytes()))
    obs = SimpleNamespace(step_count=50)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    node = agent.memory.nodes[0]
    assert node.text == "a red ceramic cup beside the sink"
    prompt, images = agent.vlm.calls[0]
    assert "basket" in prompt          # 任务上下文进入描述 prompt
    assert "kitchen counter" in prompt  # 关键帧 caption 作为上下文
    assert [name for name, _ in images] == \
        ["pointing_overlay", "instance_crop"]


def test_ingest_keeps_caption_text_when_vlm_unavailable():
    obs = SimpleNamespace(step_count=50)
    agent = _make_agent()               # 默认 VLM disabled
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert agent.memory.nodes[0].text == \
        "a kitchen counter with several objects on it"
    # VLM 可用但调用失败：同样保留 caption 文本
    agent2 = _make_agent()
    agent2.vlm = _FakeTextVLM(None)
    agent2.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"x"),
        get_frame_image=lambda fid: ({"found": False}, b""))
    agent2._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert agent2.memory.nodes[0].text == \
        "a kitchen counter with several objects on it"


def test_ingest_does_not_redescribe_existing_instance():
    agent = _make_agent()
    agent.vlm = _FakeTextVLM("new description")
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"overlay-jpeg"),
        get_frame_image=lambda fid: ({"found": True}, _jpeg_bytes()))
    obs = SimpleNamespace(step_count=50)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert len(agent.memory.nodes) == 1      # 同 candidate_id 只更新
    assert len(agent.vlm.calls) == 1         # 已有实例不重复生成


def test_wait_for_captions_logs_timeout_and_swallows_errors():
    agent = _make_agent()
    agent.client = SimpleNamespace(wait_captions=lambda timeout: False)
    os.environ["NAV_CAPTION_WAIT_S"] = "5"
    try:
        agent._wait_for_captions()
        assert any("caption backlog" in e for e in agent._events)
        agent.client = SimpleNamespace()     # 旧服务端无 wait_captions
        agent._wait_for_captions()           # 不抛异常
    finally:
        del os.environ["NAV_CAPTION_WAIT_S"]


def test_scan_flushes_tail_map_before_waiting_and_retrieving():
    agent = _make_agent()
    order = []
    agent.client = SimpleNamespace(
        flush_map=lambda: order.append("flush") or {"flushed": True},
        wait_captions=lambda timeout: order.append("wait") or True,
        ground_object=lambda phrase, top_k:
            order.append("retrieve") or [])
    agent._choose_high_level_target = lambda *args, **kwargs: 123
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    assert agent._scan_complete_decision(obs) == 123
    assert order == ["flush", "wait", "retrieve"]


def test_navagent_search_instances_uses_vlm_keywords():
    agent = _make_agent()
    red_cup = agent.memory.add(
        [0, 0, 0], "small red ceramic cup beside sink",
        evidence=[{"frame_id": 7}])
    agent.memory.add([1, 0, 0], "blue cup on table")
    reported = agent.memory.add([2, 0, 0], "reported red cup")
    agent.memory.mark_reported(reported)
    rows = agent._tool_search_instances(
        ["red", "cup"], reported=False, top_k=5)
    assert [row["id"] for row in rows] == [red_cup.iid, 2]
    assert rows[0]["matched_keywords"] == ["red", "cup"]
    assert rows[0]["frame_ids"] == [7]
    assert all(not row["reported"] for row in rows)


def test_navagent_look_instance_prefers_candidate_overlay():
    agent = _make_agent()
    node, _ = agent.memory.remember(
        [1, 2, 0], "red cup", frame_id=7, candidate_id="c7")
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda candidate_id:
            ({"found": True}, b"overlay"),
        get_frame_image=lambda frame_id: ({"found": True}, b"frame"))
    assert agent._tool_look_instance(node.iid) == b"overlay"


if __name__ == "__main__":
    test_goto_accepts_any_unreported_instance()
    test_prompt_documents_action_effects_tool_returns_and_no_confidence()
    test_search_captions_tool_call()
    test_look_instance_attaches_instance_evidence_image()
    test_reported_instance_is_not_a_navigation_target()
    test_verify_and_reject_are_not_actions()
    test_arrival_actions_are_report_scan_or_explore()
    test_scan_complete_reselects_globally_instead_of_reporting()
    test_generic_memory_tool_call()
    test_write_tool_refreshes_world_state_before_validation()
    test_write_tool_refreshes_topdown_map_image()
    test_failed_write_tool_does_not_refresh_state()
    test_broken_state_fn_falls_back_to_pre_write_state()
    test_finish_only_enforces_explicit_many_count()
    test_trace_log_written()
    test_render_topdown_uses_single_instance_layer()
    test_world_state_contains_text_evidence_and_no_anchor_table()
    test_world_state_summarizes_instances_beyond_k()
    test_world_state_summary_prefers_nearest_newest_relevant()
    test_world_state_truncates_instance_text()
    test_omitted_instance_is_valid_goto_target()
    test_navagent_memory_tools_update_and_merge()
    test_navagent_undo_merge_restores_originals()
    test_navagent_undo_merge_restores_removed_navigation_target()
    test_report_found_does_not_emit_duplicate_target_found()
    test_undo_merge_never_revokes_report()
    test_undo_merge_tool_refreshes_state()
    test_ingest_generates_instance_level_text()
    test_ingest_keeps_caption_text_when_vlm_unavailable()
    test_ingest_does_not_redescribe_existing_instance()
    test_wait_for_captions_logs_timeout_and_swallows_errors()
    test_scan_flushes_tail_map_before_waiting_and_retrieving()
    test_navagent_search_instances_uses_vlm_keywords()
    test_navagent_look_instance_prefers_candidate_overlay()
    print("decider tests passed")
