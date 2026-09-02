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
from agents import skeleton as sk
from agents.decision_state import _path_cost_m, build_world_state
from agents.map_render import render_pointcloud_topdown, render_topdown
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
            {"id": "f0", "path_cost_m": 3.0}],
        "navigation": {"active_target": {"type": "instance", "id": 1}},
    }


def test_same_cell_path_cost_is_zero_not_unreachable():
    grid = nav.OccupancyGrid(
        1.0, np.zeros(2), np.ones((4, 4), dtype=bool),
        np.zeros((4, 4), dtype=bool))
    assert _path_cost_m(grid, (1.1, 1.1), (1.4, 1.4, 0.0), 1.0) == 0.0


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
    flat = " ".join(prompt.split())
    for tool in ("search_frames(query, top_k=5)", "view_frame(frame_id)",
                 "propose_candidates(frame_id, query)",
                 "som_pick(frame_id, mask_ids, query)",
                 "instantiate_points(frame_id, pixels_1000, label)",
                 "review_crosshair(frame_id, pixel_1000, verdict, reason)",
                 "search_instances(",
                 "get_instance(instance_id)",
                 "view_instance(instance_id)",
                 "update_instance(instance_id, text)",
                 "get_agent_status()", "set_notes(text)",
                 "get_action_history(before_step, limit)"):
        assert tool in prompt
    for action in ("GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND",
                   "SCAN", "FINISH", "START_ADJUST",
                   "END_ADJUST", "MOVE_FORWARD"):
        assert action in prompt
    assert "EXPLORE" not in prompt  # EXPLORE 已从动作集中移除
    assert "12 left turns" in flat
    assert "bird's-eye map image" in prompt
    assert "blank pixels mean" in flat
    # world-state 新字段与冷启动
    assert "new_keyframes" in prompt
    assert "recent_actions" in prompt
    assert "Cold start" in prompt
    # 工具调用 JSON 格式必须写进 prompt（VLM 才能正确发起调用）
    assert '"tool_call":' in prompt
    assert '"name": "<tool_name>"' in prompt
    # 报告按距离判定（走到目标附近即可，不要求目标在视野内）+ takeover 禁工具
    assert "Success is judged by DISTANCE, not by vision" in flat
    assert "tools are disabled" in flat
    # 写工具后状态刷新说明
    assert "the refreshed world state is" in flat
    # 已删除的工具不得再出现
    assert "look_at" not in prompt
    assert "merge_instances" not in prompt
    assert "undo_merge" not in prompt
    assert "ground_target" not in prompt
    assert "point_frame(" not in prompt
    assert "semantic_rejections" in prompt
    assert "ACCEPT only" in prompt
    assert "REJECT when" in prompt
    assert "UNCERTAIN when" in prompt
    assert "automatically" in prompt
    assert "observation_count" in prompt
    assert "query_memory" not in prompt
    assert "list_instances" not in prompt
    # 决策输出 JSON 不含 confidence 字段（confidence 只出现在工具返回里）
    tail = prompt.split("Finally reply with exactly one JSON object")[-1]
    assert "confidence" not in tail


def test_default_tool_limit_is_fifteen_and_prompt_discloses_hard_limit():
    loop = DecisionLoop(_ScriptedChat([]))
    assert loop.max_tool_rounds == 15
    assert DecisionLoop(
        _ScriptedChat([]), max_tool_rounds=99).max_tool_rounds == 15
    prompt = " ".join(loop._build_prompt(
        "world_state_updated", _state()).split())
    assert "most 15 calls per decision" in prompt
    assert "HARD per-decision limit" in prompt


def test_search_frames_tool_call_uses_standard_result_envelope():
    seen = []

    def search_frames(query, top_k=5):
        seen.append((query, top_k))
        return [{"frame_id": 7, "score": 0.8, "caption": "red cup"}]

    chat = _ScriptedChat([
        {"tool_call": {"name": "search_frames", "query": "red cup",
                       "top_k": 3}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"search_frames": search_frames}).decide(
            "world_state_updated", _state())
    assert seen == [("red cup", 3)]
    assert result.action == "GOTO_FRONTIER" and result.tool_calls == 1
    feedback = chat.calls[1][0].split("Tool result:")[-1]
    assert '"ok": true' in feedback
    assert '"tool": "search_frames"' in feedback
    assert '"state_changed": false' in feedback


def test_view_instance_attaches_labeled_instance_evidence_image():
    chat = _ScriptedChat([
        {"tool_call": {"name": "view_instance", "instance_id": 3}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"view_instance": lambda instance_id: b"instance-jpeg"}) \
        .decide("world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    label = "tool_instance_3_evidence"
    assert (label, b"instance-jpeg") in chat.calls[1][1]
    assert f'"image_ref": "{label}"' in chat.calls[1][0]


def test_tool_budget_disables_more_tools_and_forces_final_action():
    called = []
    chat = _ScriptedChat([
        {"tool_call": {"name": "search_frames", "query": "red cup"}},
        {"tool_call": {"name": "search_frames", "query": "blue cup"}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"search_frames": lambda query: called.append(query) or []},
        max_tool_rounds=1).decide("world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert result.tool_calls == 1
    assert called == ["red cup"]
    assert "# FINAL ACTION ONLY" in chat.calls[1][0]
    assert "hard tool-call limit is 1" in chat.calls[1][0]
    assert "1/1 calls have been used" in chat.calls[1][0]
    assert "tool_call is disabled after the hard limit" in chat.calls[2][0]


def test_default_limit_executes_fifteen_tools_then_requests_final_action():
    called = []
    replies = [
        {"tool_call": {"name": "search_frames", "query": f"q{i}"}}
        for i in range(15)
    ] + [{"action": "GOTO_FRONTIER", "target_id": "f0"}]
    chat = _ScriptedChat(replies)
    result = DecisionLoop(
        chat, tools={"search_frames": lambda query:
                     called.append(query) or []}).decide(
                         "world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert result.tool_calls == 15
    assert called == [f"q{i}" for i in range(15)]
    assert "Tool usage: 1/15; 14 calls remain." in chat.calls[1][0]
    assert "# FINAL ACTION ONLY" in chat.calls[15][0]
    assert "15/15 calls have been used" in chat.calls[15][0]


def test_repeated_residual_tool_calls_never_become_empty_action_fallback():
    called = []
    chat = _ScriptedChat([
        {"tool_call": {"name": "search_frames", "query": "first"}},
        {"tool_call": {"name": "search_frames", "query": "residual-1"}},
        {"tool_call": {"name": "search_frames", "query": "residual-2"}},
    ])
    result = DecisionLoop(
        chat, tools={"search_frames": lambda query: called.append(query) or []},
        max_tool_rounds=1).decide("world_state_updated", _state())
    assert result.action == "GOTO_INSTANCE"
    assert result.target_id == "1"
    assert result.validation == "forced_after_tool_limit"
    assert result.tool_calls == 1
    assert called == ["first"]


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
            {"action": "GOTO_FRONTIER", "target_id": "f0"},
        ])
        result = DecisionLoop(chat).decide("world_state_updated", _state())
        assert result.action == "GOTO_FRONTIER"


def test_arrival_actions_are_report_scan_or_explore():
    for action in ("REPORT_FOUND", "SCAN"):
        target_id = "1" if action == "REPORT_FOUND" else None
        result = DecisionLoop(_ScriptedChat([
            {"action": action, "target_id": target_id}])).decide(
                "arrival", _state())
        assert result.action == action
    # EXPLORE 已从动作集移除：会被拒绝并走重试
    chat = _ScriptedChat([
        {"action": "EXPLORE"},
        {"action": "SCAN"},
    ])
    result = DecisionLoop(chat).decide("arrival", _state())
    assert result.action == "SCAN"


def test_arrival_can_choose_another_instance_or_frontier():
    for action, target_id in (("GOTO_INSTANCE", "1"),
                              ("GOTO_FRONTIER", "f0")):
        result = DecisionLoop(_ScriptedChat([
            {"action": action, "target_id": target_id}])).decide(
                "arrival", _state())
        assert result.action == action and result.target_id == target_id


def test_vlm_explicitly_controls_adjustment_state_transitions():
    for event in ("world_state_updated", "arrival", "scan_complete"):
        result = DecisionLoop(_ScriptedChat([{
            "action": "START_ADJUST"
        }])).decide(event, _state())
        assert result.action == "START_ADJUST"
    for action in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP",
                   "LOOK_DOWN", "END_ADJUST"):
        result = DecisionLoop(_ScriptedChat([{
            "action": action
        }])).decide("adjustment", _state())
        assert result.action == action


def test_adjustment_pitch_actions_obey_configured_relative_limit():
    state = _state()
    state["adjustment"] = {
        "pitch_offset_steps": 1, "max_pitch_offset_steps": 1}
    chat = _ScriptedChat([
        {"action": "LOOK_UP"},
        {"action": "LOOK_DOWN"},
    ])
    result = DecisionLoop(chat).decide("adjustment", state)
    assert result.action == "LOOK_DOWN"


def test_atomic_motion_is_rejected_outside_adjustment():
    chat = _ScriptedChat([
        {"action": "MOVE_FORWARD"},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(chat).decide("world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"


def test_adjustment_rejects_high_level_actions_until_end_adjust():
    chat = _ScriptedChat([
        {"action": "REPORT_FOUND"},
        {"action": "END_ADJUST"},
    ])
    result = DecisionLoop(chat).decide("adjustment", _state())
    assert result.action == "END_ADJUST"


def test_scan_complete_reselects_globally_instead_of_reporting():
    for action, target_id in (("GOTO_INSTANCE", "1"),
                              ("GOTO_FRONTIER", "f0")):
        result = DecisionLoop(_ScriptedChat([{
            "action": action, "target_id": target_id
        }])).decide("scan_complete", _state())
        assert result.action == action
    # EXPLORE 已移除：被拒后重试
    chat = _ScriptedChat([
        {"action": "EXPLORE"},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(chat).decide("scan_complete", _state())
    assert result.action == "GOTO_FRONTIER"
    # 白名单放宽后 scan_complete 也允许 REPORT_FOUND
    result = DecisionLoop(_ScriptedChat([
        {"action": "REPORT_FOUND", "target_id": "1"}])).decide(
            "scan_complete", _state())
    assert result.action == "REPORT_FOUND"


def test_continue_navigation_is_en_route_only():
    state = _state()
    state["navigation"]["active_target"] = {
        "type": "frontier", "id": "active_frontier"}
    result = DecisionLoop(_ScriptedChat([
        {"action": "CONTINUE_NAVIGATION"}])).decide("en_route", state)
    assert result.action == "CONTINUE_NAVIGATION"
    assert result.target_id is None
    # 其他事件拒绝 CONTINUE_NAVIGATION → 校验失败重试
    for event in ("world_state_updated", "arrival", "scan_complete"):
        chat = _ScriptedChat([
            {"action": "CONTINUE_NAVIGATION"},
            {"action": "GOTO_FRONTIER", "target_id": "f0"},
        ])
        result = DecisionLoop(chat).decide(event, _state())
        assert result.action == "GOTO_FRONTIER"


def test_en_route_can_abandon_path_for_new_target():
    for action, target_id in (("GOTO_INSTANCE", "1"),
                              ("GOTO_FRONTIER", "f0")):
        result = DecisionLoop(_ScriptedChat([{
            "action": action, "target_id": target_id
        }])).decide("en_route", _state())
        assert result.action == action and result.target_id == target_id


def test_prompt_documents_continue_navigation_and_en_route_guidance():
    prompt = DecisionLoop(_ScriptedChat([]))._build_prompt(
        "en_route", _state())
    assert "CONTINUE_NAVIGATION" in prompt
    assert "orange" in prompt and "ACTIVE star" in prompt
    assert "abandons the current path" in prompt
    assert "mid-navigation" in prompt
    # world_state_updated 的普通决策也知晓该动作（契约部分），
    # 但 en_route 引导只出现在 en_route 事件。
    plain = DecisionLoop(_ScriptedChat([]))._build_prompt(
        "world_state_updated", _state())
    assert "CONTINUE_NAVIGATION" in plain
    assert "mid-navigation" not in plain


def test_report_found_requires_active_canonical_instance_id():
    state = _state(instances=[
        {"id": 1, "text": "basket", "reported": False},
        {"id": 2, "text": "other basket", "reported": False},
    ])
    for invalid in (
            {"action": "REPORT_FOUND"},
            {"action": "REPORT_FOUND", "target_id": "2"}):
        chat = _ScriptedChat([
            invalid,
            {"action": "REPORT_FOUND", "target_id": "1"},
        ])
        result = DecisionLoop(chat).decide("arrival", state)
        assert result.action == "REPORT_FOUND" and result.target_id == "1"


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
    stale = _state(instances=[])
    fresh = _state(instances=[
        {"id": 2, "text": "resolved red cup", "reported": False},
    ])

    def instantiate_points(frame_id, pixels_1000, label=""):
        return {"instances": [{"instance_id": 2, "observation_id": 9,
                               "association": "visual_relation"}],
                "geometry_rejections": []}

    chat = _ScriptedChat([
        {"tool_call": {"name": "instantiate_points", "frame_id": 5,
                       "pixels_1000": [[500, 500]], "label": "red cup"}},
        {"action": "GOTO_INSTANCE", "target_id": "2"},
    ])
    result = DecisionLoop(
        chat, tools={"instantiate_points": instantiate_points}).decide(
            "world_state_updated", stale, state_fn=lambda: fresh)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "2"
    assert result.tool_calls == 1
    retry_prompt = chat.calls[1][0]
    assert "World state after your write" in retry_prompt
    assert "resolved red cup" in retry_prompt


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


def test_topdown_map_is_prioritized_after_current_rgb():
    chat = _ScriptedChat([{"action": "END_ADJUST"}])
    images = [("current_observation", b"fresh-rgb")] + [
        (f"panorama_{i}", f"view-{i}".encode()) for i in range(4)]
    result = DecisionLoop(chat).decide(
        "adjustment", _state(), map_png=b"local-map", images=images)
    assert result.action == "END_ADJUST"
    labels = [name for name, _payload in chat.calls[0][1]]
    assert labels[:2] == ["current_observation", "topdown_map"]


def test_failed_write_tool_does_not_refresh_state():
    calls = []

    def instantiate_points(frame_id, pixels_1000, label=""):
        return {"error": "no valid pointing result"}

    chat = _ScriptedChat([
        {"tool_call": {"name": "instantiate_points", "frame_id": 5,
                       "pixels_1000": [[500, 500]], "label": "basket"}},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ])
    result = DecisionLoop(
        chat, tools={"instantiate_points": instantiate_points}).decide(
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
    width, height = Image.open(io.BytesIO(png)).size
    assert width >= 512 and height >= 512


def test_decision_pointcloud_map_has_markers_without_region_or_trajectory_layers():
    points = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.0, 0.2], [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.5], [2.0, 1.0, 0.0],
    ])
    colors = np.tile(np.array([[12, 34, 56]], dtype=np.uint8), (5, 1))
    png = render_pointcloud_topdown(
        points, colors, pose=(0.2, 0.2, 0.0),
        instances=[{"id": 3, "xy": (1.0, 1.0), "reported": False}],
        frontiers=[{"id": "f0", "xy": (2.0, 1.0)}],
        active_target={"id": 3, "xy": (1.0, 1.0)},
        floor_z=0.0, unit_per_m=1.0, max_plot_points=100)
    image = Image.open(io.BytesIO(png))
    pixels = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert (12, 34, 56) in pixels             # reconstructed RGB base
    assert (40, 80, 220) in pixels            # agent
    assert (150, 60, 200) in pixels           # frontier
    assert (245, 145, 25) in pixels           # active target
    assert (255, 232, 160) not in pixels       # old semantic region
    assert (220, 40, 40) not in pixels         # old trajectory


def test_decision_pointcloud_map_is_strict_xy_orthographic():
    """Changing height must not move a point in the decision image."""
    kwargs = dict(crop_center=(0.0, 0.0), crop_radius=2.0,
                  floor_z=0.0, unit_per_m=1.0,
                  min_image_side=128, max_image_side=128)
    colors = np.array([[20, 80, 140]], dtype=np.uint8)
    low = render_pointcloud_topdown(
        np.array([[0.6, -0.4, 0.1]]), colors, **kwargs)
    high = render_pointcloud_topdown(
        np.array([[0.6, -0.4, 2.0]]), colors, **kwargs)
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(low))),
        np.asarray(Image.open(io.BytesIO(high))))


def test_decision_pointcloud_pixel_fusion_is_order_independent():
    points = np.array([[0.1, 0.1, 0.2], [0.1, 0.1, 1.2]])
    colors = np.array([[220, 40, 20], [20, 80, 220]], dtype=np.uint8)
    kwargs = dict(crop_center=(0.0, 0.0), crop_radius=1.0,
                  floor_z=0.0, unit_per_m=1.0,
                  min_image_side=128, max_image_side=128)
    forward = render_pointcloud_topdown(points, colors, **kwargs)
    reverse = render_pointcloud_topdown(points[::-1], colors[::-1], **kwargs)
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(forward))),
        np.asarray(Image.open(io.BytesIO(reverse))))


def test_render_topdown_local_crop_marks_pose_and_active_target():
    png = render_topdown(
        _grid(), pose=(20, 15, 0.0),
        instances=[{"id": 1, "xy": (23, 15), "reported": False}],
        active_target={"id": 1, "type": "instance", "xy": (23, 15)},
        crop_center=(20, 15), crop_radius=5.0)
    image = Image.open(io.BytesIO(png))
    assert image.size[0] >= 512 and image.size[1] >= 512
    colors = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert (40, 80, 220) in colors
    assert (245, 145, 25) in colors


def test_render_topdown_distinguishes_semantic_gap_and_raw_frontier():
    grid = _grid()
    grid.semantic_coverage_enabled = True
    grid.semantic_inspected = np.zeros_like(grid.free)
    grid.semantic_inspected[10:20, 10:20] = grid.free[10:20, 10:20]
    layers = sk.frontier_layers(grid)
    png = render_topdown(
        grid, frontier_layers=layers, min_image_side=0,
        pixels_per_cell=4, show_legend=False)
    colors = {tuple(pixel) for pixel in np.asarray(
        Image.open(io.BytesIO(png))).reshape(-1, 3)}
    assert (250, 250, 250) in colors
    assert (255, 232, 160) in colors
    assert (35, 200, 210) in colors


def test_render_topdown_caps_elongated_global_map():
    png = render_topdown(
        _grid(h=1800, w=80), pixels_per_cell=4,
        min_image_side=512, max_image_side=1536)
    width, height = Image.open(io.BytesIO(png)).size
    assert min(width, height) >= 512
    assert max(width, height) <= 1536


def test_render_topdown_distinguishes_recent_trajectory_and_shows_status():
    grid = _grid(h=100, w=100)
    grid.unit_per_m = 1.0
    trajectory = [(float(i), 20.0) for i in range(40)]
    png = render_topdown(
        grid, trajectory=trajectory, pose=(39.0, 20.0, 0.0),
        frontier_stats={"raw_clusters": 7, "reachable": 2,
                        "selectable": 1, "filtered_cooldown": 1},
        recent_trajectory_points=10)
    pixels = {tuple(pixel) for pixel in np.asarray(
        Image.open(io.BytesIO(png))).reshape(-1, 3)}
    assert (150, 105, 105) in pixels
    assert (220, 40, 40) in pixels


def test_render_topdown_exposes_traversed_occupancy_conflicts():
    free = np.zeros((12, 12), dtype=bool)
    free[3:9, 3:9] = True
    obstacle = np.zeros_like(free)
    obstacle[1, 1] = True
    traversed = np.zeros_like(free)
    traversed[1, 1] = True       # obstacle conflict
    traversed[1, 2] = True       # geometry unknown
    grid = nav.OccupancyGrid(
        1.0, np.zeros(2), free, obstacle,
        observed=free | obstacle, traversed=traversed)
    png = render_topdown(
        grid, pixels_per_cell=8, min_image_side=0,
        max_image_side=0, show_legend=False)
    pixels = {tuple(pixel) for pixel in np.asarray(
        Image.open(io.BytesIO(png))).reshape(-1, 3)}
    assert (145, 205, 225) in pixels
    assert (235, 65, 165) in pixels


def test_explore_activates_highest_utility_frontier_instead_of_random_walk():
    agent = _make_agent()
    obs = SimpleNamespace(step_count=50)
    best = {
        "world": np.array([2.0, 0.0]),
        "path": [np.array([0.0, 0.0]), np.array([2.0, 0.0])],
        "key": (2, 0), "utility": 3.5, "scale": 0.5,
    }
    agent._last_frontier_step = obs.step_count
    agent._last_frontier_clusters = [best]
    action = agent._autonomous_explore_action(obs)
    assert action == int(Action.MOVE_FORWARD)
    assert agent.mode == "explore"
    assert agent._explore_follower is not None
    assert agent._explore_follower.scale == 0.5
    assert agent._active_frontier_key == best["key"]


def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def test_en_route_decision_throttles_by_interval():
    agent = _make_agent()
    agent.decision_loop = SimpleNamespace()
    agent._explore_follower = object()
    calls = []
    agent._decider_next = lambda obs, event: (
        calls.append(event) or SimpleNamespace(
            action="CONTINUE_NAVIGATION", target_id=None), None)
    agent._explore_follow = lambda obs: int(Action.MOVE_FORWARD)
    agent._last_en_route_step = 10
    assert agent._en_route_decision(SimpleNamespace(step_count=13)) is None
    assert calls == []  # 间隔内不触发
    action = agent._en_route_decision(SimpleNamespace(step_count=15))
    assert calls == ["en_route"]
    assert action == int(Action.MOVE_FORWARD)


def test_en_route_continue_keeps_follower():
    agent = _make_agent()
    agent.decision_loop = SimpleNamespace()
    agent._explore_follower = object()
    agent._active_frontier_key = (1, 2)
    agent._last_en_route_step = 0
    agent._decider_next = lambda obs, event: (
        SimpleNamespace(action="CONTINUE_NAVIGATION", target_id=None), None)
    agent._explore_follow = lambda obs: int(Action.MOVE_FORWARD)
    action = agent._en_route_decision(SimpleNamespace(step_count=5))
    assert action == int(Action.MOVE_FORWARD)
    assert agent._explore_follower is not None  # continue 不清 follower
    assert agent._active_frontier_key == (1, 2)


def test_en_route_goto_frontier_keeps_new_follower():
    # GOTO_FRONTIER 由 _decider_next 内部重建 follower；en_route 不能再清。
    agent = _make_agent()
    agent.decision_loop = SimpleNamespace()
    agent._explore_follower = object()
    agent._active_frontier_key = (1, 2)
    agent._last_en_route_step = 0
    agent._decider_next = lambda obs, event: (
        SimpleNamespace(action="GOTO_FRONTIER", target_id="f0"),
        int(Action.MOVE_FORWARD))
    agent._explore_follow = lambda obs: int(Action.MOVE_FORWARD)
    obs = SimpleNamespace(step_count=5)
    action = agent._en_route_decision(obs)
    assert action == int(Action.MOVE_FORWARD)
    assert agent._explore_follower is not None  # 新 follower 保留
    assert agent._active_frontier_key == (1, 2)
    assert agent._last_explore_plan == 5


def test_en_route_scan_abandons_path_and_starts_spin():
    agent = _make_agent()
    agent.decision_loop = SimpleNamespace()
    agent._explore_follower = object()
    agent._active_frontier_key = (1, 2)
    agent._last_en_route_step = 0
    agent._decider_next = lambda obs, event: (
        SimpleNamespace(action="SCAN", target_id=None), None)
    action = agent._en_route_decision(SimpleNamespace(step_count=5))
    assert action == int(Action.TURN_LEFT)
    assert agent._explore_follower is None
    assert agent._active_frontier_key is None
    assert agent._scanning is True


def test_en_route_decision_failure_keeps_following():
    agent = _make_agent()
    agent.decision_loop = SimpleNamespace()
    agent._explore_follower = object()
    agent._last_en_route_step = 0

    def boom(obs, event):
        raise RuntimeError("model unavailable")

    agent._decider_next = boom
    assert agent._en_route_decision(
        SimpleNamespace(step_count=5)) is None


def test_world_state_contains_text_evidence_and_no_anchor_table():
    agent = _make_agent()
    node, _ = agent.memory.remember(
        [3, 4, 0], "possible woven basket",
        evidence=[{"frame_id": 7}], candidate_id="c7")
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(
        agent, obs, grid=_grid(),
        frontiers=[{"world": np.array([10.0, 2.0]), "size": 7,
                    "reason": "semantic", "geometry_gain": 2,
                    "semantic_gain": 9}])
    row = state["instances"][0]
    assert row["id"] == node.iid
    assert row["text"] == "possible woven basket"
    assert "belief_anchors" not in state
    assert state["instances_total"] == 1
    assert state["instances_omitted_ids"] == []
    assert state["reported_instance_ids"] == []
    assert {"id", "path_cost_m", "branch_id", "geometry_gain",
            "semantic_gain", "failure_count", "recently_attempted",
            "novelty"} <= set(state["frontiers"][0])
    # dist_m（agent 当前位置到实例的直线距离）进表，供距离判定上报
    assert row["dist_m"] is not None and row["dist_m"] > 0


def test_world_state_uses_explicit_map_snapshot_pose():
    agent = _make_agent()
    agent.memory.add([3.0, 4.0, 0.0], "basket")
    agent._current_aligned_xy = lambda: (_ for _ in ()).throw(
        AssertionError("live pose must not replace snapshot pose"))
    agent._frontier_stats = {"raw_clusters": 3, "selectable": 1}
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(agent, obs, start_xy=(0.0, 0.0))
    assert state["instances"][0]["id"] == 1
    assert state["instances"][0]["dist_m"] == 5.0  # (3,4) 距起点 5m


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
    assert state["reported_instances"] == [{
        "id": reported.iid,
        "text": "already reported",
        "observation_count": 1,
        "report_claim_id": 1,
    }]
    assert state["report_claims"][0]["instance_id"] == reported.iid
    assert all(row["observation_count"] == 1
               for row in state["instances"])
    # 摘要按距离排序，最近的一定入选；reported 不在表中
    ids = [row["id"] for row in state["instances"]]
    assert 1 in ids and reported.iid not in ids
    assert all("reported" not in row for row in state["instances"])
    ids_in_order = [row["id"] for row in state["instances"]]
    assert ids_in_order == sorted(ids_in_order)  # 距离升序=id 升序


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


def test_navagent_memory_tool_updates_canonical_instance():
    agent = _make_agent()
    a = agent.memory.add([0, 0, 0], "view A")
    updated = agent._tool_update_instance(a.iid, "same basket, front view")
    assert updated["text"] == "same basket, front view"


def test_report_found_does_not_emit_duplicate_target_found():
    agent = _make_agent()
    node = agent.memory.add([0, 0, 0], "basket")
    agent.memory.mark_reported(node)
    agent.target_instance_id = node.iid
    agent.target_point = np.asarray(node.point)
    before = agent._reported_count
    action = agent._report_found(node.iid)
    assert action != int(Action.TARGET_FOUND)
    assert agent._reported_count == before
    assert agent.target_instance_id is None


def test_report_claim_records_supporting_observations_once():
    agent = _make_agent()
    node = agent.memory.add([0, 0, 0], "basket", candidate_id="c1")
    claim = agent.memory.claim(node, step=42)
    assert claim.instance_id == node.iid
    assert claim.observation_ids == tuple(node.observation_ids)
    assert node.reported and node.report_claim_id == claim.claim_id
    assert agent.memory.claim(node, step=43) is None
    assert len(agent.memory.report_claims) == 1


def test_loop_closure_refreshes_observations_then_reselects_canonical_point():
    agent = _make_agent()
    first = agent.memory.new_observation(
        [0, 0, 0], "chair front", evidence={"point_score": 0.2},
        frame_id=1, candidate_id="c1")
    node = agent.memory.create_instance(first)
    second = agent.memory.new_observation(
        [1, 0, 0], "chair side", evidence={"point_score": 0.9},
        frame_id=2, candidate_id="c2")
    agent.memory.attach_observation(node, second)
    seen = []

    def resolve_candidates(candidate_ids):
        seen.extend(candidate_ids)
        return {
            "c1": {"found": True, "point": [10, 0, 0]},
            "c2": {"found": True, "point": [2, 0, 0]},
        }

    agent.client = SimpleNamespace(resolve_candidates=resolve_candidates)
    agent._refresh_memory_candidates([node.iid])
    assert seen == ["c1", "c2"]
    assert np.allclose(first.point, [10, 0, 0])
    assert np.allclose(second.point, [2, 0, 0])
    assert np.allclose(node.point, [2, 0, 0])
    assert node.candidate_id == "c2"


def _jpeg_bytes(size=(100, 100)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeResolverVLM:
    enabled = True

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def set_trace_context(self, **kwargs):
        pass

    def chat_json(self, prompt, images, trace_kind="json"):
        self.calls.append((prompt, images))
        return self.reply


def _ingest_hit():
    return {"found": True, "point": [1.0, 2.0, 0.0],
            "text": "a kitchen counter with several objects on it",
            "frame_id": 5, "candidate_id": "c5",
            "bbox": [10, 10, 50, 50], "point_score": 0.9}


def test_ingest_generates_instance_level_text():
    """实例化不再调专用 VLM 改写描述：实例文本直接取 caption/label。"""
    agent = _make_agent()
    agent.vlm = _FakeResolverVLM(None)
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"overlay-jpeg"),
        get_frame_image=lambda fid: ({"found": True}, _jpeg_bytes()))
    obs = SimpleNamespace(step_count=50)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    node = agent.memory.nodes[0]
    assert node.text == "a kitchen counter with several objects on it"
    assert agent.vlm.calls == []  # 实例化路径不调 VLM


def test_ingest_keeps_caption_text_when_vlm_unavailable():
    obs = SimpleNamespace(step_count=50)
    agent = _make_agent()               # 默认 VLM disabled
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert agent.memory.nodes[0].text == \
        "a kitchen counter with several objects on it"
    # VLM 可用但调用失败：同样保留 caption 文本
    agent2 = _make_agent()
    agent2.vlm = _FakeResolverVLM(None)
    agent2.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"x"),
        get_frame_image=lambda fid: ({"found": False}, b""))
    agent2._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert agent2.memory.nodes[0].text == \
        "a kitchen counter with several objects on it"


def test_ingest_does_not_redescribe_existing_instance():
    agent = _make_agent()
    agent.vlm = _FakeResolverVLM({
        "decision": "NEW", "instance_id": None,
        "description": "new description", "reason": "first",
    })
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda cid: ({"found": True}, b"overlay-jpeg"),
        get_frame_image=lambda fid: ({"found": True}, _jpeg_bytes()))
    obs = SimpleNamespace(step_count=50)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    agent._ingest_semantic_hits(obs, [_ingest_hit()], select=False)
    assert len(agent.memory.nodes) == 1      # 同 candidate_id 只更新
    assert len(agent.vlm.calls) == 0         # 实例化路径不调 VLM


def test_ingest_duplicate_review_attaches_cross_frame_observation():
    """跨帧同一物体：3m 内挂起复核，VLM 裁决 DUPLICATE 后并入既有实例。"""
    agent = _make_agent()
    obs = SimpleNamespace(step_count=50)
    first = _ingest_hit()
    second = dict(first, frame_id=6, candidate_id="c6",
                  point=[1.05, 2.02, 0.0], pixel=[14, 14])
    agent._ingest_semantic_hits(obs, [first], select=False)
    agent._ingest_semantic_hits(obs, [second], select=False)
    assert len(agent.memory.nodes) == 1  # 复核挂起，未新建
    review = agent._last_dup_reviews[0]
    assert review["neighbors"][0]["instance_id"] == 1
    out = agent._tool_resolve_duplicate(
        review["observation_id"], "DUPLICATE", duplicate_of=1,
        text="dark wooden chair, two views")
    assert out["resolved"] == "duplicate"
    node = agent.memory.nodes[0]
    assert len(node.observation_ids) == 2
    assert node.text == "dark wooden chair, two views"


def test_ingest_suspends_nearby_candidate_without_vlm():
    """3m 内已有实例：挂起 duplicate_review，不调任何 VLM。"""
    agent = _make_agent()
    agent.memory.add(
        [1.0, 2.0, 0.0], "nearby chair", frame_id=4,
        candidate_id="c4")
    agent.vlm = _FakeResolverVLM(None)
    second = dict(_ingest_hit(), frame_id=6, candidate_id="c6",
                  point=[1.05, 2.02, 0.0])
    agent._ingest_semantic_hits(
        SimpleNamespace(step_count=50), [second], select=False)
    assert len(agent.memory.nodes) == 1
    assert agent._proposals["c6"]["status"] == "duplicate_review"
    assert agent.vlm.calls == []


def test_ingest_duplicate_review_new_creates_instance():
    """复核裁决 NEW：挂起的 observation 建成独立实例。"""
    agent = _make_agent()
    agent.memory.add(
        [1.0, 2.0, 0.0], "nearby chair", frame_id=4,
        candidate_id="c4")
    second = dict(_ingest_hit(), frame_id=6, candidate_id="c6",
                  point=[1.05, 2.02, 0.0])
    agent._ingest_semantic_hits(
        SimpleNamespace(step_count=50), [second], select=False)
    assert len(agent.memory.nodes) == 1
    assert agent._proposals["c6"]["status"] == "duplicate_review"
    oid = agent._last_dup_reviews[0]["observation_id"]
    out = agent._tool_resolve_duplicate(oid, "NEW")
    assert out["resolved"] == "new"
    assert len(agent.memory.nodes) == 2


def test_suspended_evidence_replay_does_not_create_another_observation():
    """同一挂起证据重放：不新建 observation，也不重复挂起复核。"""
    agent = _make_agent()
    agent.memory.add(
        [1.0, 2.0, 0.0], "nearby chair", frame_id=4,
        candidate_id="c4")
    hit = dict(_ingest_hit(), frame_id=6, candidate_id="c6",
               point=[1.05, 2.02, 0.0])
    obs = SimpleNamespace(step_count=50)
    agent._ingest_semantic_hits(obs, [hit], select=False)
    agent._ingest_semantic_hits(obs, [hit], select=False)
    assert len(agent.memory.nodes) == 1
    assert len(agent.memory.observations) == 2


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


def test_scan_flushes_tail_map_before_waiting_without_auto_retrieving():
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
    assert order == ["flush", "wait"]


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


def test_navagent_view_instance_prefers_candidate_overlay():
    agent = _make_agent()
    node, _ = agent.memory.remember(
        [1, 2, 0], "red cup", frame_id=7, candidate_id="c7")
    agent.client = SimpleNamespace(
        get_candidate_evidence=lambda candidate_id:
            ({"found": True}, b"overlay"),
        get_frame_image=lambda frame_id: ({"found": True}, b"frame"))
    assert agent._tool_view_instance(node.iid) == b"overlay"


# ---------------------------------------------------------- nav 卡死决策（方案 A）
def test_nav_failed_event_allows_frontier_and_has_guidance():
    prompt = DecisionLoop(_ScriptedChat([]))._build_prompt(
        "nav_failed", _state())
    assert "repeated collisions" in prompt
    assert "blocked_target" in prompt
    assert "removed from the instances table" in prompt
    result = DecisionLoop(_ScriptedChat([
        {"action": "GOTO_FRONTIER", "target_id": "f0"}])).decide(
        "nav_failed", _state())
    assert result.action == "GOTO_FRONTIER"
    # 指引只出现在 nav_failed 事件
    plain = DecisionLoop(_ScriptedChat([]))._build_prompt(
        "world_state_updated", _state())
    assert "repeated collisions" not in plain


def test_goto_unreachable_instance_rejected_but_report_allowed():
    state = _state(instances=[
        {"id": 1, "text": "basket", "reported": False},
        {"id": 2, "text": "other basket", "reported": False},
    ])
    state["instances_unreachable_ids"] = ["2"]
    # GOTO 不可达实例被校验拒绝 → 重试后选可达实例
    loop = DecisionLoop(_ScriptedChat([
        {"action": "GOTO_INSTANCE", "target_id": "2"},
        {"action": "GOTO_INSTANCE", "target_id": "1"},
    ]))
    result = loop.decide("nav_failed", state)
    assert result.action == "GOTO_INSTANCE" and result.target_id == "1"
    # REPORT_FOUND 不可达实例仍放行：agent 可能就停在目标旁边直接确认
    state["navigation"]["active_target"] = {"type": "instance", "id": "2"}
    result = DecisionLoop(_ScriptedChat([
        {"action": "REPORT_FOUND", "target_id": "2"}])).decide(
        "nav_failed", state)
    assert result.action == "REPORT_FOUND"


def test_report_found_near_non_active_instance_allowed():
    # 走到目标附近即可上报：非 active 但 dist_m 在阈值内 → 放行
    state = _state(instances=[
        {"id": 1, "text": "basket", "reported": False},
        {"id": 2, "text": "other basket", "reported": False,
         "dist_m": 0.4},
    ])
    result = DecisionLoop(_ScriptedChat([
        {"action": "REPORT_FOUND", "target_id": "2"}])).decide(
        "world_state_updated", state)
    assert result.action == "REPORT_FOUND" and result.target_id == "2"


def test_report_found_far_non_active_instance_rejected():
    # 远离实例且非 active → 拒绝，重试后换动作
    state = _state(instances=[
        {"id": 1, "text": "basket", "reported": False},
        {"id": 2, "text": "other basket", "reported": False,
         "dist_m": 5.2},
    ])
    result = DecisionLoop(_ScriptedChat([
        {"action": "REPORT_FOUND", "target_id": "2"},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])).decide("world_state_updated", state)
    assert result.action == "GOTO_FRONTIER"


if __name__ == "__main__":
    test_goto_accepts_any_unreported_instance()
    test_nav_failed_event_allows_frontier_and_has_guidance()
    test_goto_unreachable_instance_rejected_but_report_allowed()
    test_report_found_near_non_active_instance_allowed()
    test_report_found_far_non_active_instance_rejected()
    test_prompt_documents_action_effects_tool_returns_and_no_confidence()
    test_search_frames_tool_call_uses_standard_result_envelope()
    test_view_instance_attaches_labeled_instance_evidence_image()
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
    test_decision_pointcloud_map_has_markers_without_region_or_trajectory_layers()
    test_decision_pointcloud_map_is_strict_xy_orthographic()
    test_decision_pointcloud_pixel_fusion_is_order_independent()
    test_render_topdown_local_crop_marks_pose_and_active_target()
    test_render_topdown_distinguishes_semantic_gap_and_raw_frontier()
    test_render_topdown_caps_elongated_global_map()
    test_render_topdown_distinguishes_recent_trajectory_and_shows_status()
    test_render_topdown_exposes_traversed_occupancy_conflicts()
    test_world_state_contains_text_evidence_and_no_anchor_table()
    test_world_state_uses_explicit_map_snapshot_pose()
    test_world_state_summarizes_instances_beyond_k()
    test_world_state_summary_prefers_nearest_newest_relevant()
    test_world_state_truncates_instance_text()
    test_omitted_instance_is_valid_goto_target()
    test_navagent_memory_tool_updates_canonical_instance()
    test_report_found_does_not_emit_duplicate_target_found()
    test_report_claim_records_supporting_observations_once()
    test_loop_closure_refreshes_observations_then_reselects_canonical_point()
    test_ingest_generates_instance_level_text()
    test_ingest_keeps_caption_text_when_vlm_unavailable()
    test_ingest_does_not_redescribe_existing_instance()
    test_ingest_visual_relation_attaches_cross_frame_observation()
    test_ingest_skips_identity_vlm_without_new_marked_photo()
    test_ingest_rejects_same_for_candidate_without_visual_evidence()
    test_wait_for_captions_logs_timeout_and_swallows_errors()
    test_scan_flushes_tail_map_before_waiting_and_retrieving()
    test_navagent_search_instances_uses_vlm_keywords()
    test_navagent_view_instance_prefers_candidate_overlay()
    print("decider tests passed")
