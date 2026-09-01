"""到达决策、扫描与 instance 入库回归测试。"""

import os
import sys
import json
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import navigator as nav
from agents.nav_agent import NavAgent
from benchmark_api import Action
from decision import DecisionResult


def _obs(step=100, rgb_shape=(48, 64, 3), previous_action=None):
    return SimpleNamespace(
        step_count=step, goal_text="Find a gray fabric sofa",
        target_mode="any", target_count=None,
        rgb=np.zeros(rgb_shape, dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=previous_action)


class _MockClient:
    def ground_frame(self, rgb, text):
        raise AssertionError(
            "arrival decision must not call current-frame ground_frame")

    def get_all_poses(self):
        return None, []


def _make_agent():
    agent = NavAgent()
    agent.target_text = "gray fabric sofa"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    return agent


def test_arrival_decision_goes_directly_to_vlm_without_grounding():
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "possibly a gray sofa")
    agent.target_instance_id = node.iid
    agent.target_candidate_id = "candidate-1"
    agent.client = _MockClient()
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"current-jpeg")
    agent._build_decider_input = lambda obs: ({"instances": [], "task": {}}, None)
    agent.decision_loop = SimpleNamespace(decide=lambda *a, **k:
        DecisionResult("REPORT_FOUND", str(node.iid),
                       reason="current evidence is sufficient"))
    result, action = agent._arrival_vlm_decision(_obs())
    assert result.action == "REPORT_FOUND"
    assert action is None
    assert node in agent.memory.available()


def test_immediate_goto_arrival_runs_arrival_decision_and_reports():
    """A zero-length GOTO path must not be converted into exploration."""
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "gray fabric sofa")
    agent.client = _MockClient()
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"current-jpeg")
    agent._build_decider_input = lambda obs, **kwargs: (
        {"instances": [], "task": {}}, None)
    replies = {
        "world_state_updated": DecisionResult(
            "GOTO_INSTANCE", str(node.iid)),
        "arrival": DecisionResult("REPORT_FOUND", str(node.iid)),
    }
    agent.decision_loop = SimpleNamespace(
        decide=lambda event, *args, **kwargs: replies[event])

    def steer(_obs, result):
        agent.target_instance_id = int(result.target_id)
        agent.target_point = np.asarray(node.point)
        agent.mode = "nav"
        return True

    agent._apply_decider_steering = steer
    agent._nav_action = lambda obs: (None, True, False)
    _result, action = agent._decider_next(
        _obs(), "world_state_updated")
    assert action == int(Action.TARGET_FOUND)
    assert agent._reported_count == 1
    assert node.reported is True


def test_failed_instance_plan_clears_stale_active_target():
    agent = _make_agent()
    node = agent.memory.add([3, 4, 0], "gray fabric sofa")
    agent._plan_to_target = lambda obs: False
    ok = agent._apply_decider_steering(
        _obs(), DecisionResult("GOTO_INSTANCE", str(node.iid)))
    assert ok is False
    assert agent.mode == "explore"
    assert agent.target_instance_id is None
    assert agent.target_point is None


def test_adjustment_executes_one_vlm_motion_per_observation_then_resumes():
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "possibly a gray sofa")
    agent.target_instance_id = node.iid
    agent.target_candidate_id = "candidate-1"
    agent.client = _MockClient()
    encoded_steps = []
    agent.vlm = SimpleNamespace(
        encode_rgb=lambda rgb: encoded_steps.append(len(encoded_steps)) or
        f"rgb-{len(encoded_steps)}".encode())
    agent._build_decider_input = lambda obs, **kwargs: (
        {"instances": [], "task": {}, "step": obs.step_count}, None)
    replies = {
        "adjustment": [DecisionResult("MOVE_FORWARD"),
                       DecisionResult("END_ADJUST")],
        "arrival": [DecisionResult(
            "REPORT_FOUND", str(node.iid), reason="view is now sufficient")],
    }
    decision_inputs = []

    def decide(event, *args, **kwargs):
        decision_inputs.append((event, list(kwargs.get("images") or [])))
        return replies[event].pop(0)

    agent.decision_loop = SimpleNamespace(decide=decide, logger=None)
    first = agent._start_adjustment(
        _obs(step=100), "arrival",
        [("current_observation", b"stale"),
         ("selected_candidate", b"evidence")])
    assert first == int(Action.MOVE_FORWARD)
    assert agent._adjusting and agent._adjust_steps == 1
    # In production act() records the fresh observation after MOVE_FORWARD.
    agent._adjust_progress["fresh_views"] = 1
    second = agent._adjustment_action(_obs(step=101))
    assert second == int(Action.TARGET_FOUND)
    assert not agent._adjusting
    assert agent._reported_count == 1
    # One fresh image for each adjustment round and one for resumed arrival.
    assert len(encoded_steps) == 3
    adjustment_inputs = [images for event, images in decision_inputs
                         if event == "adjustment"]
    assert adjustment_inputs[0][0] == ("current_observation", b"rgb-1")
    assert adjustment_inputs[1][0] == ("current_observation", b"rgb-2")
    assert ("current_observation", b"stale") not in adjustment_inputs[0]
    assert ("selected_candidate", b"evidence") in adjustment_inputs[0]
    assert ("selected_candidate", b"evidence") in adjustment_inputs[1]


def test_adjustment_state_has_pose_target_collision_and_local_map():
    agent = _make_agent()
    node = agent.memory.add([12, 10, 0], "gray fabric sofa")
    agent.target_instance_id = node.iid
    agent.target_text = "gray fabric sofa"
    agent.client = _MockClient()
    agent.grid = nav.OccupancyGrid(
        1.0, np.array([0.0, 0.0]),
        np.ones((30, 30), dtype=bool),
        np.zeros((30, 30), dtype=bool))
    agent.follower = nav.PathFollower(scale=1.0)
    agent.follower.anchor_frame = 3
    agent.follower.x, agent.follower.y, agent.follower.yaw = 10.0, 10.0, 0.5
    agent.adjust_map_radius_m = 2.0
    # 决策图必须来自带颜色的 mapping 快照；该 client 仅用于到达逻辑，
    # 因而直接提供最小快照，避免把单元测试耦合到 RPC mock。
    agent._frontier_grid = agent.grid
    agent._frontier_pointcloud = (
        np.array([[9.0, 9.0, 0.0], [10.0, 10.0, 0.0],
                  [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]]),
        np.array([[80, 80, 80], [120, 120, 120],
                  [160, 160, 160], [200, 200, 200]], dtype=np.uint8))
    agent._frontier_pose = (10.0, 10.0, 0.5)
    agent._frontier_scale = 1.0
    agent._last_frontier_step = 100
    agent._last_decision_snapshot_step = 100
    agent._adjust_source_event = "arrival"
    agent._last_motion_failed = True

    state, map_png = agent._adjustment_state(_obs(
        previous_action=int(Action.MOVE_FORWARD)))
    adjustment = state["adjustment"]
    assert adjustment["current_pose"] == {
        "x_m": 10.0, "y_m": 10.0, "yaw_deg": 28.6}
    assert adjustment["active_target"]["type"] == "instance"
    assert adjustment["active_target"]["id"] == node.iid
    assert adjustment["active_target"]["distance_m"] == 2.0
    assert adjustment["previous_action"] == {
        "id": int(Action.MOVE_FORWARD), "name": "MOVE_FORWARD"}
    assert adjustment["collision"]["detected"] is True
    assert adjustment["pitch_offset_steps"] == 0
    assert adjustment["max_pitch_offset_steps"] == 1
    assert adjustment["local_topdown_map"]["attached"] is True
    image = Image.open(io.BytesIO(map_png))
    assert image.size == (1024, 1024)
    colors = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert (40, 80, 220) in colors       # YOU
    assert (245, 145, 25) in colors      # active TARGET


def test_adjustment_can_tilt_camera_and_auto_levels_before_resume():
    agent = _make_agent()
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"tilted-evidence")
    agent._build_decider_input = lambda obs, **kwargs: (
        {"instances": [], "task": {}, "step": obs.step_count}, None)
    replies = {
        "adjustment": [DecisionResult("LOOK_DOWN"),
                       DecisionResult("END_ADJUST")],
        "world_state_updated": [DecisionResult("SCAN")],
    }
    resumed_images = []

    def decide(event, *args, **kwargs):
        if event == "world_state_updated":
            resumed_images.extend(kwargs.get("images") or [])
        return replies[event].pop(0)

    agent.decision_loop = SimpleNamespace(decide=decide, logger=None)
    assert agent._start_adjustment(
        _obs(step=200), "world_state_updated") == int(Action.LOOK_DOWN)
    assert agent._adjust_pitch_steps == -1
    # The next observation is fresh; direct private-method tests emulate the
    # accounting normally performed by NavAgent.act().
    agent._adjust_progress["fresh_views"] = 1
    agent._adjust_progress["new_keyframes"] = 1
    # END_ADJUST first emits the inverse benchmark action; the original event
    # resumes only after a fresh neutral-camera observation arrives.
    assert agent._adjustment_action(_obs(step=201)) == int(Action.LOOK_UP)
    assert agent._adjusting and agent._adjust_pitch_steps == 0
    assert agent._adjustment_action(_obs(step=202)) == int(Action.TURN_LEFT)
    assert not agent._adjusting
    assert ("adjustment_direct_evidence", b"tilted-evidence") in resumed_images


def test_adjustment_stops_when_target_cumulative_budget_is_exhausted():
    """同一 target 的跨 session 预算耗尽时，不再请求 VLM。"""
    agent = _make_agent()
    agent._adjustment_key = ("candidate", "c1")
    agent._adjustment_budgets[agent._adjustment_key] = {
        "sessions": 1, "steps": agent.adjust_max_total_steps_per_target,
        "turns": 0, "turn_streak": 0, "last_turn": None,
    }
    abandoned = []
    agent._abandon_adjustment_target = lambda obs, reason: (
        abandoned.append(reason) or int(Action.TURN_LEFT))
    agent.decision_loop = SimpleNamespace(
        decide=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("budget exhaustion must not query VLM")))

    assert agent._adjustment_action(_obs()) == int(Action.TURN_LEFT)
    assert abandoned == ["target_total_adjustment_steps_exhausted"]


def test_adjustment_rejects_repeated_same_turn_before_another_loop():
    """第三次连续同向转动直接冷却 target，避免原地旋转。"""
    agent = _make_agent()
    key = ("candidate", "c1")
    agent._adjustment_key = key
    agent._adjustment_budgets[key] = {
        "sessions": 1, "steps": 2, "turns": 2,
        "turn_streak": 2, "last_turn": "TURN_LEFT",
    }
    agent._build_decider_input = lambda obs, **kwargs: ({}, None)
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"rgb")
    agent.decision_loop = SimpleNamespace(
        decide=lambda *args, **kwargs: DecisionResult("TURN_LEFT"),
        logger=None)
    abandoned = []
    agent._abandon_adjustment_target = lambda obs, reason: (
        abandoned.append(reason) or int(Action.TURN_RIGHT))

    assert agent._adjustment_action(_obs()) == int(Action.TURN_RIGHT)
    assert abandoned == ["repeated_same_turn"]


def test_active_exploration_refreshes_frontiers_after_end_adjust():
    agent = _make_agent()
    agent._adjusting = True
    agent._adjust_source_event = "world_state_updated"
    refreshed = []
    agent._plan_exploration = lambda obs, select=True: refreshed.append(select)
    agent._decider_next = lambda obs, event, images=None: (
        DecisionResult("EXPLORE"), int(Action.TURN_LEFT))
    action = agent._end_adjustment_and_resume(_obs(step=102), "vlm")
    assert action == int(Action.TURN_LEFT)
    assert refreshed == [False]


def test_action_trace_links_actual_action_frame_and_decision():
    path = Path(".action_trace_test.jsonl")
    if path.exists():
        path.unlink()
    agent = _make_agent()
    agent._action_trace_path = str(path)
    agent._last_feed_info = {
        "frame_id": 101, "is_keyframe": True,
        "queued_keyframes": 2, "busy": False,
    }
    agent._last_decision_output = {
        "step": 100, "event": "adjustment",
        "output": {"action": "MOVE_FORWARD"},
    }
    agent._trace_action(_obs(step=100), int(Action.MOVE_FORWARD))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["actual_action"]["name"] == "MOVE_FORWARD"
    assert record["mapping_frame_id"] == 101
    assert record["last_decision"]["output"]["action"] == "MOVE_FORWARD"
    path.unlink()


def _hit(point, conf=0.8, frame_id=1, candidate_id=None, text="cup-like object"):
    return {"found": True, "point": list(point), "point_score": conf,
            "frame_id": frame_id, "candidate_id": candidate_id,
            "bbox": [0, 0, 8, 8], "depth_std": 0.8, "text": text}


def test_every_valid_3d_hit_becomes_an_instance_even_low_confidence():
    agent = _make_agent()
    agent._choose_high_level_target = lambda *args, **kwargs: 7
    result = agent._ingest_semantic_hits(
        _obs(), [_hit([3, 4, 0], conf=0.1, candidate_id="c1")])
    assert result == 7
    assert len(agent.memory.available()) == 1
    assert agent.memory.nodes[0].text == "cup-like object"
    assert agent.memory.nodes[0].evidence[0]["point_score"] == 0.1


def test_same_candidate_updates_but_different_candidates_do_not_auto_merge():
    agent = _make_agent()
    agent._choose_high_level_target = lambda *args, **kwargs: None
    agent._ingest_semantic_hits(_obs(), [
        _hit([3, 4, 0], candidate_id="c1"),
        _hit([3.1, 4, 0], frame_id=2, candidate_id="c2")])
    assert len(agent.memory.nodes) == 2
    agent._ingest_semantic_hits(_obs(step=101), [
        _hit([3.2, 4, 0], frame_id=3, candidate_id="c1")])
    assert len(agent.memory.nodes) == 2


def test_scan_is_general_panorama_without_target_grounding():
    agent = _make_agent()
    calls = {"ground_frame": 0, "ground_object": 0}

    class Client:
        def ground_frame(self, rgb, text):
            calls["ground_frame"] += 1
            raise AssertionError("SCAN must not verify the current target")

        def ground_object(self, text, top_k=3):
            calls["ground_object"] += 1
            return []

    agent.client = Client()
    agent.target_instance_id = agent.memory.add(
        [1, 2, 0], "unresolved object").iid
    captured = {}

    def choose(obs, event="world_state_updated", images=None):
        captured["event"] = event
        captured["images"] = images
        return 9

    agent._choose_high_level_target = choose
    agent._refresh_memory_candidates = lambda: None
    for step in range(11):
        assert agent._handle_scan(_obs(step=step)) == int(Action.TURN_LEFT)
    assert agent._handle_scan(_obs(step=11)) == 9
    assert calls == {"ground_frame": 0, "ground_object": 0}
    assert captured["event"] == "scan_complete"
    assert len(captured["images"]) == 4


if __name__ == "__main__":
    test_arrival_decision_goes_directly_to_vlm_without_grounding()
    test_adjustment_executes_one_vlm_motion_per_observation_then_resumes()
    test_adjustment_state_has_pose_target_collision_and_local_map()
    test_active_exploration_refreshes_frontiers_after_end_adjust()
    test_action_trace_links_actual_action_frame_and_decision()
    test_every_valid_3d_hit_becomes_an_instance_even_low_confidence()
    test_same_candidate_updates_but_different_candidates_do_not_auto_merge()
    test_scan_is_general_panorama_without_target_grounding()
    print("arrival tests passed")
