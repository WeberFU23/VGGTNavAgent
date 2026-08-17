"""到达、扫描、视觉伺服与 instance 入库回归测试。"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent
from benchmark_api import Action
from decision import DecisionResult


def _obs(step=100, rgb_shape=(48, 64, 3)):
    return SimpleNamespace(
        step_count=step, goal_text="Find a gray fabric sofa",
        target_mode="any", target_count=None,
        rgb=np.zeros(rgb_shape, dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=None)


class _MockClient:
    def __init__(self, ground_frame_result=None):
        self._gf = ground_frame_result or {"found": False}

    def ground_frame(self, rgb, text):
        if isinstance(self._gf, Exception):
            raise self._gf
        return self._gf

    def get_all_poses(self):
        return None, []


def _make_agent():
    agent = NavAgent()
    agent.target_text = "gray fabric sofa"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    return agent


def _big_centered_bbox(w=64, h=48):
    return [w * 0.3, h * 0.2, w * 0.7, h * 0.8]


def test_arrival_reports_without_decider_when_grounding_succeeds():
    agent = _make_agent()
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    assert agent._arrival_decision(_obs()) == "report_found"


def test_arrival_scans_on_low_score_or_error():
    agent = _make_agent()
    agent.client = _MockClient({"found": True, "score": 0.3})
    assert agent._arrival_decision(_obs()) == "scan"
    agent.client = _MockClient(RuntimeError("server down"))
    assert agent._arrival_decision(_obs()) == "scan"


def test_arrival_vlm_can_leave_instance_without_rejecting_it():
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "possibly a gray sofa")
    agent.target_instance_id = node.iid
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"current-jpeg")
    agent._build_decider_input = lambda obs: ({"instances": [], "task": {}}, None)
    agent.decision_loop = SimpleNamespace(decide=lambda *a, **k:
        DecisionResult("EXPLORE", reason="uncertain"))
    assert agent._arrival_decision(_obs()) == "explore"
    assert node in agent.memory.available()


def test_servo_reports_and_marks_current_instance():
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "gray fabric sofa")
    agent.target_instance_id = node.iid
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    action = agent._confirm_and_report(_obs())
    assert action == int(Action.TARGET_FOUND)
    assert node.reported
    assert agent.memory.count_reported() == 1


def test_servo_timeout_enters_scan_instead_of_reporting():
    agent = _make_agent()
    node = agent.memory.add([1, 2, 0], "uncertain sofa")
    agent.target_instance_id = node.iid
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.servo_max_steps = 1
    agent.client = _MockClient({"found": False})
    assert agent._confirm_and_report(_obs()) == int(Action.TURN_LEFT)
    assert agent._servo_step(_obs(step=101)) == int(Action.TURN_LEFT)
    assert agent._scanning and not node.reported


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
    assert calls == {"ground_frame": 0, "ground_object": 1}
    assert captured["event"] == "scan_complete"
    assert len(captured["images"]) == 4


if __name__ == "__main__":
    test_arrival_reports_without_decider_when_grounding_succeeds()
    test_arrival_scans_on_low_score_or_error()
    test_arrival_vlm_can_leave_instance_without_rejecting_it()
    test_servo_reports_and_marks_current_instance()
    test_servo_timeout_enters_scan_instead_of_reporting()
    test_every_valid_3d_hit_becomes_an_instance_even_low_confidence()
    test_same_candidate_updates_but_different_candidates_do_not_auto_merge()
    test_scan_is_general_panorama_without_target_grounding()
    print("arrival tests passed")
