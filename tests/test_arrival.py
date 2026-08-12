"""Phase 3 单元测试：分级置信度账本 + 到达例程 + 末端视觉伺服。

mock client/VLM，不依赖 server 与真实模型；覆盖确认/否决/超时三分支。

    python tests/test_arrival.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evidence import ObservationLedger
from agents.nav_agent import NavAgent
from benchmark_api import Action
from decision import DecisionResult


# ----------------------------------------------------------------------
# ObservationLedger 分级置信度
# ----------------------------------------------------------------------
def test_ledger_single_obs_stays_belief():
    ledger = ObservationLedger(min_obs=2, min_pose_sep=0.5)
    outcome, anchor = ledger.add_observation(
        "sofa", [1, 2, 0], 0.8, merge_dist=0.75, frame_id=10, step=5,
        obs_xy=[0, 0])
    assert outcome == "belief"
    assert anchor.n_obs == 1


def test_ledger_independent_obs_promotes():
    ledger = ObservationLedger(min_obs=2, min_pose_sep=0.5)
    ledger.add_observation("sofa", [1, 2, 0], 0.8, merge_dist=0.75,
                           frame_id=10, step=5, obs_xy=[0, 0])
    outcome, anchor = ledger.add_observation(
        "sofa", [1.1, 2, 0], 0.85, merge_dist=0.75, frame_id=20, step=9,
        obs_xy=[2, 0])                      # 不同位姿 -> 独立帧
    assert outcome == "confirmed"
    assert anchor.n_obs == 2
    assert anchor.score == pytest.approx(0.85)      # 保留高置信


def test_ledger_same_pose_is_duplicate():
    ledger = ObservationLedger(min_obs=2, min_pose_sep=0.5)
    ledger.add_observation("sofa", [1, 2, 0], 0.8, merge_dist=0.75,
                           frame_id=10, step=5, obs_xy=[0, 0])
    outcome, anchor = ledger.add_observation(
        "sofa", [1, 2, 0], 0.9, merge_dist=0.75, frame_id=11, step=6,
        obs_xy=[0.1, 0])                    # 位姿太近 -> 非独立
    assert outcome == "duplicate"
    assert anchor.n_obs == 1


def test_ledger_force_belief_never_promotes():
    ledger = ObservationLedger(min_obs=2, min_pose_sep=0.5)
    for i in range(3):
        outcome, anchor = ledger.add_observation(
            "sofa", [1, 2, 0], 0.6, merge_dist=0.75, frame_id=i, step=i,
            obs_xy=[i * 2, 0], force_belief=True)   # 小/远目标
        assert outcome == "belief"
    assert anchor.n_obs == 3


def test_ledger_discard_and_count():
    ledger = ObservationLedger(min_obs=2)
    ledger.add_observation("sofa", [1, 2, 0], 0.8, merge_dist=0.75)
    ledger.add_observation("chair", [5, 5, 0], 0.4, merge_dist=0.75)
    ledger.discard_near("sofa", [1, 2, 0], 0.75)
    assert ledger.belief_anchors("sofa") == []
    assert ledger.count_unresolved(min_score=0.5) == 0
    assert ledger.count_unresolved() == 1


# ----------------------------------------------------------------------
# NavAgent 测试夹具
# ----------------------------------------------------------------------
def _make_agent(backend="semantic_memory"):
    agent = NavAgent()
    agent.semantic_backend = backend
    agent.target_text = "gray fabric sofa"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0,
                                       actions=[])
    return agent


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


def _big_centered_bbox(w=64, h=48):
    return [w * 0.3, h * 0.2, w * 0.7, h * 0.8]   # 占比大且居中


# ----------------------------------------------------------------------
# 到达例程三分支
# ----------------------------------------------------------------------
def test_arrival_confirm_branch():
    agent = _make_agent()
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    assert agent._arrival_decision(_obs()) == "report_found"


def test_arrival_scan_branch_low_score():
    agent = _make_agent()
    agent.client = _MockClient({"found": True, "score": 0.3})  # < 0.5
    assert agent._arrival_decision(_obs()) == "scan"


def test_arrival_scan_branch_on_error():
    agent = _make_agent()
    agent.client = _MockClient(RuntimeError("server down"))
    assert agent._arrival_decision(_obs()) == "scan"


def test_arrival_reject_branch_vlm():
    agent = _make_agent()
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    agent.vlm = SimpleNamespace(encode_rgb=lambda rgb: b"current-jpeg")
    agent.decision_loop = SimpleNamespace(
        decide=lambda *a, **k: DecisionResult(
            "REJECT", confidence=0.9, reason="wrong color"))
    assert agent._arrival_decision(_obs()) == "reject"


def test_arrival_min_conf_differs_by_backend():
    agent = _make_agent(backend="clip_sam")
    agent.client = _MockClient({"found": True, "score": 0.3})  # >= 0.25
    assert agent._arrival_decision(_obs()) == "report_found"
    agent = _make_agent(backend="semantic_memory")
    agent.client = _MockClient({"found": True, "score": 0.3})  # < 0.5
    assert agent._arrival_decision(_obs()) == "scan"


# ----------------------------------------------------------------------
# 末端视觉伺服
# ----------------------------------------------------------------------
def test_servo_confirm_when_close_and_centered():
    agent = _make_agent()
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": _big_centered_bbox()})
    action = agent._confirm_and_report(_obs())
    assert action == int(Action.TARGET_FOUND)
    assert agent.mode == "reported"
    assert not agent._servo_active
    # 实例记忆写入并标记 visited
    assert agent.memory.count_visited("gray fabric sofa") == 1


def test_servo_steers_then_times_out_to_coordinate():
    agent = _make_agent()
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.servo_max_steps = 2
    # 目标偏左且不够近 -> 先转向，超限后退回坐标判定（仍 TARGET_FOUND）
    agent.client = _MockClient({"found": True, "score": 0.9,
                                "bbox": [2, 20, 8, 28]})
    action = agent._confirm_and_report(_obs())
    assert action == int(Action.TURN_LEFT)
    assert agent._servo_active
    action = agent._servo_step(_obs(step=101))
    assert action == int(Action.TURN_LEFT)
    action = agent._servo_step(_obs(step=102))     # 第 3 步超限
    assert action == int(Action.TARGET_FOUND)
    assert not agent._servo_active


def test_servo_target_lost_counts_toward_cap():
    agent = _make_agent()
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.servo_max_steps = 1
    agent.client = _MockClient({"found": False})
    action = agent._confirm_and_report(_obs())
    assert action == int(Action.TURN_LEFT)          # 找回目标
    action = agent._servo_step(_obs(step=101))
    assert action == int(Action.TARGET_FOUND)       # 超限回坐标判定


def test_clip_backend_skips_servo():
    agent = _make_agent(backend="clip_sam")
    agent.target_point = np.array([1.0, 2.0, 0.0])
    agent.client = _MockClient({"found": True, "score": 0.9})
    action = agent._confirm_and_report(_obs())
    assert action == int(Action.TARGET_FOUND)
    assert not agent._servo_active


# ----------------------------------------------------------------------
# semantic 命中准入（分级置信度 -> 实例记忆）
# ----------------------------------------------------------------------
def _hit(point, conf=0.8, frame_id=1, obs_xy=(0, 0), bbox=None,
         depth_std=None):
    pose = np.eye(4)
    pose[:3, 3] = [obs_xy[0], obs_xy[1], 0.0]
    return {"found": True, "point": list(point), "point_score": conf,
            "sam_score": conf, "frame_id": frame_id,
            "pose": pose.tolist(), "candidate_id": None,
            "bbox": bbox, "depth_std": depth_std}


def test_ingest_single_hit_stays_belief():
    agent = _make_agent()
    agent.client = _MockClient()
    agent._ingest_semantic_hits(_obs(), [_hit([3, 4, 0], obs_xy=(0, 0))])
    assert agent.memory.unvisited("gray fabric sofa") == []
    anchors = agent.ledger.belief_anchors("gray fabric sofa")
    assert len(anchors) == 1


def test_ingest_two_independent_hits_confirm():
    agent = _make_agent()
    agent.client = _MockClient()
    hits = [_hit([3, 4, 0], obs_xy=(0, 0), frame_id=1),
            _hit([3.1, 4, 0], obs_xy=(3, 0), frame_id=2)]
    agent._ingest_semantic_hits(_obs(), hits)
    nodes = agent.memory.unvisited("gray fabric sofa")
    assert len(nodes) == 1
    assert nodes[0].n_obs == 2
    assert agent.ledger.belief_anchors("gray fabric sofa") == []


def test_ingest_low_conf_and_blacklist_filtered():
    agent = _make_agent()
    agent.client = _MockClient()
    agent._ingest_semantic_hits(_obs(), [_hit([3, 4, 0], conf=0.2)])
    assert agent.ledger.belief_anchors("gray fabric sofa") == []
    # 已拉黑位置不再入账
    agent.memory.add_or_merge("gray fabric sofa", [3, 4, 0], 0.0,
                              merge_dist=0.75, status="rejected")
    agent._ingest_semantic_hits(_obs(), [_hit([3, 4, 0], conf=0.9)])
    assert agent.ledger.belief_anchors("gray fabric sofa") == []


def test_ingest_small_target_forced_belief():
    agent = _make_agent()
    agent.client = _MockClient()
    hits = [_hit([3, 4, 0], obs_xy=(0, 0), frame_id=1, bbox=[0, 0, 8, 8]),
            _hit([3, 4, 0], obs_xy=(3, 0), frame_id=2, bbox=[0, 0, 8, 8])]
    agent._ingest_semantic_hits(_obs(), hits)       # <32px -> 强制 belief
    assert agent.memory.unvisited("gray fabric sofa") == []
    assert len(agent.ledger.belief_anchors("gray fabric sofa")) == 1


def test_ingest_noisy_depth_forced_belief():
    agent = _make_agent()
    agent.client = _MockClient()
    hits = [_hit([3, 4, 0], obs_xy=(0, 0), frame_id=1, depth_std=0.8),
            _hit([3, 4, 0], obs_xy=(3, 0), frame_id=2, depth_std=0.8)]
    agent._ingest_semantic_hits(_obs(), hits)       # 深度方差过大
    assert agent.memory.unvisited("gray fabric sofa") == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
