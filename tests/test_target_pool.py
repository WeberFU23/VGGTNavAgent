"""get_target_pool() 评测采集接口的纯数学单测。

用满足锚点约束的真值 Sim(3)（proper rotation + scale + translation）
生成合成 SLAM 点，验证变换回 habitat 世界系的结果；不需要 habitat、
mapping server 或 VLM。
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent


def _make_agent(scale=0.42):
    agent = NavAgent()
    agent.calibrator = SimpleNamespace(current_scale=lambda: scale)
    agent._metric_snapshot.update(scale=scale, source="test", revision=1,
                                  pending=None, pending_count=0, far_count=0)
    return agent


def _world_forward(compass):
    """benchmark compass 约定：绕世界 +Y 右手 yaw，0 时面向 -Z。"""
    return np.array([-math.sin(compass), 0.0, -math.cos(compass)])


def _ground_truth_sim3(g0, c0, s, slam_anchor, yaw_s0):
    """从锚点约束独立构造真值相似变换 p_w = s·Q·p_s + t。

    约束：SLAM 锚点位置 a0 -> g0；SLAM 锚点朝向 yaw_s0 -> 世界
    forward(c0)；SLAM up (z_s) -> 世界 up (y_w)；Q 为 proper rotation。
    两个正交方向像唯一确定 proper rotation，用于独立检验实现。
    """
    f_s = np.array([math.cos(yaw_s0), math.sin(yaw_s0), 0.0])
    u_s = np.array([0.0, 0.0, 1.0])
    f_w = _world_forward(c0)
    u_w = np.array([0.0, 1.0, 0.0])
    w_s = np.cross(f_s, u_s)
    w_w = np.cross(f_w, u_w)
    basis_s = np.stack([f_s, u_s, w_s], axis=1)
    basis_w = np.stack([f_w, u_w, w_w], axis=1)
    Q = basis_w @ basis_s.T
    assert np.linalg.det(Q) > 0.99  # proper rotation
    t = g0 - s * (Q @ np.asarray(slam_anchor, dtype=np.float64))
    return Q, t


def _world_to_slam(p_w, Q, t, s):
    return (Q.T @ (np.asarray(p_w, dtype=np.float64) - t)) / s


def _agent_with_anchor(c0=0.7, yaw_s0=-1.1, scale=0.42):
    agent = _make_agent(scale=scale)
    g0 = np.array([3.0, 1.5, -2.0])
    slam_anchor = np.array([0.1, -0.2, 0.05])
    agent._pool_world_anchor = (g0, c0)
    agent._pool_slam_anchor = (float(slam_anchor[0]), float(slam_anchor[1]),
                               float(slam_anchor[2]), yaw_s0)
    return agent, g0, slam_anchor


def test_not_ready_returns_empty():
    agent = _make_agent()
    agent.memory.add(point=[1.0, 2.0, 0.5], text="chair")
    assert agent.get_target_pool() == []  # 无锚点
    agent._pool_world_anchor = (np.zeros(3), 0.0)
    assert agent.get_target_pool() == []  # 无 SLAM 锚点
    agent._pool_slam_anchor = (0.0, 0.0, 0.0, 0.0)
    agent.calibrator = SimpleNamespace(current_scale=lambda: None)
    agent._metric_snapshot["scale"] = None  # 尺度未播种
    assert agent.get_target_pool() == []  # 无尺度估计


def test_roundtrip_through_ground_truth_sim3():
    c0, yaw_s0, s = 0.7, -1.1, 0.42
    agent, g0, slam_anchor = _agent_with_anchor(c0, yaw_s0, s)
    Q, t = _ground_truth_sim3(g0, c0, s, slam_anchor, yaw_s0)

    world_points = [np.array([4.2, 1.1, -3.5]),
                    np.array([-1.0, 2.2, 0.3]),
                    np.array([3.0, 1.5, -2.0])]  # 第三个即锚点自身
    labels = ["red chair", "basket near shelf", "anchor object"]
    for p_w, label in zip(world_points, labels):
        agent.memory.add(point=_world_to_slam(p_w, Q, t, s), text=label)

    pool = agent.get_target_pool()
    assert len(pool) == 3
    for entry, p_w, label in zip(pool, world_points, labels):
        assert set(entry) == {"position", "reported", "label"}
        assert np.allclose(entry["position"], p_w, atol=1e-9)
        assert entry["reported"] is False
        assert entry["label"] == label


def test_heading_maps_to_world_forward():
    """锚点处的 SLAM forward 方向经变换后必须等于世界 forward。"""
    c0, yaw_s0, s = -2.3, 0.4, 1.7
    agent, g0, slam_anchor = _agent_with_anchor(c0, yaw_s0, s)
    forward_s = slam_anchor + np.array([math.cos(yaw_s0),
                                        math.sin(yaw_s0), 0.0])
    agent.memory.add(point=slam_anchor, text="a")
    agent.memory.add(point=forward_s, text="b")
    pool = agent.get_target_pool()
    direction = np.asarray(pool[1]["position"]) - np.asarray(
        pool[0]["position"])
    direction = direction / np.linalg.norm(direction)
    assert np.allclose(direction, _world_forward(c0), atol=1e-9)


def test_reported_flag_and_label_truncation():
    agent, g0, slam_anchor = _agent_with_anchor()
    node = agent.memory.add(point=[0.5, 0.5, 0.1], text="x" * 500)
    agent.memory.claim(node, step=3)
    agent.memory.add(point=[1.0, 1.0, 0.2], text="short")
    pool = agent.get_target_pool()
    assert [entry["reported"] for entry in pool] == [True, False]
    assert len(pool[0]["label"]) == 100
    assert pool[1]["label"] == "short"


def test_capture_world_anchor_first_valid_wins():
    agent = _make_agent()
    bad = SimpleNamespace(gps=None, compass=0.0)
    agent._capture_pool_world_anchor(bad)
    assert agent._pool_world_anchor is None
    bad = SimpleNamespace(gps=[1.0, float("nan"), 0.0], compass=0.0)
    agent._capture_pool_world_anchor(bad)
    assert agent._pool_world_anchor is None
    obs = SimpleNamespace(gps=[1.0, 2.0, 3.0], compass=0.5)
    agent._capture_pool_world_anchor(obs)
    assert agent._pool_world_anchor is not None
    assert np.allclose(agent._pool_world_anchor[0], [1.0, 2.0, 3.0])
    assert agent._pool_world_anchor[1] == 0.5
    later = SimpleNamespace(gps=[9.0, 9.0, 9.0], compass=1.0)
    agent._capture_pool_world_anchor(later)
    assert np.allclose(agent._pool_world_anchor[0], [1.0, 2.0, 3.0])


def test_get_target_pool_never_raises():
    agent = _make_agent()
    agent._pool_world_anchor = ("garbage", None)
    agent._pool_slam_anchor = (0.0, 0.0)  # 长度不对
    assert agent.get_target_pool() == []
