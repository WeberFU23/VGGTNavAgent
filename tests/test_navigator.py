"""navigator 单元测试：合成场景（房间 + 带门洞的墙），可直接 python 运行。

    python tests/test_navigator.py
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import navigator as nav
from agents.memory import InstanceMemory
from agents.nav_agent import NavAgent
from benchmark_api import Action
from decision import StrategicDecision, TargetSpec, VLMDecisionClient


def _rot(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def make_scene(seed=0):
    """10x6m 房间，x=5 处有墙、y∈[2,3] 为门洞。整体随机旋转。

    返回 (points_aligned_gt, cam_centers_gt, poses, Q)
    points/cam_centers 在"真值对齐坐标系"（z 向上）下给出用于断言；
    poses 与喂给栅格的点为 Q 旋转后的版本，模拟 SLAM 的任意基。
    """
    rng = np.random.default_rng(seed)
    pts = []
    # 地板 0.2m 网格 + 噪声
    xs = np.arange(0, 10.001, 0.2)
    ys = np.arange(0, 6.001, 0.2)
    gx, gy = np.meshgrid(xs, ys)
    floor = np.stack([gx.ravel(), gy.ravel(),
                      rng.normal(0, 0.01, gx.size)], axis=1)
    pts.append(floor)
    # 墙（门洞 y∈[2,3] 留空）
    for y0, y1 in ((0.0, 2.0), (3.0, 6.0)):
        wy = np.arange(y0, y1 + 1e-6, 0.1)
        wz = np.arange(0.0, 2.0 + 1e-6, 0.1)
        my, mz = np.meshgrid(wy, wz)
        wall = np.stack([np.full(my.size, 5.0), my.ravel(), mz.ravel()],
                        axis=1)
        pts.append(wall)
    points = np.concatenate(pts)

    # 相机轨迹：z=1.5 高，x-y 平面内走弧线、恰从门洞（y∈[2,3]）穿过
    # 墙体（直线轨迹会让协方差退化；穿墙轨迹不物理，会误导走廊逻辑）
    poses = []
    for x in np.linspace(1, 9, 17):
        y = 1.0 + 1.5 * math.sin((x - 1.0) / 8.0 * math.pi)
        x_c = np.array([0, -1, 0.0])
        y_c = np.array([0, 0, -1.0])
        z_c = np.array([1, 0, 0.0])
        R = np.stack([x_c, y_c, z_c], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, 1.5]
        poses.append(T)
    poses = np.stack(poses)

    Q = _rot([1, 2, 0.3], 25)
    poses_q = poses.copy()
    poses_q[:, :3, :3] = Q @ poses[:, :3, :3]
    poses_q[:, :3, 3] = (Q @ poses[:, :3, 3].T).T
    points_q = (Q @ points.T).T
    return points, poses[:, :3, 3].copy(), poses_q, points_q


def test_gravity_alignment():
    _, _, poses_q, _ = make_scene()
    R = nav.gravity_alignment(poses_q)
    up = R[2]  # 第三行 = 对齐系的 z' 轴在原系中的方向
    # z' 应与相机"上"方向一致：up·(cam_up_mean) > 0
    cam_up = -(poses_q[:, :3, 1].mean(0))
    assert up @ cam_up > 0
    # 对齐后相机高度应近似恒定且接近真值 1.5
    h = (poses_q[:, :3, 3] @ R.T)[:, 2]
    assert h.std() < 0.05, f"相机高度抖动 {h.std()}"
    assert abs(h.mean() - 1.5) < 0.05, f"相机高度 {h.mean()} 偏离真值"
    print("gravity_alignment OK, 相机高度 mean=%.3f std=%.4f" % (h.mean(), h.std()))


def test_gravity_alignment_straight_trajectory():
    """位置协方差退化时，旋转提供的 up 仍应稳定。"""
    _, _, poses, _ = make_scene()
    up = -poses[0, :3, 1]
    direction = poses[0, :3, 2]
    direction -= np.dot(direction, up) * up
    direction /= np.linalg.norm(direction)
    poses[:, :3, 3] = up * 1.5 + \
        np.linspace(0.0, 8.0, len(poses))[:, None] * direction
    R = nav.gravity_alignment(poses)
    heights = poses[:, :3, 3] @ R.T
    assert heights[:, 2].std() < 1e-6


def test_dead_reckon_undo():
    fol = nav.PathFollower(scale=2.0)
    before = (fol.x, fol.y, fol.yaw)
    fol.dead_reckon(nav.MOVE_FORWARD)
    fol.undo_dead_reckon(nav.MOVE_FORWARD)
    assert np.allclose((fol.x, fol.y, fol.yaw), before)


def test_multi_target_finish_policy():
    agent = object.__new__(NavAgent)
    agent._target_mode = "many"
    agent._target_count = 2
    agent._reported_count = 2
    agent._no_hit_queries = 0
    agent.finish_patience = 5
    agent.vlm = VLMDecisionClient(enabled=False)
    obs = SimpleNamespace(step_count=10, max_steps=100)
    assert agent._should_finish(obs)

    agent._target_mode = "all"
    agent._target_count = None
    agent._reported_count = 1
    agent._no_hit_queries = 5
    agent.target_text = "sink"
    agent.memory = InstanceMemory()
    agent.explore_replan_interval = 25
    agent._last_frontier_step = 80
    agent._last_frontier_count = 0
    agent._frontier_empty_streak = 3
    agent.finish_frontier_patience = 3
    agent._last_map_growth_step = 0
    agent.finish_map_stable_steps = 50
    obs = SimpleNamespace(step_count=95, max_steps=100)
    assert agent._should_finish(obs)
    agent._last_frontier_count = 1
    assert not agent._should_finish(obs)


def test_instruction_only_target_phrase():
    agent = object.__new__(NavAgent)
    agent.vlm = VLMDecisionClient(enabled=False)
    agent.target_spec = None
    agent._target_spec_source = None
    agent._target_mode = "any"
    agent._target_count = None
    obs = SimpleNamespace(goal_text="Find any bag.")
    assert agent._target_phrase(obs) == "bag"
    agent.target_spec = None
    agent._target_mode = "many"
    agent._target_count = 2
    obs.goal_text = "Find exactly two baskets."
    assert agent._target_phrase(obs) == "baskets"
    agent.target_spec = None
    agent._target_mode = "all"
    agent._target_count = None
    obs.goal_text = "Go to all sink objects."
    assert agent._target_phrase(obs) == "sink objects"
    agent.target_spec = None
    obs.goal_text = "Navigate to the red fabric chair with wooden legs."
    assert agent._target_phrase(obs) == "red fabric chair with wooden legs"

    class _ParsingVLM:
        enabled = True

        @staticmethod
        def parse_instruction(instruction, target_mode, target_count):
            assert instruction == "Find the television near the sofa."
            return TargetSpec("television", "television near the sofa", 0.9)

    agent.vlm = _ParsingVLM()
    agent.target_spec = None
    obs.goal_text = "Find the television near the sofa."
    assert agent._target_phrase(obs) == "television"
    assert agent.target_spec.target_description == "television near the sofa"


def test_vlm_candidate_integration():
    class _Client:
        @staticmethod
        def get_candidate_evidence(candidate_id):
            return {"found": True}, ("jpeg-" + candidate_id).encode()

        @staticmethod
        def get_state():
            return {"num_submaps": 2, "queued_keyframes": 0, "busy": False}

    class _Calibrator:
        @staticmethod
        def current_scale():
            return 1.0

    class _VLM:
        enabled = True

        @staticmethod
        def choose_candidate(*args, **kwargs):
            return StrategicDecision(
                decision="navigate", candidate_id="c2",
                rejected_candidate_ids=["c1"], confidence=0.9,
                reason="c2 matches")

    agent = object.__new__(NavAgent)
    agent.vlm = _VLM()
    agent.client = _Client()
    agent.calibrator = _Calibrator()
    agent.target_spec = TargetSpec("tv", "flat television", 0.9)
    agent._target_spec_source = "Find the TV."
    agent._target_mode = "any"
    agent._target_count = None
    agent._reported_count = 0
    agent._no_hit_queries = 0
    agent.memory = InstanceMemory()
    agent._explore_hint = "none"
    agent._explore_hint_steps = 0
    agent.vlm_candidate_conf = 0.35
    obs = SimpleNamespace(
        goal_text="Find the TV.", rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        step_count=40, max_steps=100)
    candidates = [
        {"candidate_id": "c1", "point": [0, 0, 0]},
        {"candidate_id": "c2", "point": [2, 0, 0]},
    ]
    selected, evidence = agent._vlm_candidate_decision(obs, candidates)
    assert selected["candidate_id"] == "c2"
    assert evidence == b"jpeg-c2"
    assert sum(1 for n in agent.memory.nodes
               if n.status == "rejected") == 1


def test_runtime_memory_route_uses_persistent_instances():
    agent = object.__new__(NavAgent)
    agent.memory = InstanceMemory()
    agent.target_text = "bag"
    agent._target_mode = "many"
    agent._target_count = 2
    agent._reported_count = 0
    agent._current_aligned_xy = lambda: (0.0, 0.0)
    for x in (10.0, 2.0, 5.0):
        agent.memory.add_or_merge(
            "bag", [x, 0, 0], 0.9, 0.5,
            status="confirmed", candidate_id=f"c{x}")
    ordered = agent._ordered_memory_nodes()
    assert [float(nd.point[0]) for nd in ordered] == [2.0, 5.0]

def test_grid_and_astar():
    _, _, poses_q, points_q = make_scene()
    R = nav.gravity_alignment(poses_q)
    pts_a = points_q @ R.T
    cams_a = poses_q[:, :3, 3] @ R.T
    grid = nav.OccupancyGrid.build(pts_a, cams_a)
    assert grid is not None

    # 对齐系与真值系的关系由构造决定（fwd=+x, up=+z）：
    # p_a = (p·Qe_y, -p·Qe_x, p·Qe_z)，即世界 (x,y,z) -> 对齐 (y,-x,z)。
    # 因此墙（世界 x=5）在对齐系是 y=-5 的横墙，门洞（世界 y∈[2,3]）
    # 在对齐系是 x∈[2,3]。路径必须穿过门洞。
    start_xy = cams_a[0, :2]
    goal_xy = cams_a[-1, :2]
    path = grid.astar(start_xy, goal_xy)
    assert path is not None, "A* 未找到路径"
    straight = np.linalg.norm(goal_xy - start_xy)
    plen = sum(np.linalg.norm(path[i + 1] - path[i])
               for i in range(len(path) - 1))
    assert plen >= straight * 0.99
    print(f"astar OK: {len(path)} 点, 长度 {plen:.2f} (直线 {straight:.2f})")

    # 穿墙点（|y+5|<0.3 的路径点）必须落在门洞附近
    crossings = [p for p in path if abs(p[1] + 5.0) < 0.3]
    assert crossings, "路径没有跨过墙线？"
    for p in crossings:
        assert 1.4 < p[0] < 3.6, f"路径穿墙点 {p} 不在门洞内"
    print(f"门洞通过 OK: {len(crossings)} 个穿墙点均在门洞附近")

    sp = grid.shortcut(path)
    assert len(sp) <= len(path)
    print(f"shortcut OK: {len(sp)} 点")

    # 穿墙检查：真值系下墙在 x=5、门洞 y∈[2,3]。
    # 对齐系与真值系差未知旋转，无法直接断言坐标，但可验证
    # "直线穿墙必失败"：栅格内取墙两侧格子的视线应为 False。
    # 找墙格：障碍格
    obs_cells = np.argwhere(grid.obstacle)
    assert len(obs_cells) > 0, "墙没有被标记为障碍"
    print(f"障碍格数量 {len(obs_cells)} (膨胀后)")
    return grid, path, R, cams_a


def test_path_follower():
    _, _, poses_q, points_q = make_scene()
    R = nav.gravity_alignment(poses_q)
    pts_a = points_q @ R.T
    cams_a = poses_q[:, :3, 3] @ R.T
    grid = nav.OccupancyGrid.build(pts_a, cams_a)
    start_xy = cams_a[0, :2]
    goal_xy = cams_a[-1, :2]
    path = grid.shortcut(grid.astar(start_xy, goal_xy))

    fol = nav.PathFollower(scale=1.0, reach_m=0.8, waypoint_m=0.3)
    fol.set_path([list(p) for p in path])
    fol.update_anchor(poses_q[0], R, frame_id=1)

    steps = 0
    arrived = False
    while steps < 2000:
        action, arrived = fol.next_action()
        if arrived:
            break
        assert action is not None
        fol.dead_reckon(action)
        steps += 1
    assert arrived, "跟随器未能在限定步数内到达"
    dist = math.hypot(fol.x - goal_xy[0], fol.y - goal_xy[1])
    assert dist < 0.8 + 1e-6
    print(f"path_follower OK: {steps} 个动作到达, 终点偏差 {dist:.2f}m")


def test_nearest_traversable():
    _, _, poses_q, points_q = make_scene()
    R = nav.gravity_alignment(poses_q)
    pts_a = points_q @ R.T
    cams_a = poses_q[:, :3, 3] @ R.T
    grid = nav.OccupancyGrid.build(pts_a, cams_a)
    obs = np.argwhere(grid.obstacle)
    cell = (int(obs[0][1]), int(obs[0][0]))  # 一个障碍格 (x, y)
    assert not grid.traversable(cell)
    near = grid.nearest_traversable(cell, max_radius=30)
    assert near is not None and grid.traversable(near)
    print("nearest_traversable OK:", cell, "->", near)


def test_freespace_connectivity():
    """点云自由空间：真房间连通、镜面鬼房间被连通域剔除、桌面成障碍。"""
    rng = np.random.default_rng(7)
    pts = []
    # 真房间地板 10x6m
    xs = np.arange(0, 10.001, 0.15)
    ys = np.arange(0, 6.001, 0.15)
    gx, gy = np.meshgrid(xs, ys)
    pts.append(np.stack([gx.ravel(), gy.ravel(),
                         rng.normal(0, 0.01, gx.size)], axis=1))
    # 四面围墙（无门），高 0~2m
    for wall_x, wall_y in ((0.0, None), (10.0, None), (None, 0.0), (None, 6.0)):
        t = np.arange(0.0, (6.0 if wall_y is not None else 10.0) + 1e-6, 0.1)
        z = np.arange(0.0, 2.0 + 1e-6, 0.1)
        mt, mz = np.meshgrid(t, z)
        if wall_x is not None:
            pts.append(np.stack([np.full(mt.size, wall_x), mt.ravel(),
                                 mz.ravel()], axis=1))
        else:
            pts.append(np.stack([mt.ravel(), np.full(mt.size, wall_y),
                                 mz.ravel()], axis=1))
    # 桌面 1x1m，高 0.75m
    tx = np.arange(7.0, 8.001, 0.1)
    ty = np.arange(4.0, 5.001, 0.1)
    tx, ty = np.meshgrid(tx, ty)
    pts.append(np.stack([tx.ravel(), ty.ravel(),
                         np.full(tx.size, 0.75)], axis=1))
    # 镜面鬼房间：墙外 x∈[11,14] 的假地板（镜子反射产生的假深度）
    mx = np.arange(11.0, 14.001, 0.15)
    my = np.arange(0.0, 6.001, 0.15)
    mx, my = np.meshgrid(mx, my)
    pts.append(np.stack([mx.ravel(), my.ravel(),
                         rng.normal(0, 0.01, mx.size)], axis=1))
    points = np.concatenate(pts)

    # 相机轨迹：真房间内 z=1.5 的折线
    cam = np.stack([np.linspace(1, 9, 17),
                    1.0 + 1.5 * np.sin(np.linspace(0, math.pi, 17)),
                    np.full(17, 1.5)], axis=1)

    grid = nav.OccupancyGrid.build(points, cam)
    assert grid is not None

    c_room = grid.world_to_cell([5.0, 3.0, 1.5])
    c_mirror = grid.world_to_cell([12.5, 3.0, 1.5])
    c_wall = grid.world_to_cell([0.0, 3.0, 1.5])
    c_table = grid.world_to_cell([7.5, 4.5, 1.5])

    assert grid.traversable(c_room), "真房间中心应可行走"
    assert not grid.free[c_mirror[1], c_mirror[0]], \
        "镜面鬼房间不连通，应被剔除"
    assert grid.obstacle[c_wall[1], c_wall[0]], "墙应是障碍"
    assert not grid.traversable(c_table), "桌面（0.75m）应在障碍层"
    print("freespace connectivity OK: room free, mirror/wall/table blocked")


def test_frame_points_freespace():
    """逐帧投票自由空间：底部行锚定地板，房间连通、墙成障碍、门洞可穿。"""
    rng = np.random.default_rng(3)
    # 地板 10x6 z=0
    xs = np.arange(0, 10.001, 0.15)
    ys = np.arange(0, 6.001, 0.15)
    gx, gy = np.meshgrid(xs, ys)
    floor = np.stack([gx.ravel(), gy.ravel(),
                      rng.normal(0, 0.005, gx.size)], axis=1)
    floor_rows = np.full(len(floor), 95, dtype=np.int32)
    # 墙 x=5，门洞 y∈[2,3]
    walls = []
    for y0, y1 in ((0.0, 2.0), (3.0, 6.0)):
        wy = np.arange(y0, y1 + 1e-6, 0.05)
        wz = np.arange(0.0, 2.0 + 1e-6, 0.05)
        my, mz = np.meshgrid(wy, wz)
        walls.append(np.stack([np.full(my.size, 5.0), my.ravel(),
                               mz.ravel()], axis=1))
    wall = np.concatenate(walls)
    wall_rows = np.full(len(wall), 50, dtype=np.int32)

    pts = np.concatenate([floor, wall]).astype(np.float32)
    rows = np.concatenate([floor_rows, wall_rows])
    frames = []
    for cx in (1.0, 9.0):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [cx, 3.0, 1.5]
        frames.append({"points": pts, "rows": rows, "pose": pose,
                       "frame_id": len(frames)})

    grid = nav.OccupancyGrid.from_frame_points(frames, np.eye(3))
    assert grid is not None
    c_left = grid.world_to_cell([2.5, 3.0, 1.5])
    c_right = grid.world_to_cell([7.5, 3.0, 1.5])
    c_wall = grid.world_to_cell([5.0, 0.5, 1.5])
    c_door = grid.world_to_cell([5.0, 2.5, 1.5])
    assert grid.traversable(c_left), "左房间应可行走"
    assert grid.traversable(c_right), "右房间应可行走"
    assert grid.obstacle[c_wall[1], c_wall[0]], "墙应是障碍"
    assert grid.traversable(c_door), "门洞应可通行"
    # 两侧连通（门洞连通性）：A* 应能找到从左到右的路径
    path = grid.astar([2.5, 3.0], [7.5, 3.0])
    assert path is not None, "经门洞的路径应存在"
    print("frame_points freespace OK: rooms free, wall blocked, door open")


if __name__ == "__main__":
    test_gravity_alignment()
    test_gravity_alignment_straight_trajectory()
    test_dead_reckon_undo()
    test_multi_target_finish_policy()
    test_instruction_only_target_phrase()
    test_vlm_candidate_integration()
    test_runtime_memory_route_uses_persistent_instances()
    test_grid_and_astar()
    test_path_follower()
    test_nearest_traversable()
    test_freespace_connectivity()
    test_frame_points_freespace()
    print("ALL TESTS PASSED")




def test_expand_prompts_basket():
    """basket 同义词展开 + 复数归一化（"baskets" 也能匹配 "basket" 条目）。"""
    from mapping.semantic import Sam3Grounder
    g = Sam3Grounder()
    out = g.expand_prompts("basket")
    assert out[0] == "basket"
    assert "laundry basket" in out and "storage basket" in out and "hamper" in out
    out2 = g.expand_prompts("baskets")
    assert out2[0] == "baskets"
    assert "laundry baskets" in out2
    # 无同义词的类别保持原样
    assert g.expand_prompts("sink") == ["sink"]
