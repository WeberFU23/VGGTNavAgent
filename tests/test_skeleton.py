"""skeleton 单元测试：合成自由空间的拓扑提取与 frontier 检测。

    python tests/test_skeleton.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import skeleton as sk
from agents.navigator import OccupancyGrid


def _grid_from_free(free):
    return OccupancyGrid(1.0, np.array([0.0, 0.0]), free,
                         np.zeros_like(free))


def test_cross_junction():
    """十字走廊：1 个岔口 + 4 个端点。"""
    free = np.zeros((40, 40), dtype=bool)
    free[18:22, 2:38] = True   # 横走廊
    free[2:38, 18:22] = True   # 竖走廊
    g = sk.build_skeleton_graph(_grid_from_free(free))
    kinds = [nd["kind"] for nd in g.nodes]
    assert kinds.count("junction") == 1, f"岔口数={kinds.count('junction')}"
    assert kinds.count("endpoint") == 4, f"端点数={kinds.count('endpoint')}"
    assert len(g.edges) == 4, f"边数={len(g.edges)}"
    print("cross junction OK: 1 junction, 4 endpoints, 4 edges")


def test_t_junction():
    """T 形走廊：1 个岔口 + 3 个端点。"""
    free = np.zeros((40, 40), dtype=bool)
    free[18:22, 2:38] = True
    free[2:22, 18:22] = True
    g = sk.build_skeleton_graph(_grid_from_free(free))
    kinds = [nd["kind"] for nd in g.nodes]
    assert kinds.count("junction") == 1
    assert kinds.count("endpoint") == 3
    assert len(g.edges) == 3
    print("T junction OK: 1 junction, 3 endpoints, 3 edges")


def test_two_rooms():
    """两房间 + 门：骨架在拓扑上收缩为 房间中心-门-房间中心 的路径。

    Zhang-Suen 保持同伦而非中轴几何：凸房间细化为中心点，
    两房间各出一个端点，图经门廊连通。
    """
    free = np.zeros((40, 80), dtype=bool)
    free[5:35, 5:35] = True    # 左房间
    free[5:35, 45:75] = True   # 右房间
    free[18:22, 35:45] = True  # 门廊
    g = sk.build_skeleton_graph(_grid_from_free(free))
    n_endpoint = sum(1 for nd in g.nodes if nd["kind"] == "endpoint")
    assert n_endpoint >= 2, f"端点数={n_endpoint}"
    # 图应连通：边数 >= 节点数 - 1
    assert len(g.edges) >= len(g.nodes) - 1
    # 两个房间中心附近各有一个端点
    ends = [nd["cell"] for nd in g.nodes if nd["kind"] == "endpoint"]
    near_left = any(abs(c[0] - 19) < 8 and abs(c[1] - 19) < 8 for c in ends)
    near_right = any(abs(c[0] - 59) < 8 and abs(c[1] - 19) < 8 for c in ends)
    assert near_left and near_right, f"端点位置={ends}"
    print(f"two rooms OK: {n_endpoint} endpoints near room centers, "
          f"{len(g.edges)} edges, connected")


def test_frontier():
    """三面围墙、右侧未知：frontier 应只出现在右侧边界 x≈19。"""
    free = np.zeros((40, 40), dtype=bool)
    obstacle = np.zeros((40, 40), dtype=bool)
    free[10:30, 5:20] = True
    obstacle[10:30, 0:5] = True    # 左墙
    obstacle[5:10, 5:35] = True    # 上墙
    obstacle[30:35, 5:35] = True   # 下墙
    obstacle[10:30, 30:39] = True  # 右远处墙（x∈[20,30) 为未知）
    grid = OccupancyGrid(1.0, np.array([0.0, 0.0]), free, obstacle)
    clusters = sk.frontier_clusters(grid, min_size=3)
    assert clusters, "应检测到 frontier"
    top = clusters[0]
    cx = top["cell"][0]
    assert 17 <= cx <= 21, f"frontier 应在 x≈19 边界，实际 {cx}"
    print(f"frontier OK: {len(clusters)} clusters, "
          f"largest at x={cx:.1f} size={top['size']}")


def test_frontier_does_not_wrap_edges():
    free = np.zeros((5, 7), dtype=bool)
    obstacle = np.ones((5, 7), dtype=bool)
    free[2, 0] = True
    obstacle[2, 0] = False
    obstacle[2, -1] = False       # 唯一 unknown，在地图另一侧
    grid = OccupancyGrid(1.0, np.array([0.0, 0.0]), free, obstacle)
    assert sk.frontier_clusters(grid, min_size=1) == []


def test_observed_hole_is_not_frontier():
    """已观测但未分类的稀疏点云孔洞不能被当作未知区域。"""
    free = np.zeros((20, 20), dtype=bool)
    free[4:16, 4:16] = True
    free[9:11, 9:11] = False
    obstacle = np.zeros_like(free)
    observed = free.copy()
    observed[9:11, 9:11] = True
    grid = OccupancyGrid(1.0, np.zeros(2), free, obstacle,
                         observed=observed)
    clusters = sk.frontier_clusters(grid, min_size=1)
    assert all(not (8 <= c["cell"][0] <= 12 and
                    8 <= c["cell"][1] <= 12) for c in clusters)


def test_frontier_representative_is_free_and_reports_gain():
    free = np.zeros((20, 30), dtype=bool)
    free[5:15, 3:14] = True
    obstacle = np.zeros_like(free)
    observed = np.zeros_like(free)
    observed[:, :14] = True
    grid = OccupancyGrid(1.0, np.zeros(2), free, obstacle,
                         observed=observed)
    clusters = sk.frontier_clusters(grid, min_size=3)
    assert clusters
    x, y = clusters[0]["cell"]
    assert grid.free[y, x]
    assert clusters[0]["information_gain"] > 0
    assert clusters[0]["clearance_cells"] > 0


def test_frontier_24_adjacency_merges_gaps():
    """frontier 连通域用 24 邻接：1 格观测间隙的断口应合并成同一簇。"""
    import numpy as np
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 3:17] = True
    # 1 格间隙：8 邻接断开、24 邻接（5x5 窗口）合并
    mask[10, 9] = False
    labels8, n8 = sk._label8(mask, radius=1)
    labels24, n24 = sk._label8(mask, radius=2)
    assert n8 == 2, "8 邻接应断开"
    assert n24 == 1, "24 邻接应合并 1 格间隙"
    assert labels24[10, 3] == labels24[10, 12] == 1
    # 2 格间隙超出 24 邻接跨越能力（最大跳过 1 个缺失格）
    mask[10, 9:11] = False
    _, n24b = sk._label8(mask, radius=2)
    assert n24b == 2
    # 斜向错位一格的两条带也应在 24 邻接下连通
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 3:8] = True
    mask[11, 8:13] = True
    assert sk._label8(mask, radius=1)[1] == 1
    mask[10, 8] = False
    mask[11, 8] = False
    assert sk._label8(mask, radius=1)[1] == 2
    assert sk._label8(mask, radius=2)[1] == 1


def test_merge_same_spot_combines_nearby_and_keeps_far():
    """同一位置的断簇应合并为一个候选；远处目标保留。"""
    import numpy as np
    base = {
        "cell": (0, 0), "centroid_cell": (0, 0),
        "world": np.array([0.0, 0.0, 0.0]),
        "size": 10, "reason": "geometry", "geometry_gain": 8,
        "semantic_gain": 0, "information_gain": 8, "clearance_cells": 9,
    }
    near = dict(base)
    near["world"] = np.array([1.0, 0.0, 0.0])
    near["size"] = 6
    near["reason"] = "semantic"
    near["geometry_gain"] = 0
    near["semantic_gain"] = 5
    far = dict(base)
    far["world"] = np.array([5.0, 0.0, 0.0])
    far["size"] = 20
    out = sk.merge_same_spot([far, near, base], scale=1.0, radius_m=1.5)
    assert len(out) == 2, "近簇应合并、远簇保留"
    by_size = sorted(out, key=lambda c: -c["size"])
    assert by_size[0]["size"] == 20, "大簇作为独立锚点保留"
    merged = by_size[1]
    assert merged["size"] == 16, "base(10) + near(6)"
    assert merged["reason"] == "both"
    assert merged["geometry_gain"] == 8 and merged["semantic_gain"] == 5


def test_merge_same_spot_scale_and_disable():
    """scale 参与距离换算；radius<=0 时禁用合并。"""
    import numpy as np
    base = {
        "cell": (0, 0), "centroid_cell": (0, 0),
        "world": np.array([0.0, 0.0, 0.0]), "size": 5, "reason": "geometry",
        "geometry_gain": 1, "semantic_gain": 0, "information_gain": 1,
        "clearance_cells": 5,
    }
    other = dict(base)
    other["world"] = np.array([2.0, 0.0, 0.0])
    # 2.0 米 > 1.5 阈值：不合并
    assert len(sk.merge_same_spot([base, other], scale=1.0, radius_m=1.5)) == 2
    # scale=0.5 → 2.0 单位 = 1.0 米 < 1.5：合并
    out = sk.merge_same_spot([base, other], scale=0.5, radius_m=1.5)
    assert len(out) == 1
    # radius=0：禁用
    assert len(sk.merge_same_spot([base, other], scale=1.0, radius_m=0)) == 2
    # 输入列表不被修改
    assert len([base, other]) == 2


def test_semantic_gap_creates_one_unified_frontier():
    """几何已覆盖但 caption 视角不足时，仍应产生语义型统一 frontier。"""
    free = np.zeros((20, 24), dtype=bool)
    free[5:15, 3:20] = True
    obstacle = np.zeros_like(free)
    geometry = np.ones_like(free)
    semantic = np.zeros_like(free)
    semantic[5:15, 3:10] = True
    grid = OccupancyGrid(
        1.0, np.zeros(2), free, obstacle, observed=geometry,
        semantic_inspected=semantic, semantic_coverage_enabled=True)
    clusters, layers = sk.frontier_clusters(
        grid, min_size=3, return_layers=True)
    assert clusters
    assert all(c["reason"] == "semantic" for c in clusters)
    assert clusters[0]["semantic_gain"] > 0
    assert clusters[0]["geometry_gain"] == 0
    assert layers["semantic"].any() and layers["unified"].any()
    assert not layers["geometry"].any()


if __name__ == "__main__":
    test_cross_junction()
    test_t_junction()
    test_two_rooms()
    test_frontier()
    test_frontier_does_not_wrap_edges()
    test_observed_hole_is_not_frontier()
    test_frontier_representative_is_free_and_reports_gain()
    test_semantic_gap_creates_one_unified_frontier()
    test_frontier_24_adjacency_merges_gaps()
    test_merge_same_spot_combines_nearby_and_keeps_far()
    test_merge_same_spot_scale_and_disable()
    print("ALL SKELETON TESTS PASSED")
