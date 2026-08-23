"""骨架拓扑提取（SSMG-Nav 式，纯 numpy，client 侧）。

自由空间栅格 -> 形态学细化（Zhang-Suen）-> 一像素宽中轴 ->
按 8 邻域度数分类：端点（度1，死角/房间深处）、岔口（度>=3，决策点）、
连接段（度2，被压缩为边）-> 拓扑图。岔口像素先按 8 连通聚成节点，
连接段按 8 连通聚成边并挂到相邻节点。短毛刺（细化伪影）剪除。

另提供统一 frontier 检测：几何未知边界与语义未检查边界共用一套候选，
但保留来源和两类信息增益供排序、日志与鸟瞰图解释。
"""

from collections import deque

import numpy as np


def zhang_suen_thinning(binary, max_iter=200):
    """Zhang-Suen 形态学细化，输入输出均为 bool 数组。"""
    img = binary.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in range(2):
            P = np.pad(img, 1)
            P2 = P[0:-2, 1:-1]
            P3 = P[0:-2, 2:]
            P4 = P[1:-1, 2:]
            P5 = P[2:, 2:]
            P6 = P[2:, 1:-1]
            P7 = P[2:, 0:-2]
            P8 = P[1:-1, 0:-2]
            P9 = P[0:-2, 0:-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = np.stack([P2, P3, P4, P5, P6, P7, P8, P9, P2])
            A = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(axis=0)
            base = (B >= 2) & (B <= 6) & (A == 1) & (img == 1)
            if step == 0:
                m = base & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                m = base & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def _label8(mask):
    """8 连通标记，返回 (labels int32, n)。"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                            and not labels[ny, nx]:
                        labels[ny, nx] = n
                        q.append((ny, nx))
    return labels, n


def _neighbor_count(skel):
    P = np.pad(skel.astype(np.uint8), 1)
    return sum(P[1 + dy:1 + dy + skel.shape[0],
                 1 + dx:1 + dx + skel.shape[1]]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy, dx) != (0, 0))


class SkeletonGraph:
    """拓扑图：nodes=[{id, cell, world, kind}]，edges=[{n1, n2, cells, length}]。"""

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def node_by_id(self, nid):
        for nd in self.nodes:
            if nd["id"] == nid:
                return nd
        return None


def extract_skeleton(free_mask):
    """自由空间 -> (骨架 bool, 度数 int)。"""
    skel = zhang_suen_thinning(free_mask)
    deg = _neighbor_count(skel)
    return skel, deg


def build_skeleton_graph(grid, min_spur_cells=8):
    """从 OccupancyGrid 的自由空间构建拓扑图。

    min_spur_cells: 长度小于该值的端点毛刺（细化伪影）剪除。
    返回 SkeletonGraph；自由空间为空时返回 None。
    """
    if not grid.free.any():
        return None
    skel, deg = extract_skeleton(grid.free)
    if not skel.any():
        return None

    special = skel & (deg != 2)      # 端点(1)+岔口(>=3)+孤立点(0)
    nlabels, n_nodes = _label8(special)
    nodes = []
    for nid in range(1, n_nodes + 1):
        ys, xs = np.nonzero(nlabels == nid)
        cell = (float(xs.mean()), float(ys.mean()))
        kind = "junction" if deg[ys, xs].max() >= 3 else "endpoint"
        nodes.append({
            "id": nid,
            "cell": cell,
            "world": grid.cell_to_world(cell),
            "kind": kind,
        })

    conn = skel & ~special
    clabels, n_chains = _label8(conn)
    edges = []
    for cid in range(1, n_chains + 1):
        ys, xs = np.nonzero(clabels == cid)
        # 链的 8 邻域内触及的节点
        adj = set()
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(nlabels.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(nlabels.shape[1], x + 2)
            adj.update(int(v) for v in
                       np.unique(nlabels[y0:y1, x0:x1]) if v > 0)
        cells = list(zip(xs.tolist(), ys.tolist()))
        adj = sorted(adj)
        if len(adj) >= 2:
            edges.append({"n1": adj[0], "n2": adj[1],
                          "cells": cells, "length": len(cells)})
        elif len(adj) == 1:
            # 自环链（节点贴着自己的环），保留为自环边
            edges.append({"n1": adj[0], "n2": adj[0],
                          "cells": cells, "length": len(cells)})

    # 剪短毛刺：端点节点 + 唯一短边
    if min_spur_cells > 0:
        degree = {}
        for e in edges:
            degree[e["n1"]] = degree.get(e["n1"], 0) + 1
            degree[e["n2"]] = degree.get(e["n2"], 0) + 1
        drop_nodes = set()
        keep_edges = []
        for e in edges:
            spur = (e["n1"] != e["n2"] and e["length"] < min_spur_cells
                    and (degree.get(e["n1"], 0) == 1
                         or degree.get(e["n2"], 0) == 1))
            if spur:
                if degree.get(e["n1"], 0) == 1:
                    drop_nodes.add(e["n1"])
                if degree.get(e["n2"], 0) == 1:
                    drop_nodes.add(e["n2"])
            else:
                keep_edges.append(e)
        edges = keep_edges
        nodes = [nd for nd in nodes if nd["id"] not in drop_nodes]

    return SkeletonGraph(nodes, edges)


def _neighbor_ring(mask):
    """返回 mask 本身及其 8 邻域；显式 pad，禁止地图边缘首尾相连。"""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    ring = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                y0 = 1 + dy
                x0 = 1 + dx
                ring |= padded[y0:y0 + mask.shape[0],
                               x0:x0 + mask.shape[1]]
    return ring


def frontier_layers(grid):
    """构造同一栅格快照上的几何、语义和统一 frontier 像素层。

    候选始终落在可达自由格上。几何 frontier 位于自由区朝向未重建区的
    边缘；语义 frontier 位于已检查自由区朝向未检查自由区的边缘。这样
    “需要探索”由两层的 OR 决定，又不会把整片未检查区域都画成目标。
    """
    free = np.asarray(grid.free, dtype=bool)
    geometry_observed = np.asarray(getattr(
        grid, "geometry_observed",
        getattr(grid, "observed", free | grid.obstacle)), dtype=bool)
    geometry_unknown = ~geometry_observed
    geometry_frontier = free & _neighbor_ring(geometry_unknown)

    semantic_enabled = bool(getattr(
        grid, "semantic_coverage_enabled", False))
    semantic_inspected = np.asarray(getattr(
        grid, "semantic_inspected", geometry_observed), dtype=bool)
    semantic_gap = free & ~semantic_inspected if semantic_enabled \
        else np.zeros_like(free)
    # 代表点选在边界的已检查侧，便于到达后面向未检查区域继续取景。
    semantic_frontier = (free & semantic_inspected &
                         _neighbor_ring(semantic_gap)) if semantic_enabled \
        else np.zeros_like(free)
    unified = geometry_frontier | semantic_frontier
    return {
        "geometry_unknown": geometry_unknown,
        "semantic_gap": semantic_gap,
        "geometry": geometry_frontier,
        "semantic": semantic_frontier,
        "unified": unified,
    }


def frontier_clusters(grid, min_size=5, info_radius=5,
                      return_layers=False):
    """聚类统一 frontier，返回一套候选及其几何/语义来源。"""
    layers = frontier_layers(grid)
    frontier = layers["unified"]
    if not frontier.any():
        return ([], layers) if return_layers else []
    labels, n = _label8(frontier)
    clusters = []
    for cid in range(1, n + 1):
        ys, xs = np.nonzero(labels == cid)
        if len(ys) < min_size:
            continue
        centroid = (float(xs.mean()), float(ys.mean()))
        # 均值可能落在未知/障碍内。代表点必须取自簇内真实自由格，并
        # 优先选择局部自由空间更宽、其次更接近几何中心的格子。
        best_cell = None
        best_key = None
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 2), min(grid.free.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(grid.free.shape[1], x + 3)
            clearance = int(np.asarray(grid.free[y0:y1, x0:x1]).sum())
            center_d2 = (x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
            key = (clearance, -center_d2)
            if best_key is None or key > best_key:
                best_key, best_cell = key, (int(x), int(y))
        # 分开记录几何与语义增益，统一 utility 可再按任务需要加权。
        y0 = max(0, int(ys.min()) - info_radius)
        y1 = min(frontier.shape[0], int(ys.max()) + info_radius + 1)
        x0 = max(0, int(xs.min()) - info_radius)
        x1 = min(frontier.shape[1], int(xs.max()) + info_radius + 1)
        geometry_gain = int(
            layers["geometry_unknown"][y0:y1, x0:x1].sum())
        semantic_gain = int(
            layers["semantic_gap"][y0:y1, x0:x1].sum())
        has_geometry = bool(layers["geometry"][ys, xs].any())
        has_semantic = bool(layers["semantic"][ys, xs].any())
        reason = "both" if has_geometry and has_semantic else \
            ("geometry" if has_geometry else "semantic")
        clusters.append({
            "cell": best_cell,
            "centroid_cell": centroid,
            "world": grid.cell_to_world(best_cell),
            "size": int(len(ys)),
            "reason": reason,
            "geometry_gain": geometry_gain,
            "semantic_gain": semantic_gain,
            "information_gain": geometry_gain + semantic_gain,
            "clearance_cells": int(best_key[0]),
        })
    clusters.sort(key=lambda c: -c["size"])
    return (clusters, layers) if return_layers else clusters
