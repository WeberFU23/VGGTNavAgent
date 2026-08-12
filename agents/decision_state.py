"""决策输入组装器（Phase 4b，agent 端）。

每次决策时把世界状态预计算好喂给 VLM——VLM 不记账、不算几何，
dist/path_cost/覆盖率全部由这里用确定性状态机的数据算好；唯一真源
是状态机。输出与俯视标注地图（map_render）中的编号严格一一对应。
"""

import math

import numpy as np


def build_world_state(agent, observation, grid=None, frontiers=None):
    """组装决策用世界状态 JSON。

    agent: NavAgent（访问 memory/ledger/belief/calibrator/_events）；
    grid: 当前占据栅格（有则预计算 A* path_cost）；
    frontiers: 当前 frontier clusters（与地图上的编号顺序一致）。
    """
    scale = agent.calibrator.current_scale() or 1.0
    start = agent._current_aligned_xy()
    goal_text = str(getattr(observation, "goal_text", "") or "")

    # 实例表：confirmed + visited 都列出（VLM 不得重选 visited）
    instances = []
    for nd in agent.memory.nodes:
        if nd.status == "rejected":
            continue
        dist_m = None
        if start is not None:
            dist_m = math.hypot(nd.point[0] - start[0],
                                nd.point[1] - start[1]) * scale
        instances.append({
            "id": nd.iid,
            "category": nd.category,
            "status": nd.status,
            "confidence": round(nd.score, 3),
            "n_obs": nd.n_obs,
            "frame_id": nd.frame_id,
            "candidate_id": nd.candidate_id,
            "dist_m": round(dist_m, 2) if dist_m is not None else None,
            "path_cost_m": _path_cost_m(grid, start, nd.point, scale),
        })

    # belief 锚点表（未复核观测，探索先验）
    anchors = []
    for i, a in enumerate(agent.ledger.belief_anchors(agent.target_text)):
        dist_m = None
        if start is not None:
            dist_m = math.hypot(a.point[0] - start[0],
                                a.point[1] - start[1]) * scale
        anchors.append({
            "id": f"b{i}",
            "category": a.category,
            "confidence": round(a.score, 3),
            "n_obs": a.n_obs,
            "dist_m": round(dist_m, 2) if dist_m is not None else None,
        })

    # frontier 表（语义线索来自 caption 检索信念）
    frontier_rows = []
    for i, c in enumerate(frontiers or []):
        dist_m = None
        if start is not None:
            dist_m = math.hypot(c["world"][0] - start[0],
                                c["world"][1] - start[1]) * scale
        frontier_rows.append({
            "id": f"f{i}",
            "dist_m": round(dist_m, 2) if dist_m is not None else None,
            "size": int(c.get("size", 0)),
            "information_gain": int(c.get("information_gain", 0)),
            "path_cost_m": (round(float(c["path_cost_m"]), 2)
                            if c.get("path_cost_m") is not None else None),
            "semantic_hint": round(float(c.get(
                "semantic_hint", agent.belief.belief_at(c["world"], None))), 3),
            "failure_count": int(c.get("failure_count", 0)),
            "utility": (round(float(c["utility"]), 3)
                        if c.get("utility") is not None else None),
        })

    # 近期事件：最近 10 条原文 + 更早压缩统计
    events = list(getattr(agent, "_events", []))
    recent = events[-10:]

    # 终止账本
    unexplored_ratio = None
    if grid is not None:
        observed = getattr(grid, "observed",
                           np.asarray(grid.free) | np.asarray(grid.obstacle))
        unknown = ~np.asarray(observed, dtype=bool)
        unexplored_ratio = float(unknown.sum()) / float(unknown.size)

    return {
        "task": {
            "goal": goal_text,
            "mode": agent._target_mode,
            "found": agent._reported_count,
            "expected": (int(agent._target_count)
                         if agent._target_count is not None else None),
        },
        "step": int(observation.step_count),
        "max_steps": int(observation.max_steps),
        "instances": instances,
        "belief_anchors": anchors,
        "frontiers": frontier_rows,
        "recent_events": recent,
        "older_events_total": max(0, len(events) - len(recent)),
        "termination": {
            "unexplored_ratio": (round(unexplored_ratio, 4)
                                 if unexplored_ratio is not None else None),
            "unresolved_anchor_count":
                agent.ledger.count_unresolved(agent.target_text),
            "frontier_count": len(frontiers or []),
            "reachable_frontier_count": getattr(
                agent, "_last_reachable_frontier_count", len(frontiers or [])),
            "pending_instance_count": len(
                agent.memory.unvisited(agent.target_text)),
            "recent_queries_without_new_candidate": agent._no_hit_queries,
        },
    }


def _path_cost_m(grid, start, point, scale):
    """A* 路径代价（米），预计算好——VLM 不算几何。不可达返回 None。"""
    if grid is None or start is None:
        return None
    try:
        path = grid.astar(start, tuple(np.asarray(point)[:2]))
        if not path or len(path) < 2:
            return None
        length = sum(float(np.linalg.norm(
            np.asarray(path[i + 1]) - np.asarray(path[i])))
            for i in range(len(path) - 1))
        return round(length * scale, 2)
    except Exception:
        return None
