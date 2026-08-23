"""决策输入组装器（Phase 4b，agent 端）。

每次决策时把地图、实例记忆和预计算几何组织成 JSON。VLM 解释实例
文本并选择工具/目标，但不估计坐标或路径长度。

实例表是有界摘要：未报告实例按"最近邻 + 最新 + 任务相关"并集取
top-K（NAV_STATE_MAX_INSTANCES，默认 30），其余折叠为 id 列表，
全文与证据经 search_instances / inspect_instance 按需查询。
"""

import math
import os
import re

import numpy as np

# world-state 实例表上限与单行文本截断长度
MAX_STATE_INSTANCES = 30
STATE_TEXT_CHARS = 120

_STOPWORDS = {"the", "and", "for", "with", "that", "this"}


def _task_keywords(text):
    """任务短语的确定性关键词（实例相关性排序用，不做语义判断）。"""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [w for w in dict.fromkeys(words)
            if len(w) >= 3 and w not in _STOPWORDS]


def _truncate(text):
    text = str(text or "")
    return text if len(text) <= STATE_TEXT_CHARS \
        else text[:STATE_TEXT_CHARS - 3] + "..."


def build_world_state(agent, observation, grid=None, frontiers=None,
                      start_xy=None, scale=None):
    """组装决策用世界状态 JSON。

    agent: NavAgent（访问 memory/calibrator/_events）；
    grid: 当前占据栅格（有则预计算 A* path_cost）；
    frontiers: 当前 frontier clusters（与地图上的编号顺序一致）。
    start_xy: 与该栅格快照一致的当前二维位置；未提供时兼容旧调用。
    """
    scale = scale or agent.calibrator.current_scale() or 1.0
    start = (np.asarray(start_xy, dtype=np.float64)[:2]
             if start_xy is not None else agent._current_aligned_xy())
    goal_text = str(getattr(observation, "goal_text", "") or "")

    # 实例表：未报告实例的 top-K 摘要 + 折叠列表。A* 路径代价只对入选
    # 摘要的实例预计算（长 episode 下对全部实例跑 A* 太贵）。
    max_instances = int(os.environ.get(
        "NAV_STATE_MAX_INSTANCES", str(MAX_STATE_INSTANCES)))
    nodes = list(agent.memory.nodes)
    unreported = [nd for nd in nodes if not nd.reported]
    reported_ids = [nd.iid for nd in nodes if nd.reported]

    dist_m = {}
    for nd in unreported:
        d = None
        if start is not None:
            d = math.hypot(nd.point[0] - start[0],
                           nd.point[1] - start[1]) * scale
        dist_m[nd.iid] = round(d, 2) if d is not None else None

    keywords = _task_keywords(
        getattr(agent, "target_text", "") or goal_text)
    selected = _select_summary(unreported, dist_m, keywords, max_instances)
    selected_ids = {nd.iid for nd in selected}

    instances = []
    for nd in selected:
        instances.append({
            "id": nd.iid,
            "text": _truncate(nd.text),
            "reported": False,
            "evidence_count": len(nd.evidence),
            "dist_m": dist_m[nd.iid],
            "path_cost_m": _path_cost_m(grid, start, nd.point, scale),
        })
    omitted_ids = [nd.iid for nd in unreported if nd.iid not in selected_ids]

    # 对外只有一套 frontier；reason/gain 解释它主要补几何还是语义信息。
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
            "reason": c.get("reason", "geometry"),
            "geometry_gain": int(c.get("geometry_gain", 0)),
            "semantic_gain": int(c.get("semantic_gain", 0)),
            "information_gain": int(c.get("information_gain", 0)),
            "path_cost_m": (round(float(c["path_cost_m"]), 2)
                            if c.get("path_cost_m") is not None else None),
            "failure_count": int(c.get("failure_count", 0)),
            "utility": (round(float(c["utility"]), 3)
                        if c.get("utility") is not None else None),
        })

    # 近期事件：最近 10 条原文 + 更早压缩统计
    events = list(getattr(agent, "_events", []))
    recent = events[-10:]

    # 终止账本
    unexplored_ratio = None
    coverage = None
    if grid is not None:
        geometry = np.asarray(getattr(
            grid, "geometry_observed",
            getattr(grid, "observed",
                    np.asarray(grid.free) | np.asarray(grid.obstacle))),
            dtype=bool)
        free = np.asarray(grid.free, dtype=bool)
        semantic_enabled = bool(getattr(
            grid, "semantic_coverage_enabled", False))
        semantic = np.asarray(getattr(
            grid, "semantic_inspected", geometry), dtype=bool)
        geometry_missing = ~geometry
        semantic_missing = free & ~semantic if semantic_enabled \
            else np.zeros_like(free)
        incomplete = geometry_missing | semantic_missing
        unexplored_ratio = float(incomplete.sum()) / float(incomplete.size)
        free_count = max(int(free.sum()), 1)
        coverage = {
            "semantic_enabled": semantic_enabled,
            "geometry_unobserved_ratio": round(
                float(geometry_missing.sum()) / float(geometry.size), 4),
            "semantic_uninspected_free_ratio": (
                round(float(semantic_missing.sum()) / free_count, 4)
                if semantic_enabled else None),
            "incomplete_ratio": round(unexplored_ratio, 4),
        }

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
        "instances_total": len(nodes),
        "instances_omitted_ids": omitted_ids,
        "reported_instance_ids": reported_ids,
        "frontiers": frontier_rows,
        "map_coverage": coverage,
        "recent_events": recent,
        "older_events_total": max(0, len(events) - len(recent)),
        "termination": {
            "unexplored_ratio": (round(unexplored_ratio, 4)
                                 if unexplored_ratio is not None else None),
            "frontier_count": len(frontiers or []),
            "reachable_frontier_count": getattr(
                agent, "_last_reachable_frontier_count", len(frontiers or [])),
            "frontier_filters": dict(getattr(
                agent, "_frontier_stats", {}) or {}),
            "unreported_instance_count": len(agent.memory.available()),
            "recent_queries_without_new_candidate": agent._no_hit_queries,
        },
    }


def _select_summary(unreported, dist_m, keywords, k):
    """top-K 摘要选择：最近邻 ∪ 最新 ∪ 任务关键词相关，按距离排序。

    纯确定性几何/文本排序，不做语义判断；k 是硬上限。"""
    if len(unreported) <= k:
        return sorted(unreported,
                      key=lambda nd: _dist_key(nd, dist_m))
    m = max(1, k // 3)
    by_dist = sorted(unreported, key=lambda nd: _dist_key(nd, dist_m))
    by_recency = sorted(unreported, key=lambda nd: (-nd.step, -nd.iid))
    scored = []
    for nd in unreported:
        haystack = nd.text.lower()
        hits = sum(1 for w in keywords if w in haystack)
        if hits > 0:
            scored.append((hits, nd))
    relevant = [nd for _hits, nd in sorted(
        scored, key=lambda s: (-s[0], _dist_key(s[1], dist_m)))]

    chosen, seen = [], set()
    for group in (by_dist[:m], by_recency[:m], relevant[:m]):
        for nd in group:
            if nd.iid not in seen:
                seen.add(nd.iid)
                chosen.append(nd)
    for nd in by_dist:                      # 并集不足 K 时按距离补齐
        if len(chosen) >= k:
            break
        if nd.iid not in seen:
            seen.add(nd.iid)
            chosen.append(nd)
    return sorted(chosen[:k], key=lambda nd: _dist_key(nd, dist_m))


def _dist_key(nd, dist_m):
    d = dist_m.get(nd.iid)
    return (d is None, d if d is not None else 0.0, nd.iid)


def _path_cost_m(grid, start, point, scale):
    """A* 路径代价（米），预计算好——VLM 不算几何。不可达返回 None。"""
    if grid is None or start is None:
        return None
    try:
        path = grid.astar(start, tuple(np.asarray(point)[:2]))
        if not path:
            return None
        if len(path) == 1:
            return 0.0
        length = sum(float(np.linalg.norm(
            np.asarray(path[i + 1]) - np.asarray(path[i])))
            for i in range(len(path) - 1))
        return round(length * scale, 2)
    except Exception:
        return None
