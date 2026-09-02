"""决策输入组装器（Phase 4b，agent 端）。

每次决策时把地图、实例记忆和预计算几何组织成 JSON。VLM 解释实例
文本并选择工具/目标，但不估计坐标或路径长度。

实例表是有界摘要：未报告实例按"最近邻 + 最新 + 任务相关"并集取
top-K（NAV_STATE_MAX_INSTANCES，默认 30），其余折叠为 id 列表，
全文与证据经 search_instances / get_instance 按需查询。
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
    # 导航确认不可达的实例默认不进表（VLM 不会主动再选；REPORT_FOUND
    # 校验仍可用——agent 可能就停在目标旁直接确认）。但 agent 就在不可达
    # 实例旁边时（dist_m ≤ NAV_REPORT_NEAR_DIST_M）保留条目并打 unreachable
    # 标记：走到目标附近即可上报，成功按距离判定，不要求目标在视野内。
    report_near_m = float(os.environ.get("NAV_REPORT_NEAR_DIST_M", "1.0"))
    unreachable_ids = getattr(agent, "_unreachable_instance_ids",
                              None) or set()
    keep_unreachable = set()
    for nd in nodes:
        if nd.iid not in unreachable_ids:
            continue
        if start is None:
            continue
        d = math.hypot(nd.point[0] - start[0],
                       nd.point[1] - start[1]) * scale
        if d <= report_near_m:
            keep_unreachable.add(nd.iid)
    unreported = [nd for nd in nodes
                  if not nd.reported
                  and nd.iid not in (unreachable_ids - keep_unreachable)]
    reported_ids = [nd.iid for nd in nodes if nd.reported]
    reported_instances = [{
        "id": nd.iid,
        "text": _truncate(nd.text),
        "observation_count": len(getattr(nd, "observation_ids", [])),
        "report_claim_id": getattr(nd, "report_claim_id", None),
    } for nd in nodes if nd.reported]

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
    # 实例化去重预筛：每个实例列出邻近实例（instance_dup_radius_m 内），
    # VLM 在实例化复核（resolve_duplicate）前可提前看到空间冲突。
    nearby_radius_m = float(os.environ.get(
        "NAV_INSTANCE_DUPLICATE_RADIUS_M", "3.0"))
    nearby_map = {}
    for nd in nodes:
        rows = []
        for other in nodes:
            if other.iid == nd.iid:
                continue
            d = math.hypot(float(other.point[0] - nd.point[0]),
                           float(other.point[1] - nd.point[1])) * scale
            if d <= nearby_radius_m:
                rows.append({"id": other.iid, "dist_m": round(d, 2),
                             "reported": bool(other.reported)})
        if rows:
            rows.sort(key=lambda row: row["dist_m"])
            nearby_map[nd.iid] = rows[:4]
    for nd in selected:
        row = {
            "id": nd.iid,
            "text": _truncate(nd.text),
            "observation_count": len(getattr(nd, "observation_ids", [])),
            "path_cost_m": _path_cost_m(grid, start, nd.point, scale),
            "dist_m": dist_m.get(nd.iid),
        }
        if nd.iid in unreachable_ids:
            row["unreachable"] = True
        if nd.iid in nearby_map:
            row["nearby"] = nearby_map[nd.iid]
        instances.append(row)
    omitted_ids = [nd.iid for nd in unreported if nd.iid not in selected_ids]

    # 对外只有一套 frontier；路径代价预计算成米制，其余排序细节不暴露。
    frontier_rows = []
    for i, c in enumerate(frontiers or []):
        frontier_rows.append({
            "id": f"f{i}",
            "path_cost_m": (round(float(c["path_cost_m"]), 2)
                            if c.get("path_cost_m") is not None else None),
            "branch_id": c.get("branch_id"),
            "geometry_gain": int(c.get("geometry_gain", 0)),
            "semantic_gain": int(c.get("semantic_gain", 0)),
            "failure_count": int(c.get("failure_count", 0)),
            "recently_attempted": bool(c.get("recently_attempted", False)),
            "novelty": str(c.get("novelty", "unknown")),
        })

    proposals = list(getattr(agent, "_proposals", {}).values())
    proposal_counts = {
        status: sum(1 for row in proposals if row.get("status") == status)
        for status in ("pending", "uncertain", "rejected", "active",
                       "geometry_rejected", "duplicate_review")}

    # 被拒像素记忆 top-10（按拒绝次数）：propose 已被硬过滤，VLM 应
    # 从新视角（靠近目标）重新 propose，而不是换 prompt 在同一帧硬试。
    rejected_spots = sorted(
        ({"frame_id": key[0], "pixel": [key[1], key[2]],
          "count": entry["count"],
          "reason": str(entry.get("reason") or "")[:80],
          "step": entry["step"]}
         for key, entry in getattr(agent, "_rejected_spots", {}).items()),
        key=lambda row: (-row["count"], -row["step"]))[:10]
    # geometry 解析失败的重看导航 top-5（按尝试次数）：系统已把 agent
    # 导航到对应帧位姿附近，VLM 应等到达后从近处新视角重新 propose。
    revisit_targets = []
    for fid, entry in getattr(agent, "_revisit_targets", {}).items():
        dist_m = None
        point = entry.get("point")
        if point is not None and start is not None:
            aligned = agent._aligned_point(np.asarray(point, dtype=np.float64))
            dist_m = round(math.hypot(aligned[0] - start[0],
                                      aligned[1] - start[1]) * scale, 2)
        revisit_targets.append({"frame_id": int(fid),
                                "attempts": int(entry.get("attempts", 0)),
                                "dist_m": dist_m})
    revisit_targets.sort(key=lambda row: (-row["attempts"], row["dist_m"]
                                          if row["dist_m"] is not None
                                          else float("inf")))
    revisit_targets = revisit_targets[:5]

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
        "steps_remaining": max(
            0, int(observation.max_steps) - int(observation.step_count)),
        "instances": instances,
        "instances_total": len(nodes),
        "instances_omitted_ids": omitted_ids,
        "instances_unreachable_ids": sorted(
            str(i) for i in unreachable_ids),
        "reported_instance_ids": reported_ids,
        "reported_instances": reported_instances,
        "report_claims": [claim.as_dict() for claim in
                          getattr(agent.memory, "report_claims", [])[-20:]],
        "frontiers": frontier_rows,
        "frontier_branches": [dict(row) for row in
                              getattr(agent, "_frontier_branches", [])[:8]],
        "proposal_summary": proposal_counts,
        "rejected_spots": rejected_spots,
        "revisit_targets": revisit_targets,
        # VLM 自己维护的跨决策工作记忆（经 set_notes 工具改写）。
        "notes": getattr(agent, "_notes", ""),
        # 最近 3 步高层动作流水（不含进行中的条目）；更早的经
        # get_action_history 按需查询。
        "recent_actions": [
            dict(entry) for entry in getattr(agent, "_action_log", [])
            if entry.get("outcome") is not None][-3:],
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
