"""模式感知多目标规划：开放路径 TSP 排序 + 目标选择。

评测的路径效率指标就是"最优访问序"（evaluator 的 open-TSP DP），
规划器用同款目标直接优化被测量的东西：
- 小规模（<=max_exact）精确 DP；大规模贪心最近邻兜底；
- 距离函数由调用方注入（栅格测地优先，欧氏兜底）；
- 每次只执行序列第一段，有新信息（新实例/确认失败/地图更新）重规划，
  避免两个候选间来回折返（BFM）。
"""

import itertools

import numpy as np


def route_order(start_xy, goals, dist_fn, max_exact=8):
    """返回访问 goals 的最优顺序（索引列表）；空 goals 返回 []。

    goals: [(x, y), ...]；dist_fn(a, b) -> 距离（a/b 为 (x, y) 元组）。
    """
    n = len(goals)
    if n == 0:
        return []
    if n == 1:
        return [0]
    start_d = [dist_fn(start_xy, g) for g in goals]
    if n > max_exact:
        # 贪心最近邻
        order, remaining, cur = [], set(range(n)), None
        while remaining:
            if cur is None:
                nxt = min(remaining, key=lambda i: start_d[i])
            else:
                nxt = min(remaining,
                          key=lambda i: dist_fn(goals[cur], goals[i]))
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        return order

    pair = {}
    for i, j in itertools.permutations(range(n), 2):
        pair[(i, j)] = dist_fn(goals[i], goals[j])

    layers = [{(1 << i, i): start_d[i] for i in range(n)}]
    full = (1 << n) - 1
    for _ in range(1, n):
        prev_layer = layers[-1]
        nxt_dp = {}
        for (mask, last), dist in prev_layer.items():
            for j in range(n):
                if mask & (1 << j):
                    continue
                state = (mask | (1 << j), j)
                val = dist + pair[(last, j)]
                if state not in nxt_dp or val < nxt_dp[state]:
                    nxt_dp[state] = val
        layers.append(nxt_dp)
    # 回溯最优序列
    best_last = min(range(n), key=lambda i: layers[-1][(full, i)])
    order, mask, last = [], full, best_last
    while mask:
        order.append(last)
        prev_mask = mask & ~(1 << last)
        if prev_mask:
            # int.bit_count() is unavailable in the benchmark's Python 3.9.
            layer = layers[bin(prev_mask).count("1") - 1]
            last = min((i for i in range(n) if prev_mask & (1 << i)),
                       key=lambda i: layer[(prev_mask, i)] + pair[(i, last)])
        mask = prev_mask
    order.reverse()
    return order


def select_goal_any(start_xy, instances, dist_fn, score_weight=1.0):
    """any 模式：score / (1 + 距离) 最高的已确认实例。"""
    best, best_v = None, -1.0
    for nd in instances:
        d = dist_fn(start_xy, tuple(nd.point[:2]))
        v = nd.score * score_weight / (1.0 + d)
        if v > best_v:
            best, best_v = nd, v
    return best


def plan_multi(start_xy, instances, dist_fn, need):
    """many/all 模式：返回 (有序访问列表, 缺口)。

    instances: 已确认未访问实例；need: 还需访问的数量。
    列表长度 <= need（不够的部分由探索补足，即缺口 > 0）。
    """
    if not instances or need <= 0:
        return [], max(need, 0)
    goals = [tuple(nd.point[:2]) for nd in instances]
    order = route_order(start_xy, goals, dist_fn)
    chosen = [instances[i] for i in order[:need]]
    return chosen, max(need - len(chosen), 0)
