"""memory + planner 单元测试。

    python tests/test_memory_planner.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import planner
from agents.belief import BeliefMap
from agents.memory import InstanceMemory


def euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def test_merge_and_dedupe():
    mem = InstanceMemory()
    n1, new1 = mem.add_or_merge("bag", [0, 0, 0], 0.9, merge_dist=1.0)
    n2, new2 = mem.add_or_merge("bag", [0.3, 0.2, 0], 0.8, merge_dist=1.0)
    assert new1 and not new2 and n1 is n2, "近距离同类应合并"
    assert n1.score == 0.9, "低分不覆盖高分"
    n3, new3 = mem.add_or_merge("bag", [5, 5, 0], 0.85, merge_dist=1.0)
    assert new3, "远处同类是新实例（第二个包）"
    n4, new4 = mem.add_or_merge("book", [0.3, 0.2, 0], 0.8, merge_dist=1.0)
    assert new4, "不同类别不合并"
    print(f"merge OK: {mem.nodes}")


def test_status_flow():
    mem = InstanceMemory()
    n1, _ = mem.add_or_merge("bag", [0, 0, 0], 0.9, 1.0)
    n2, _ = mem.add_or_merge("bag", [5, 5, 0], 0.85, 1.0)
    assert len(mem.unvisited("bag")) == 2
    mem.mark_visited(n1)
    assert mem.count_visited("bag") == 1
    assert len(mem.unvisited("bag")) == 1
    # 已访问实例附近的新命中应被识别（不会再次导航过去）
    assert mem.is_visited("bag", [0.2, 0.1, 0], 1.0)
    assert not mem.is_visited("bag", [5, 5, 0], 1.0)
    mem.mark_rejected(n2)
    assert mem.is_rejected("bag", [5.3, 5, 0], 1.0)
    assert len(mem.unvisited("bag")) == 0
    # visited/rejected 不被低分新命中降级
    n1b, new = mem.add_or_merge("bag", [0.1, 0, 0], 0.99, 1.0)
    assert not new and n1b.status == "visited"
    print("status flow OK")


def test_route_order_optimal():
    # 一条线上 3 个点，起点在一端：最优序显然 1->2->3
    goals = [(10, 0), (2, 0), (5, 0)]
    order = planner.route_order((0, 0), goals, euclid)
    assert order == [1, 2, 0], f"order={order}"
    # 起点在中间的折返陷阱：贪心从最近开始也可能错，DP 应全局最优
    goals = [(1, 0), (10, 0), (2, 0), (11, 0)]
    order = planner.route_order((0, 0), goals, euclid)
    total = sum(euclid(*( ( (0,0) if k==0 else goals[order[k-1]] ), goals[order[k]]))
                for k in range(len(order)))
    assert total <= 21.0, f"route length={total}"
    print(f"route DP OK: order={order} length={total:.1f}")


def test_plan_multi():
    mem = InstanceMemory()
    for i, x in enumerate((2, 5, 10)):
        mem.add_or_merge("bag", [x, 0, 0], 0.9, 1.0)
    inst = mem.unvisited("bag")
    chosen, gap = planner.plan_multi((0, 0), inst, euclid, need=2)
    assert len(chosen) == 2 and gap == 0
    chosen, gap = planner.plan_multi((0, 0), inst, euclid, need=5)
    assert len(chosen) == 3 and gap == 2, "实例不足时报告探索缺口"
    print("plan_multi OK: need=2 全满足, need=5 缺口=2")


def test_candidate_identity_and_status_guard():
    mem = InstanceMemory()
    node, _ = mem.add_or_merge(
        "sink", [0, 0, 0], 0.8, 0.5, candidate_id="c1")
    # 同一 candidate 经图优化移动后仍是同一实例，不受距离门槛影响。
    same, is_new = mem.add_or_merge(
        "sink", [2, 0, 0], 0.9, 0.5, candidate_id="c1")
    assert same is node and not is_new
    mem.refresh_point(node, [2, 0, 0])
    mem.mark_visited(node)
    mem.mark_rejected(node)
    assert node.status == "visited", "visited 不得被后续拒绝降级"


def test_belief_uses_spatial_anchors_not_node_ids():
    class Client:
        @staticmethod
        def query_text(_text, top_k=10):
            pose = np.eye(4)
            pose[:3, 3] = [3.0, 2.0, 0.0]
            return [{"frame_id": 7, "score": 0.8, "pose": pose}]

    class Graph:
        nodes = [{"id": 1, "world": (3.0, 2.0)}]

    belief = BeliefMap(query_interval=1)
    belief.update(Client(), Graph(), "sink", np.eye(3), step=1)
    near = belief.belief_at((3.0, 2.0), Graph())
    # 新图复用了 id=1，但位置改变；信念仍应留在原空间位置。
    Graph.nodes = [{"id": 1, "world": (30.0, 20.0)}]
    far = belief.belief_at((30.0, 20.0), Graph())
    assert near > far


if __name__ == "__main__":
    test_merge_and_dedupe()
    test_status_flow()
    test_route_order_optimal()
    test_plan_multi()
    test_candidate_identity_and_status_guard()
    test_belief_uses_spatial_anchors_not_node_ids()
    print("ALL MEMORY/PLANNER TESTS PASSED")
