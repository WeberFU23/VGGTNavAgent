"""统一 3D instance memory 与确定性路径规划回归测试。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory import InstanceMemory
from agents import planner


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_each_candidate_becomes_instance_without_proximity_merge():
    mem = InstanceMemory()
    n1, fresh1 = mem.remember([0, 0, 0], "first view", candidate_id="c1")
    n2, fresh2 = mem.remember([0.1, 0, 0], "second view", candidate_id="c2")
    assert fresh1 and fresh2
    assert n1.iid != n2.iid
    assert len(mem.nodes) == 2


def test_stable_candidate_updates_same_instance_and_evidence():
    mem = InstanceMemory()
    n1, _ = mem.remember([0, 0, 0], "possible cup",
                         evidence=[{"frame_id": 1}], candidate_id="c1")
    n2, fresh = mem.remember([1, 0, 0], "ignored replacement",
                             evidence=[{"frame_id": 2}], candidate_id="c1")
    assert not fresh and n1 is n2
    assert np.allclose(n1.point, [1, 0, 0])
    assert n1.text == "possible cup"
    assert len(n1.evidence) == 2


def test_vlm_can_update_and_merge_instances():
    mem = InstanceMemory()
    n1, _ = mem.remember([0, 0, 0], "view A", candidate_id="a")
    n2, _ = mem.remember([2, 0, 0], "view B", candidate_id="b")
    mem.update_text(n1.iid, "same red cup seen from the left")
    merged = mem.merge([n2.iid, n1.iid], text="red cup; two views")
    assert merged.iid == n1.iid
    assert merged.text == "red cup; two views"
    assert np.allclose(merged.point, [1, 0, 0])
    assert len(mem.nodes) == 1


def test_reported_instances_are_not_available():
    mem = InstanceMemory()
    n1 = mem.add([0, 0, 0], "cup")
    n2 = mem.add([5, 0, 0], "cup-like object")
    assert mem.mark_reported(n1)
    assert not mem.mark_reported(n1)
    assert mem.available() == [n2]
    assert mem.count_reported() == 1


def test_planner_uses_all_unreported_instances():
    mem = InstanceMemory()
    for x in (5, 1, 3):
        mem.add([x, 0, 0], f"candidate at {x}")
    selected = planner.select_goal_any((0, 0), mem.available(), _dist)
    assert np.allclose(selected.point[:2], [1, 0])
    ordered, _gap = planner.plan_multi((0, 0), mem.available(), _dist, need=3)
    assert {node.iid for node in ordered} == {1, 2, 3}


if __name__ == "__main__":
    test_each_candidate_becomes_instance_without_proximity_merge()
    test_stable_candidate_updates_same_instance_and_evidence()
    test_vlm_can_update_and_merge_instances()
    test_reported_instances_are_not_available()
    test_planner_uses_all_unreported_instances()
    print("memory/planner tests passed")
