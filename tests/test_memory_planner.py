"""统一 3D instance memory 与确定性路径规划回归测试。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory import InstanceMemory
from agents.entity_resolver import EntityResolver
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


def test_same_frame_reinstantiation_is_observation_idempotent():
    mem = InstanceMemory()
    n1, fresh1 = mem.remember(
        [0, 0, 0], "view A", frame_id=7, candidate_id="a",
        pixel=[100, 100])
    n2, fresh2 = mem.remember(
        [0.02, 0, 0], "view B", frame_id=7, candidate_id="b",
        pixel=[104, 103])
    assert fresh1 and not fresh2 and n1 is n2
    assert len(mem.nodes) == 1
    assert len(mem.observations) == 1


def test_replaying_old_view_keeps_its_observation_and_canonical_point():
    mem = InstanceMemory()
    first = mem.new_observation(
        [0, 0, 0], "chair front",
        evidence={"point_score": 0.2}, frame_id=1,
        candidate_id="c1", pixel=[100, 100])
    node = mem.create_instance(first)
    second = mem.new_observation(
        [1, 0, 0], "chair side",
        evidence={"point_score": 0.9}, frame_id=2,
        candidate_id="c2", pixel=[200, 200])
    mem.attach_observation(node, second)

    replay = mem.find_replay(
        candidate_id="c3", frame_id=1, pixel=[103, 102])
    mem.register_replay(
        replay, candidate_id="c3",
        evidence={"frame_id": 1, "pixel": [103, 102]},
        point=[9, 0, 0], step=10)

    assert mem._candidate_to_observation["c3"] == first.oid
    assert len(mem.observations) == 2
    assert np.allclose(first.point, [9, 0, 0])
    assert np.allclose(node.point, second.point)  # best real view stays canonical
    assert node.step == 10


def test_entity_resolver_attaches_cross_frame_same_object():
    mem = InstanceMemory()
    first = mem.new_observation(
        [0, 0, 0], "chair front", frame_id=1, candidate_id="c1")
    node = mem.create_instance(first)
    second = mem.new_observation(
        [0.1, 0, 0], "chair side", frame_id=2, candidate_id="c2")
    resolver = EntityResolver(candidate_radius_m=1.0)
    result = resolver.resolve(
        mem, second, scale=1.0,
        compare_fn=lambda observation, nearby: {
            "decision": "SAME", "instance_id": node.iid,
            "description": "same chair from two views",
            "reason": "same distinctive back",
        })
    assert not result.is_new and result.node is node
    assert len(node.observation_ids) == 2
    assert node.text == "same chair from two views"


def test_entity_resolver_preserves_uncertain_neighbor_as_new():
    mem = InstanceMemory()
    mem.create_instance(mem.new_observation(
        [0, 0, 0], "chair A", frame_id=1, candidate_id="c1"))
    second = mem.new_observation(
        [0.3, 0, 0], "chair B", frame_id=2, candidate_id="c2")
    result = EntityResolver(candidate_radius_m=1.0).resolve(
        mem, second, scale=1.0,
        compare_fn=lambda observation, nearby: {
            "decision": "UNCERTAIN", "instance_id": None,
            "description": "possibly another chair",
        })
    assert result.is_new and len(mem.nodes) == 2


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
    test_same_frame_reinstantiation_is_observation_idempotent()
    test_replaying_old_view_keeps_its_observation_and_canonical_point()
    test_entity_resolver_attaches_cross_frame_same_object()
    test_entity_resolver_preserves_uncertain_neighbor_as_new()
    test_reported_instances_are_not_available()
    test_planner_uses_all_unreported_instances()
    print("memory/planner tests passed")
