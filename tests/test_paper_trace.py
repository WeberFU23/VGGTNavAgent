"""Diagnostics must preserve decisions, prompts and mutable-memory semantics."""

import copy
import json
from types import SimpleNamespace

import numpy as np

from agents.nav_agent import NavAgent
from decision import DecisionLoop, DecisionTraceLogger


def state():
    return {"step": 7, "max_steps": 100, "steps_remaining": 93,
            "task": {"mode": "many", "found": 0, "expected": 2},
            "instances": [{"id": 1, "path_cost_m": 2.0}],
            "frontiers": [{"id": "f0", "path_cost_m": 1.0}]}


def run_script(trace=False, broken=False):
    current = state()
    calls, records = [], []
    replies = iter([
        {"tool_call": {"name": "commit_candidates", "reviews": []}},
        {"action": "FINISH", "reason": "done"},
    ])

    def chat(prompt, images):
        calls.append((prompt, images))
        return next(replies)

    def commit_candidates(reviews):
        current["instances"].append({"id": 2, "path_cost_m": 4.0})
        return {"instances": [2], "long_evidence": "e" * 8000}

    class BrokenLogger:
        def log(self, record):
            raise OSError("disk full")

    loop = DecisionLoop(chat, tools={"commit_candidates": commit_candidates},
                        logger=BrokenLogger() if broken else None)
    opts = {"trace_sink": records.append, "trace_context": {"episode": "ep"},
            "trace_snapshot_fn": lambda: {"instances": current["instances"]}} if trace else {}
    result = loop.decide("world_state_updated", current,
                         state_fn=lambda: current, **opts)
    return result, calls, records, current


def test_capture_and_disk_failure_preserve_prompts_tools_and_action():
    plain, plain_calls, _, _ = run_script()
    logged, logged_calls, records, current = run_script(trace=True, broken=True)
    assert plain.as_dict() == logged.as_dict()
    assert plain_calls == logged_calls  # no new content or tool invocations
    record = records[0]
    assert record["model_outputs"][-1]["output"]["action"] == "FINISH"
    assert record["output"]["action"] == "GOTO_INSTANCE"
    assert record["decision_origin"] == "harness"
    assert record["validation"] == "finish_downgraded"
    assert len(record["tools"][0]["response"]["result"]["long_evidence"]) == 8000
    assert '"truncated": true' in logged_calls[1][0]
    assert [r["id"] for r in record["world_state"]["instances"]] == [1, 2]
    current["instances"][1]["path_cost_m"] = 999  # later SLAM/state changes
    assert record["world_state"]["instances"][1]["path_cost_m"] == 4.0
    assert record["agent_snapshot"]["instances"][1]["path_cost_m"] == 4.0


def test_decisions_on_same_step_have_distinct_ids_and_call_correlation(tmp_path):
    records, call_ids = [], []
    loop = DecisionLoop(lambda *_: {"action": "GOTO_INSTANCE", "target_id": 1},
                        logger=DecisionTraceLogger(tmp_path / "trace.jsonl"))
    for _ in range(2):
        loop.decide("arrival", state(), trace_sink=records.append,
                    trace_call_context=lambda did, idx: call_ids.append((did, idx)))
    assert records[0]["decision_id"] != records[1]["decision_id"]
    assert call_ids == [(r["decision_id"], 1) for r in records]
    disk = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(disk) == 2 and disk[0]["world_state"] == state()


def test_snapshot_failure_keeps_valid_decision_and_reports_missing_data():
    records = []
    def broken():
        raise RuntimeError("snapshot unavailable")
    result = DecisionLoop(lambda *_: {"action": "GOTO_INSTANCE", "target_id": 1}).decide(
        "arrival", state(), trace_snapshot_fn=broken, trace_sink=records.append)
    assert result.action == "GOTO_INSTANCE"
    assert records[0]["snapshot_error"] == "snapshot unavailable"


def test_agent_snapshot_is_read_only_and_keeps_identity_without_world_transform():
    agent = NavAgent()
    node = agent.memory.add(point=[1., 2., 3.], text="cup")
    before = copy.deepcopy(node.point)
    # No RPC is allowed from either diagnostic getter.
    class NoRPC:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected RPC: {name}")
    agent.client = NoRPC()
    record = agent.get_paper_trace()
    assert record["final_memory"]["instances"][0]["instance_id"] == node.iid
    assert record["final_memory"]["world_pool_status"] == "transform_unavailable"
    assert np.array_equal(node.point, before)
    record["final_memory"]["instances"][0]["point"][0] = 999
    assert np.array_equal(node.point, before)
    # Restore a harmless client for existing atexit finalization.
    agent.client = SimpleNamespace()


def test_optional_pool_ids_preserve_existing_contract():
    agent = NavAgent()
    agent._pool_world_anchor = (np.zeros(3), 0.)
    agent._pool_slam_anchor = (0., 0., 0., 0.)
    agent._metric_snapshot.update(scale=1., revision=1)
    node = agent.memory.add(point=[1., 2., 0.], text="cup")
    old = agent.get_target_pool()
    diagnostic = agent.get_target_pool(include_ids=True)
    assert set(old[0]) == {"position", "label", "reported"}
    assert diagnostic[0].pop("instance_id") == node.iid
    assert diagnostic == old
