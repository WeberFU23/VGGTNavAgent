"""Automatic cross-view association for canonical object instances."""

import json
import os
import threading
import time


class ResolutionResult:
    __slots__ = ("node", "observation", "is_new", "method", "verdict",
                 "reason")

    def __init__(self, node, observation=None, is_new=False,
                 method="new", verdict="NEW", reason=""):
        self.node = node
        self.observation = observation
        self.is_new = bool(is_new)
        self.method = str(method)
        self.verdict = str(verdict)
        self.reason = str(reason or "")[:300]


class EntityResolver:
    """Resolve one new observation to a canonical InstanceNode.

    Geometry only retrieves plausible candidates. Cross-frame identity is
    accepted exclusively from the focused visual relation call supplied by the
    agent. Unavailable, invalid or uncertain judgments create a new entity to
    preserve recall.
    """

    def __init__(self, candidate_radius_m=1.2, max_candidates=3,
                 trace_path=None):
        self.candidate_radius_m = max(0.05, float(candidate_radius_m))
        self.max_candidates = max(1, int(max_candidates))
        self.trace_path = str(trace_path or "").strip()
        self._lock = threading.Lock()
        self._warned = False

    @classmethod
    def from_env(cls, trace_path=None):
        return cls(
            candidate_radius_m=float(os.environ.get(
                "NAV_ENTITY_CANDIDATE_RADIUS_M", "1.2")),
            max_candidates=int(os.environ.get(
                "NAV_ENTITY_MAX_CANDIDATES", "3")),
            trace_path=trace_path)

    def resolve(self, memory, observation, scale, compare_fn=None):
        nearby = memory.nearby(
            observation.point, scale, self.candidate_radius_m,
            top_k=self.max_candidates)
        candidates = [node for _distance, node in nearby]
        distances = {node.iid: distance for distance, node in nearby}
        response = None
        if callable(compare_fn):
            try:
                response = compare_fn(observation, nearby)
            except Exception as exc:
                response = {
                    "decision": "UNCERTAIN",
                    "reason": f"resolver exception: {type(exc).__name__}",
                }

        verdict, matched, description, reason = self._validate_response(
            response, candidates)
        if verdict == "SAME" and matched is not None:
            node = memory.attach_observation(
                matched, observation, text=description)
            result = ResolutionResult(
                node, observation, is_new=False, method="visual_relation",
                verdict=verdict, reason=reason)
        else:
            node = memory.create_instance(observation, text=description)
            result = ResolutionResult(
                node, observation, is_new=True,
                method=("visual_relation" if response is not None
                        else "new_without_visual_relation"),
                verdict=verdict, reason=reason)
        self._trace(observation, distances, result)
        return result

    @staticmethod
    def _validate_response(response, candidates):
        if not isinstance(response, dict):
            return "UNCERTAIN", None, "", "visual resolver unavailable"
        verdict = str(response.get("decision") or "UNCERTAIN").upper()
        if verdict not in {"SAME", "NEW", "UNCERTAIN"}:
            verdict = "UNCERTAIN"
        description = str(response.get("description") or "")[:2000]
        reason = str(response.get("reason") or "")[:300]
        if verdict != "SAME":
            return verdict, None, description, reason
        try:
            iid = int(response.get("instance_id"))
        except (TypeError, ValueError):
            return ("UNCERTAIN", None, description,
                    "SAME missing valid instance_id")
        matched = next((node for node in candidates if node.iid == iid), None)
        if matched is None:
            return ("UNCERTAIN", None, description,
                    "SAME id not in candidate set")
        return verdict, matched, description, reason

    def _trace(self, observation, distances, result):
        if not self.trace_path:
            return
        record = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "observation_id": observation.oid,
            "frame_id": observation.frame_id,
            "candidate_id": observation.candidate_id,
            "candidate_instances": [
                {"id": iid, "distance_m": round(float(distance), 3)}
                for iid, distance in sorted(distances.items(),
                                            key=lambda item: item[1])],
            "decision": result.verdict,
            "canonical_instance_id": result.node.iid,
            "is_new": result.is_new,
            "method": result.method,
            "reason": result.reason,
        }
        try:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            with self._lock, open(self.trace_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            if not self._warned:
                print(f"[EntityResolver] trace write failed: {exc}", flush=True)
                self._warned = True
