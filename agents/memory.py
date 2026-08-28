"""Canonical object memory built from immutable pointing observations.

``candidate_id`` identifies mapping evidence, not a physical object. Every
pointing result first becomes an :class:`ObservationRecord`; the entity
resolver then attaches it to one canonical :class:`InstanceNode`. Reporting
is an atomic claim on that canonical instance.
"""

import math

import numpy as np


class ObservationRecord:
    """One append-only visual observation of a possible object.

    Its identity and evidence stay fixed; only the mapped 3D point may be
    refreshed after SLAM loop closure.
    """

    __slots__ = ("oid", "point", "text", "evidence", "frame_id", "step",
                 "candidate_id", "pixel", "bbox")

    def __init__(self, oid, point, text="", evidence=None, frame_id=None,
                 step=0, candidate_id=None, pixel=None, bbox=None):
        self.oid = int(oid)
        self.point = np.asarray(point, dtype=np.float64)
        self.text = str(text or "")[:2000]
        self.evidence = dict(evidence or {})
        self.frame_id = frame_id
        self.step = int(step)
        self.candidate_id = candidate_id
        self.pixel = _pair(pixel)
        self.bbox = _quad(bbox)

    def as_dict(self):
        return {
            "observation_id": self.oid,
            "point": [round(float(value), 3) for value in self.point],
            "text": self.text,
            "frame_id": self.frame_id,
            "candidate_id": self.candidate_id,
            "pixel": list(self.pixel) if self.pixel is not None else None,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "step": self.step,
        }


class ReportClaim:
    """An emitted TARGET_FOUND transaction for one canonical instance."""

    __slots__ = ("claim_id", "instance_id", "step", "observation_ids")

    def __init__(self, claim_id, instance_id, step, observation_ids):
        self.claim_id = int(claim_id)
        self.instance_id = int(instance_id)
        self.step = int(step)
        self.observation_ids = tuple(int(value) for value in observation_ids)

    def as_dict(self):
        return {
            "claim_id": self.claim_id,
            "instance_id": self.instance_id,
            "step": self.step,
            "observation_ids": list(self.observation_ids),
        }


class InstanceNode:
    """Canonical physical object entity exposed to planning and the VLM."""

    __slots__ = ("iid", "point", "text", "evidence", "reported",
                 "frame_id", "step", "candidate_id", "attach_node",
                 "observation_ids", "report_claim_id")

    def __init__(self, iid, point, text="", evidence=None, frame_id=None,
                 step=0, candidate_id=None, reported=False,
                 observation_ids=None):
        self.iid = int(iid)
        self.point = np.asarray(point, dtype=np.float64)
        self.text = str(text or "")[:2000]
        self.evidence = []
        self.reported = bool(reported)
        self.frame_id = frame_id
        self.step = int(step)
        self.candidate_id = candidate_id
        self.attach_node = None
        self.observation_ids = list(observation_ids or [])
        self.report_claim_id = None
        for item in evidence or []:
            self.add_evidence(item)

    def add_evidence(self, evidence):
        item = dict(evidence or {})
        if item and item not in self.evidence:
            self.evidence.append(item)
            self.evidence = self.evidence[-20:]

    def __repr__(self):
        state = "claimed" if self.reported else "available"
        return (f"InstanceNode(#{self.iid} {state} obs="
                f"{len(self.observation_ids)} pt={self.point.round(2)} "
                f"text={self.text[:40]!r})")


class InstanceMemory:
    """Episode-local observation store and canonical instance registry."""

    def __init__(self, exact_pixel_radius=8.0, exact_bbox_iou=0.9):
        self.nodes = []
        self.observations = {}
        self.report_claims = []
        self._next_id = 1
        self._next_observation_id = 1
        self._next_claim_id = 1
        self._candidate_to_observation = {}
        self._observation_to_instance = {}
        self.exact_pixel_radius = float(exact_pixel_radius)
        self.exact_bbox_iou = float(exact_bbox_iou)

    def get(self, instance_id):
        try:
            iid = int(instance_id)
        except (TypeError, ValueError):
            return None
        return next((node for node in self.nodes if node.iid == iid), None)

    def get_observation(self, observation_id):
        try:
            return self.observations.get(int(observation_id))
        except (TypeError, ValueError):
            return None

    def instance_for_observation(self, observation_id):
        iid = self._observation_to_instance.get(int(observation_id))
        return self.get(iid) if iid is not None else None

    def find_replay(self, candidate_id=None, frame_id=None, pixel=None,
                    bbox=None):
        """Return an entity for an already-seen observation, if any.

        Candidate equality is request idempotency only. Same-frame pixel/bbox
        checks catch repeated instantiation calls that received fresh cN handles.
        They are deliberately not used across frames.
        """
        observation = self._find_replay_observation(
            candidate_id=candidate_id, frame_id=frame_id,
            pixel=pixel, bbox=bbox)
        return (self.instance_for_observation(observation.oid)
                if observation is not None else None)

    def _find_replay_observation(self, candidate_id=None, frame_id=None,
                                 pixel=None, bbox=None):
        """Return the exact prior observation behind an idempotent replay."""
        if candidate_id is not None:
            oid = self._candidate_to_observation.get(str(candidate_id))
            if oid is not None:
                return self.get_observation(oid)
        pixel = _pair(pixel)
        bbox = _quad(bbox)
        if frame_id is None or (pixel is None and bbox is None):
            return None
        for observation in self.observations.values():
            if observation.frame_id != frame_id:
                continue
            same_pixel = pixel is not None and observation.pixel is not None \
                and math.hypot(pixel[0] - observation.pixel[0],
                               pixel[1] - observation.pixel[1]) \
                <= self.exact_pixel_radius
            same_bbox = bbox is not None and observation.bbox is not None \
                and _bbox_iou(bbox, observation.bbox) >= self.exact_bbox_iou
            if same_pixel or same_bbox:
                return observation
        return None

    def register_replay(self, node, candidate_id=None, evidence=None,
                        point=None, step=0):
        """Attach a replayed evidence handle without creating an observation."""
        if node is None:
            return None
        evidence = dict(evidence or {})
        matched = self._find_replay_observation(
            candidate_id=candidate_id, frame_id=evidence.get("frame_id"),
            pixel=evidence.get("pixel"), bbox=evidence.get("bbox"))
        if matched is None and node.observation_ids:
            matched = self.get_observation(node.observation_ids[-1])
        if candidate_id is not None and matched is not None:
            self._candidate_to_observation[str(candidate_id)] = matched.oid
        node.add_evidence(evidence)
        node.step = max(node.step, int(step))
        if matched is not None and point is not None:
            self.refresh_observation_point(matched, point)
        return node

    def new_observation(self, point, text="", evidence=None, frame_id=None,
                        step=0, candidate_id=None, pixel=None, bbox=None):
        observation = ObservationRecord(
            self._next_observation_id, point, text=text, evidence=evidence,
            frame_id=frame_id, step=step, candidate_id=candidate_id,
            pixel=pixel, bbox=bbox)
        self._next_observation_id += 1
        self.observations[observation.oid] = observation
        if candidate_id is not None:
            self._candidate_to_observation[str(candidate_id)] = observation.oid
        return observation

    def create_instance(self, observation, text=""):
        node = InstanceNode(
            self._next_id, observation.point,
            text=text or observation.text,
            evidence=[observation.evidence], frame_id=observation.frame_id,
            step=observation.step, candidate_id=observation.candidate_id,
            observation_ids=[observation.oid])
        self._next_id += 1
        self.nodes.append(node)
        self._observation_to_instance[observation.oid] = node.iid
        return node

    def attach_observation(self, node, observation, text=""):
        if node is None or observation is None:
            return None
        if observation.oid not in node.observation_ids:
            node.observation_ids.append(observation.oid)
        self._observation_to_instance[observation.oid] = node.iid
        node.add_evidence(observation.evidence)
        if text:
            node.text = str(text)[:2000]
        self._select_canonical_observation(node)
        return node

    def _select_canonical_observation(self, node):
        observations = [self.get_observation(oid)
                        for oid in node.observation_ids]
        observations = [obs for obs in observations if obs is not None]
        if not observations:
            return

        def quality(obs):
            score = float(obs.evidence.get("point_score", 0.0) or 0.0)
            depth_std = obs.evidence.get("depth_std")
            penalty = min(float(depth_std), 1.0) if depth_std is not None else 0.0
            return score - penalty, obs.step, obs.oid

        best = max(observations, key=quality)
        node.point = np.asarray(best.point, dtype=np.float64)
        node.frame_id = best.frame_id
        node.candidate_id = best.candidate_id
        node.step = max(node.step, max(obs.step for obs in observations))

    def add(self, point, text="", evidence=None, frame_id=None, step=0,
            candidate_id=None, pixel=None, bbox=None):
        rows = list(evidence or [])
        observation = self.new_observation(
            point, text=text, evidence=(rows[0] if rows else {}),
            frame_id=frame_id, step=step, candidate_id=candidate_id,
            pixel=pixel, bbox=bbox)
        return self.create_instance(observation, text=text)

    def remember(self, point, text="", evidence=None, frame_id=None, step=0,
                 candidate_id=None, pixel=None, bbox=None):
        """Compatibility entry point with observation-level idempotency only.

        Production ingestion uses EntityResolver for cross-frame identity.
        """
        rows = list(evidence or [])
        first = dict(rows[0]) if rows else {}
        pixel = pixel if pixel is not None else first.get("pixel")
        bbox = bbox if bbox is not None else first.get("bbox")
        existing = self.find_replay(candidate_id, frame_id, pixel, bbox)
        if existing is not None:
            for item in rows:
                existing.add_evidence(item)
            self.register_replay(existing, candidate_id, first, point, step)
            return existing, False
        observation = self.new_observation(
            point, text=text, evidence=first, frame_id=frame_id, step=step,
            candidate_id=candidate_id, pixel=pixel, bbox=bbox)
        return self.create_instance(observation, text=text), True

    def nearby(self, point, scale, radius_m, top_k=3):
        point = np.asarray(point, dtype=np.float64)
        scale = float(scale or 1.0)
        rows = []
        for node in self.nodes:
            distance_m = float(np.linalg.norm(node.point[:2] - point[:2])) * scale
            if distance_m <= float(radius_m):
                rows.append((distance_m, node))
        rows.sort(key=lambda item: (item[0], item[1].iid))
        return rows[:max(1, int(top_k))]

    def update_text(self, instance_id, text):
        node = self.get(instance_id)
        if node is not None:
            node.text = str(text or "")[:2000]
        return node

    def add_evidence(self, instance_id, evidence):
        node = self.get(instance_id)
        if node is not None:
            node.add_evidence(evidence)
        return node

    def refresh_point(self, node, point):
        if node is not None:
            node.point = np.asarray(point, dtype=np.float64)

    def refresh_observation_point(self, observation, point):
        """Refresh loop-closure geometry without changing observation identity."""
        if observation is None:
            return
        observation.point = np.asarray(point, dtype=np.float64)
        node = self.instance_for_observation(observation.oid)
        if node is not None:
            self._select_canonical_observation(node)

    def claim(self, node, step=0, observation_ids=None):
        """Atomically claim one canonical entity; returns None on duplicates."""
        if node is None or node.reported:
            return None
        ids = list(observation_ids or node.observation_ids)
        claim = ReportClaim(self._next_claim_id, node.iid, step, ids)
        self._next_claim_id += 1
        self.report_claims.append(claim)
        node.reported = True
        node.report_claim_id = claim.claim_id
        return claim

    def mark_reported(self, node):
        return self.claim(node) is not None

    def available(self):
        return [node for node in self.nodes if not node.reported]

    def count_reported(self):
        return sum(1 for node in self.nodes if node.reported)

    def attach_to_skeleton(self, graph):
        if graph is None or not graph.nodes:
            return
        worlds = np.array([node["world"] for node in graph.nodes])
        for instance in self.nodes:
            distance = np.linalg.norm(worlds - instance.point[:2], axis=1)
            instance.attach_node = graph.nodes[int(np.argmin(distance))]["id"]


def _pair(value):
    try:
        values = list(value)
        if len(values) >= 2:
            return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        pass
    return None


def _quad(value):
    try:
        values = list(value)
        if len(values) >= 4:
            return tuple(float(item) for item in values[:4])
    except (TypeError, ValueError):
        pass
    return None


def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * \
        max(0.0, min(ay1, by1) - max(ay0, by0))
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0
