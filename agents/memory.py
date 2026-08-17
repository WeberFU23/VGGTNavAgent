"""3D 实例记忆：VLM 可读写文本，确定性模块维护几何与证据。

每个具有有效 3D 点的感知结果都是 instance。记忆不区分 belief、
confirmed 或 rejected，也不维护语义黑名单。VLM 通过工具读取证据、更新
实例文本并决定是否导航、合并或报告；代码只保证 ID、坐标和证据引用真实。
"""

import numpy as np


class InstanceNode:
    __slots__ = ("iid", "point", "text", "evidence", "reported",
                 "frame_id", "step", "candidate_id", "attach_node")

    def __init__(self, iid, point, text="", evidence=None, frame_id=None,
                 step=0, candidate_id=None, reported=False):
        self.iid = int(iid)
        self.point = np.asarray(point, dtype=np.float64)
        self.text = str(text or "")[:2000]
        self.evidence = []
        self.reported = bool(reported)
        self.frame_id = frame_id
        self.step = int(step)
        self.candidate_id = candidate_id
        self.attach_node = None
        for item in evidence or []:
            self.add_evidence(item)

    def add_evidence(self, evidence):
        item = dict(evidence or {})
        if not item:
            return
        if item not in self.evidence:
            self.evidence.append(item)
            self.evidence = self.evidence[-20:]

    def __repr__(self):
        state = "reported" if self.reported else "available"
        return (f"InstanceNode(#{self.iid} {state} "
                f"pt={self.point.round(2)} text={self.text[:40]!r})")


class InstanceMemory:
    """Episode 内唯一的语义空间记忆。"""

    def __init__(self):
        self.nodes = []
        self._next_id = 1
        # merge 快照历史：keeper 合并前状态 + 被删除节点的完整记录，
        # 供 undo_merge 恢复误合并。只保留最近 50 次。
        self._merge_history = []

    def get(self, instance_id):
        try:
            iid = int(instance_id)
        except (TypeError, ValueError):
            return None
        return next((node for node in self.nodes if node.iid == iid), None)

    def _by_candidate(self, candidate_id):
        if candidate_id is None:
            return None
        return next((node for node in self.nodes
                     if node.candidate_id == candidate_id), None)

    def add(self, point, text="", evidence=None, frame_id=None, step=0,
            candidate_id=None):
        node = InstanceNode(
            self._next_id, point, text=text, evidence=evidence,
            frame_id=frame_id, step=step, candidate_id=candidate_id)
        self._next_id += 1
        self.nodes.append(node)
        return node

    def remember(self, point, text="", evidence=None, frame_id=None, step=0,
                 candidate_id=None):
        """登记 pointing 产生的实例。

        只按 mapping server 的稳定 candidate_id 更新同一记录，不使用距离、
        类别或置信度规则自动合并。跨视角重复实例由 VLM 调 merge_instances。
        返回 (node, is_new)。
        """
        point = np.asarray(point, dtype=np.float64)
        existing = self._by_candidate(candidate_id)
        if existing is None:
            node = self.add(
                point, text=text, evidence=evidence, frame_id=frame_id,
                step=step, candidate_id=candidate_id)
            return node, True
        existing.point = point
        existing.frame_id = frame_id if frame_id is not None else existing.frame_id
        existing.step = max(existing.step, int(step))
        if text and not existing.text:
            existing.text = str(text)[:2000]
        for item in evidence or []:
            existing.add_evidence(item)
        return existing, False

    def update_text(self, instance_id, text):
        node = self.get(instance_id)
        if node is None:
            return None
        node.text = str(text or "")[:2000]
        return node

    def add_evidence(self, instance_id, evidence):
        node = self.get(instance_id)
        if node is None:
            return None
        node.add_evidence(evidence)
        return node

    @staticmethod
    def _snapshot(node):
        """节点完整状态快照（undo_merge 用）。"""
        return {
            "iid": node.iid,
            "point": node.point.copy(),
            "text": node.text,
            "evidence": [dict(item) for item in node.evidence],
            "reported": node.reported,
            "frame_id": node.frame_id,
            "step": node.step,
            "candidate_id": node.candidate_id,
        }

    def merge(self, instance_ids, text=""):
        """按 VLM 明确指令合并实例，保留最小 ID 作为稳定 ID。

        合并前快照所有参与节点到 _merge_history；误合并可用
        undo_merge 恢复。"""
        ids = []
        for value in instance_ids or []:
            try:
                iid = int(value)
            except (TypeError, ValueError):
                continue
            if iid not in ids:
                ids.append(iid)
        nodes = [self.get(iid) for iid in ids]
        nodes = [node for node in nodes if node is not None]
        if len(nodes) < 2:
            return None
        nodes.sort(key=lambda node: node.iid)
        keep = nodes[0]
        self._merge_history.append({
            "keep_id": keep.iid,
            "keep": self._snapshot(keep),
            "removed": [self._snapshot(node) for node in nodes[1:]],
        })
        self._merge_history = self._merge_history[-50:]
        keep.point = np.median(
            np.stack([node.point for node in nodes]), axis=0)
        keep.reported = any(node.reported for node in nodes)
        keep.step = max(node.step for node in nodes)
        latest = max(nodes, key=lambda node: node.step)
        keep.frame_id = latest.frame_id
        keep.candidate_id = latest.candidate_id
        for node in nodes:
            for item in node.evidence:
                keep.add_evidence(item)
        if text:
            keep.text = str(text)[:2000]
        self.nodes = [node for node in self.nodes
                      if node is keep or node not in nodes]
        return keep

    def undo_merge(self):
        """撤销最近一次 merge：恢复 keeper 合并前状态并重建被删除节点。

        reported 只增不减——撤销不会吞掉合并后已发生的报告。
        返回 {"keep_id", "restored_ids"}；无可撤销记录返回 None。"""
        while self._merge_history:
            record = self._merge_history.pop()
            keep = self.get(record["keep_id"])
            if keep is None:
                continue            # keeper 已不存在，尝试更早的记录
            snap = record["keep"]
            keep.point = snap["point"]
            keep.text = snap["text"]
            keep.evidence = [dict(item) for item in snap["evidence"]]
            keep.reported = keep.reported or snap["reported"]
            keep.frame_id = snap["frame_id"]
            keep.step = snap["step"]
            keep.candidate_id = snap["candidate_id"]
            restored = []
            for snap in record["removed"]:
                if self.get(snap["iid"]) is not None:
                    continue
                node = InstanceNode(
                    snap["iid"], snap["point"], text=snap["text"],
                    evidence=snap["evidence"], frame_id=snap["frame_id"],
                    step=snap["step"], candidate_id=snap["candidate_id"],
                    reported=snap["reported"])
                self.nodes.append(node)
                restored.append(node)
            self.nodes.sort(key=lambda node: node.iid)
            return {"keep_id": keep.iid,
                    "restored_ids": [node.iid for node in restored]}
        return None

    def refresh_point(self, node, point):
        node.point = np.asarray(point, dtype=np.float64)

    def mark_reported(self, node):
        if node is None or node.reported:
            return False
        node.reported = True
        return True

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
