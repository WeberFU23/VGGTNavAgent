"""多目标实例记忆：语义定位到的物体实例持久化为节点。

为 many/all 多目标任务服务：
- confirmed（扫描确认，未访问）-> visited（已宣布 TARGET_FOUND）；
- rejected（确认失败/被 VLM 否决）持久拉黑，避免重复确认同一误检；
- 同类别、地图距离 < merge_dist 的命中合并为同一实例
  （"找到第一个包后继续找下一个不同的包"靠这个去重）。

记忆本身是单位无关的：所有距离阈值由调用方按当前尺度换算后传入。
"""

import numpy as np


class InstanceNode:
    __slots__ = ("iid", "category", "point", "score", "status",
                 "frame_id", "step", "candidate_id", "attach_node", "n_obs")

    def __init__(self, iid, category, point, score, status,
                 frame_id=None, step=0, candidate_id=None):
        self.iid = iid
        self.category = category
        self.point = np.asarray(point, dtype=np.float64)  # (3,) 地图坐标
        self.score = float(score)
        self.status = status              # confirmed / visited / rejected
        self.frame_id = frame_id
        self.step = int(step)
        self.candidate_id = candidate_id
        self.attach_node = None           # 骨架节点 id（attach_to_skeleton）
        self.n_obs = 1                    # 独立观测帧数（分级置信度）

    def __repr__(self):
        return (f"InstanceNode(#{self.iid} {self.category} "
                f"{self.status} score={self.score:.2f} "
                f"pt={self.point.round(2)})")


class InstanceMemory:
    def __init__(self):
        self.nodes = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def _find(self, category, point, dist, candidate_id=None):
        if candidate_id is not None:
            for nd in self.nodes:
                if nd.category == category and nd.candidate_id == candidate_id:
                    return nd
        best, best_d = None, dist
        for nd in self.nodes:
            if nd.category != category:
                continue
            d = float(np.linalg.norm(nd.point[:2] - point[:2]))
            if d < best_d:
                best, best_d = nd, d
        return best

    def add_or_merge(self, category, point, score, merge_dist,
                     status="confirmed", frame_id=None, step=0,
                     candidate_id=None):
        """同类别近距离实例合并（保留高置信的位置与分数）。
        返回 (node, is_new)。已 visited/rejected 的实例不降级。"""
        point = np.asarray(point, dtype=np.float64)
        existing = self._find(category, point, merge_dist, candidate_id)
        if existing is not None:
            if score > existing.score and existing.status == "confirmed":
                existing.point = point
                existing.score = float(score)
                existing.frame_id = frame_id
                existing.candidate_id = candidate_id
            return existing, False
        node = InstanceNode(self._next_id, category, point, score, status,
                            frame_id=frame_id, step=step,
                            candidate_id=candidate_id)
        self._next_id += 1
        self.nodes.append(node)
        return node, True

    def refresh_point(self, node, point):
        """图优化后刷新实例的统一导航坐标，不改变语义状态。"""
        node.point = np.asarray(point, dtype=np.float64)

    def mark_visited(self, node):
        node.status = "visited"

    def mark_rejected(self, node):
        if node.status != "visited":
            node.status = "rejected"

    # ------------------------------------------------------------------
    def is_rejected(self, category, point, dist):
        nd = self._find(category, point, dist)
        return nd is not None and nd.status == "rejected"

    def is_visited(self, category, point, dist):
        nd = self._find(category, point, dist)
        return nd is not None and nd.status == "visited"

    def unvisited(self, category=None):
        """confirmed 且未访问的实例（planner 的候选）。"""
        return [nd for nd in self.nodes
                if nd.status == "confirmed"
                and (category is None or nd.category == category)]

    def count_visited(self, category=None):
        return sum(1 for nd in self.nodes if nd.status == "visited"
                   and (category is None or nd.category == category))

    def count_confirmed(self, category=None):
        return sum(1 for nd in self.nodes if nd.status in
                   ("confirmed", "visited")
                   and (category is None or nd.category == category))

    # ------------------------------------------------------------------
    def attach_to_skeleton(self, graph):
        """把每个实例吸附到最近骨架节点（world 距离）。"""
        if graph is None or not graph.nodes:
            return
        worlds = np.array([nd["world"] for nd in graph.nodes])
        for nd in self.nodes:
            d = np.linalg.norm(worlds - nd.point[:2], axis=1)
            nd.attach_node = graph.nodes[int(np.argmin(d))]["id"]
