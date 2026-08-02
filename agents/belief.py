"""空间信念：目标文本与关键帧的 CLIP 相似度锚定到地图坐标。

server 的 query_text 返回 top-K 关键帧（score + 位姿），把每帧分数
记到关键帧的对齐地图坐标——零额外 VLM 开销且不依赖临时节点编号。
探索时 frontier 按 大小/(1+距离) × (1+权重×信念) 打分，
"目标更可能出现在哪类区域"的先验由此进入探索排序。

VLM 批量看图打分（contact-sheet）留作后续增强接口；当前版本
只依赖已有 CLIP 通道，不增加任何调用成本。
"""

import numpy as np


class BeliefMap:
    def __init__(self, max_keyframes=10, query_interval=40):
        self.max_keyframes = int(max_keyframes)
        self.query_interval = int(query_interval)
        # frame_id -> (aligned world xy, score)。骨架每次重建时 node_id
        # 都会重新编号，不能作为跨重建的持久键。
        self.anchors = {}
        self._last_query_step = -10 ** 9

    def reset(self):
        self.anchors = {}
        self._last_query_step = -10 ** 9

    def update(self, client, graph, target_text, align_R, step):
        """按间隔从 server 拉 CLIP 检索结果并更新空间信念。"""
        if not target_text or graph is None or not graph.nodes:
            return
        if align_R is None:
            return
        if step - self._last_query_step < self.query_interval:
            return
        self._last_query_step = step
        try:
            results = client.query_text(
                target_text, top_k=self.max_keyframes)
        except Exception:
            return
        if not results:
            return
        for r in results:
            raw_pose = r.get("pose")
            pose = np.asarray(raw_pose if raw_pose is not None
                              else np.zeros((4, 4)), dtype=np.float64)
            if pose.shape != (4, 4):
                continue
            c = pose[:3, 3] @ align_R.T
            s = float(r.get("score", 0.0))
            fid = int(r.get("frame_id", -1))
            key = fid if fid >= 0 else (r.get("submap_id"),
                                        r.get("frame_index"))
            old = self.anchors.get(key)
            if old is None or s >= old[1]:
                self.anchors[key] = (np.asarray(c[:2], dtype=np.float64), s)

    def belief_at(self, world_xy, graph):
        """返回某 world xy（如 frontier 质心）的距离衰减信念。"""
        if not self.anchors:
            return 0.0
        xy = np.asarray(world_xy, dtype=np.float64)[:2]
        # 空间锚定并随距离衰减，避免旧 keyframe 给整张新骨架同一 node id
        # 的位置错误继承高分。
        return max(float(score) / (1.0 + np.linalg.norm(pos - xy))
                   for pos, score in self.anchors.values())
