"""分级置信度观测账本（semantic_memory 后端，agent 端）。

替代旧 NAV_MIN_SAM 单阈值逻辑：
- 单帧 pointing 观测 -> belief 锚点：不登记发现，只做 frontier 打分先验；
- >= min_obs 个不同位姿的独立帧观测（同一位置簇）-> confirmed，
  写入 InstanceMemory 进入 TSP（实例合并逻辑不动，只改置信度来源）；
- 小/远目标（pointing 置信低、patch 深度方差大、目标像素占比小）由
  调用方以 force_belief=True 强制留 belief，逼近后复核。

计数信几何不信文本：确认数量以这里几何独立的簇为准，VLM 报的实例数
只当提示。
"""

import numpy as np


class BeliefAnchor:
    __slots__ = ("category", "point", "score", "observations", "step")

    def __init__(self, category, point, score, step):
        self.category = category
        self.point = np.asarray(point, dtype=np.float64)  # (3,) 对齐地图坐标
        self.score = float(score)
        # [(obs_xy (2,) 或 None, frame_id, step)] —— 每次独立观测
        self.observations = []
        self.step = int(step)

    @property
    def n_obs(self):
        return len(self.observations)

    def __repr__(self):
        return (f"BeliefAnchor({self.category} n_obs={self.n_obs} "
                f"score={self.score:.2f} pt={self.point.round(2)})")


class ObservationLedger:
    def __init__(self, min_obs=2, min_pose_sep=0.0):
        self.min_obs = int(min_obs)
        # 两次观测的拍照位姿间距 >= 此值（地图单位）才算"独立帧"
        self.min_pose_sep = float(min_pose_sep)
        self.anchors = []

    # ------------------------------------------------------------------
    def _find(self, category, point, dist):
        best, best_d = None, dist
        for a in self.anchors:
            if a.category != category:
                continue
            d = float(np.linalg.norm(a.point[:2] - point[:2]))
            if d < best_d:
                best, best_d = a, d
        return best

    def add_observation(self, category, point, score, merge_dist,
                        frame_id=None, step=0, obs_xy=None,
                        force_belief=False):
        """登记一次观测。返回 (outcome, anchor)：
        "belief"    —— 仍是单帧/低置信锚点；
        "confirmed" —— 独立观测数达标，可写入实例记忆；
        "duplicate" —— 同一位姿的重复观测，计数不变。
        """
        point = np.asarray(point, dtype=np.float64)
        anchor = self._find(category, point, merge_dist)
        is_independent = True
        if anchor is None:
            anchor = BeliefAnchor(category, point, score, step)
            self.anchors.append(anchor)
        else:
            if obs_xy is not None:
                for prev_xy, _fid, _st in anchor.observations:
                    if prev_xy is None:
                        continue
                    if np.linalg.norm(np.asarray(obs_xy)[:2] - prev_xy) \
                            < self.min_pose_sep:
                        is_independent = False
                        break
            if score > anchor.score:
                anchor.point = point
                anchor.score = float(score)
        if is_independent:
            anchor.observations.append((
                None if obs_xy is None else np.asarray(obs_xy, dtype=float)[:2],
                frame_id, int(step)))
            anchor.step = int(step)
        else:
            return "duplicate", anchor
        if not force_belief and anchor.n_obs >= self.min_obs:
            return "confirmed", anchor
        return "belief", anchor

    # ------------------------------------------------------------------
    def discard(self, anchor):
        if anchor in self.anchors:
            self.anchors.remove(anchor)

    def discard_near(self, category, point, dist):
        """位置被确认/访问/拉黑后，清掉对应 belief 锚点。"""
        point = np.asarray(point, dtype=np.float64)
        for a in list(self.anchors):
            if a.category == category and \
                    np.linalg.norm(a.point[:2] - point[:2]) < dist:
                self.anchors.remove(a)

    def belief_anchors(self, category=None):
        """frontier 打分先验用的锚点列表。"""
        return [a for a in self.anchors
                if category is None or a.category == category]

    def count_unresolved(self, category=None, min_score=0.0):
        """未复核的高置信锚点数（终止账本用）。"""
        return sum(1 for a in self.anchors
                   if a.score >= min_score
                   and (category is None or a.category == category))
