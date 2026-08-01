"""导航 agent：随机探索 -> FOUND-IT 式实例定位 -> 栅格规划 -> 路径跟随。

流程（多目标状态机）：
1. EXPLORE：沿用 MappingAgent 的随机探索 + 喂图建图；每 NAV_QUERY_INTERVAL
   步向 server 发 ground_object(目标短语)，命中（SAM 分数达标）则取出
   3D 目标点。
2. 拿到目标点后用点云构建 2D 占据栅格（agents/navigator.py），A* 规划，
   进入 NAV 模式沿路径输出离散动作。位姿锚定最新关键帧 + 航位推算，
   定期重建栅格并重规划（地图随探索增长，回环也会改写历史位姿）。
3. 距目标点 < NAV_REACH_M 米时返回 TARGET_FOUND（评测阈值 1.0m，留裕量）。

目标短语只来自公开 instruction（或显式 NAV_TARGET 调试覆盖），不读取
query_program、GPS、深度或仿真器姿态。

运行方式与 MappingAgent 相同：
    --agent agents.nav_agent:NavAgent
"""

import math
import os
import re

import numpy as np

from benchmark_api import Action
from agents import navigator as nav
from agents.mapping_agent import MappingAgent
from decision import TargetSpec, VLMDecisionClient


class NavAgent(MappingAgent):
    def __init__(self):
        super().__init__()
        self.query_interval = int(os.environ.get("NAV_QUERY_INTERVAL", "20"))
        self.replan_interval = int(os.environ.get("NAV_REPLAN_INTERVAL", "20"))
        self.warmup_steps = int(os.environ.get("NAV_WARMUP_STEPS", "40"))
        self.min_sam = float(os.environ.get("NAV_MIN_SAM", "0.5"))
        self.verify_min = float(os.environ.get("NAV_VERIFY_MIN", "0.25"))
        self.reach_m = float(os.environ.get("NAV_REACH_M", "0.8"))
        self.finish_patience = int(os.environ.get("NAV_FINISH_PATIENCE", "5"))
        self.ground_top_k = int(os.environ.get("NAV_GROUND_TOP_K", "3"))
        self.vlm_candidate_limit = int(
            os.environ.get("NAV_VLM_CANDIDATE_LIMIT", "4"))
        self.vlm_candidate_conf = float(
            os.environ.get("NAV_VLM_CANDIDATE_CONF", "0.35"))
        self.vlm_verify_conf = float(
            os.environ.get("NAV_VLM_VERIFY_CONF", "0.50"))
        self.vlm_finish_conf = float(
            os.environ.get("NAV_VLM_FINISH_CONF", "0.60"))
        self.vlm = VLMDecisionClient.from_env()
        print(f"[NavAgent] VLM 战略层: "
              f"{'enabled (' + self.vlm.model + ')' if self.vlm.enabled else 'disabled'}")
        self._nav_reset_state()

    def _nav_reset_state(self):
        self.mode = "explore"           # explore / nav / reported
        self.target_text = None
        self.target_point = None        # 地图坐标（未缩放单位），(3,)
        self.target_candidate_id = None
        self.follower = None
        self.grid = None
        self.align_R = None
        self._last_query_step = -10 ** 9
        self._last_plan_step = -10 ** 9
        self._last_anchor_step = -10 ** 9
        self._plan_failures = 0
        self._verify_failures = 0
        self._scanning = False          # 到达后原地 360° 扫描确认中
        self._scan_steps = 0
        self._bad_points = []           # 扫描确认失败的目标点（地图坐标）
        self._found_points = []
        self._reported_count = 0
        self._no_hit_queries = 0
        self._target_mode = "any"
        self._target_count = None
        self.target_spec = None
        self._target_spec_source = None
        self._selected_evidence = None
        self._explore_hint = "none"
        self._explore_hint_steps = 0
        self._last_finish_vlm_step = -10 ** 9

    def reset(self):
        super().reset()
        self._nav_reset_state()

    # ------------------------------------------------------------------
    def _target_phrase(self, observation):
        override = os.environ.get("NAV_TARGET")
        if override:
            source = str(observation.goal_text or "").strip()
            if self.target_spec is None or self._target_spec_source != source:
                self.target_spec = TargetSpec(
                    grounding_query=override,
                    target_description=source or override,
                    confidence=1.0)
                self._target_spec_source = source
            return override
        source = str(observation.goal_text or "").strip()
        if self.target_spec is not None and self._target_spec_source == source:
            return self.target_spec.grounding_query
        # 只从自然语言本身提取 grounding phrase，不读取 query_program。
        # 保留颜色/材质等描述属性，仅去掉导航命令和冠词。
        text = source
        text = re.sub(
            r"^(?:please\s+)?(?:find|locate|look\s+for|navigate\s+to|go\s+to)\s+",
            "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:any|a|an|the)\s+", "", text,
                      flags=re.IGNORECASE)
        fallback = text.strip().rstrip(".!?") or source
        spec = None
        if self.vlm.enabled:
            spec = self.vlm.parse_instruction(
                source, self._target_mode, self._target_count)
        self.target_spec = spec or TargetSpec(
            grounding_query=fallback,
            target_description=source or fallback,
            confidence=0.0)
        self._target_spec_source = source
        if spec is not None:
            print(f"[NavAgent] VLM 解析目标: '{source}' -> "
                  f"'{spec.grounding_query}' ({spec.confidence:.2f})")
        return self.target_spec.grounding_query

    def _explore_action(self, observation):
        """执行 VLM 给出的短宏提示；碰撞恢复始终优先。"""
        if self._last_motion_failed:
            self._explore_hint_steps = 0
            return super()._explore_action(observation)
        if self._explore_hint_steps > 0:
            action_by_hint = {
                "forward": Action.MOVE_FORWARD,
                "turn_left": Action.TURN_LEFT,
                "turn_right": Action.TURN_RIGHT,
                "scan": Action.TURN_LEFT,
            }
            action = action_by_hint.get(self._explore_hint)
            if action is not None:
                self._explore_hint_steps -= 1
                return int(action)
        return super()._explore_action(observation)

    def _set_explore_hint(self, hint):
        self._explore_hint = str(hint or "none")
        self._explore_hint_steps = {
            "forward": 3, "turn_left": 1, "turn_right": 1, "scan": 3,
        }.get(self._explore_hint, 0)

    def act(self, observation):
        self._feed_frame(observation)
        self._target_mode = str(observation.target_mode or "any").lower()
        self._target_count = observation.target_count
        if self.follower is not None and self._last_motion_failed and \
                observation.previous_action == int(Action.MOVE_FORWARD):
            self.follower.undo_dead_reckon(int(Action.MOVE_FORWARD))

        if self.mode == "explore" and self._should_finish(observation):
            action = int(Action.FINISH)
        elif self.mode == "reported":
            if self._should_finish(observation):
                action = int(Action.FINISH)
            else:
                self._clear_current_target()
                self.mode = "explore"
                action = self._explore_action(observation)
        elif self.mode == "done":
            action = self._explore_action(observation)
        elif self.mode == "nav":
            if self._scanning:
                action = self._handle_scan(observation)
            else:
                action, arrived = self._nav_action(observation)
                if arrived:
                    arrival = self._arrival_decision(observation)
                    if arrival == "report_found":
                        print(f"[NavAgent] step={observation.step_count} "
                              f"到达目标点附近且视觉确认通过，TARGET_FOUND "
                              f"(目标='{self.target_text}')")
                        action = self._report_found()
                    elif arrival == "reject":
                        action = self._reject_current_target(observation)
                    else:
                        # 距离到了但当前朝向看不到目标：原地 360° 扫描确认
                        self._scanning = True
                        self._scan_steps = 0
                        action = int(Action.TURN_LEFT)
                elif action is None:    # 路径走丢，退回探索
                    self.mode = "explore"
                    action = self._explore_action(observation)
        else:
            action = self._explore_action(observation)
            self._periodic_anchor(observation)
            self._maybe_query_target(observation)

        self._record_and_update(observation, action)
        if self.follower is not None and self.follower.anchor_frame >= 0:
            self.follower.dead_reckon(action)
        return action

    def _clear_current_target(self):
        self.target_point = None
        self.target_candidate_id = None
        self.follower = None
        self.grid = None
        self.align_R = None
        self._plan_failures = 0
        self._scanning = False
        self._selected_evidence = None

    def _report_found(self):
        if self.target_point is not None:
            self._found_points.append(self.target_point.copy())
        self._reported_count += 1
        self._no_hit_queries = 0
        self.mode = "reported"
        self._scanning = False
        return int(Action.TARGET_FOUND)

    def _reject_current_target(self, observation):
        if self.target_point is not None:
            self._bad_points.append(self.target_point.copy())
        print(f"[NavAgent] step={observation.step_count} VLM 拒绝当前候选")
        self._clear_current_target()
        self.mode = "explore"
        return self._explore_action(observation)

    def _decision_state(self, observation):
        mapping_state = {}
        try:
            raw = self.client.get_state()
            mapping_state = {
                "num_submaps": raw.get("num_submaps", 0),
                "queued_keyframes": raw.get("queued_keyframes", 0),
                "mapping_busy": bool(raw.get("busy")),
            }
        except Exception:
            pass
        return {
            "step": int(observation.step_count),
            "max_steps": int(observation.max_steps),
            "target_mode": self._target_mode,
            "required_count": self._target_count,
            "reported_instance_count": self._reported_count,
            "recent_queries_without_new_candidate": self._no_hit_queries,
            "rejected_or_duplicate_location_count": len(self._bad_points),
            "has_agent_estimated_metric_scale":
                self.calibrator.current_scale() is not None,
            **mapping_state,
        }

    def _should_finish(self, observation):
        if self._target_mode == "many" and self._target_count is not None:
            return self._reported_count >= int(self._target_count)
        if self._target_mode == "all":
            late = observation.step_count >= int(0.8 * observation.max_steps)
            ready = self._reported_count > 0 and late and \
                self._no_hit_queries >= self.finish_patience
            if not ready or not self.vlm.enabled:
                return ready
            if observation.step_count - self._last_finish_vlm_step < \
                    self.query_interval:
                return False
            self._last_finish_vlm_step = observation.step_count
            self._target_phrase(observation)
            decision = self.vlm.decide_finish(
                observation.goal_text, self.target_spec,
                self._decision_state(observation), observation.rgb)
            if decision is None:
                return ready
            self._set_explore_hint(decision.exploration_hint)
            print(f"[NavAgent] VLM FINISH 判断: {decision.decision} "
                  f"conf={decision.confidence:.2f} reason={decision.reason}")
            return decision.decision == "finish" and \
                decision.confidence >= self.vlm_finish_conf
        return False

    # ------------------------------------------------------------------
    # EXPLORE：查询目标
    # ------------------------------------------------------------------
    def _periodic_anchor(self, observation):
        """探索期间也定期刷新位姿锚点——否则锚点只在上次规划时更新，
        随机探索撞墙的航位推算误差会无界累积（实测漂数米导致假到达）。"""
        step = observation.step_count
        if step - self._last_anchor_step < 5:
            return
        if self.calibrator.current_scale() is None:
            return
        self._last_anchor_step = step
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 3:
                self._refresh_anchor(poses, frame_ids)
        except Exception:
            pass

    def _arrival_decision(self, observation):
        """SAM3 当前帧确认后，再由 VLM 核对完整语义属性。"""
        try:
            r = self.client.ground_frame(observation.rgb, self.target_text)
        except Exception as e:
            print(f"[NavAgent] ground_frame 失败: {e}")
            return "scan"
        found = bool(r.get("found")) and \
            r.get("score", 0.0) >= self.verify_min
        print(f"[NavAgent] 视觉确认: found={r.get('found')} "
              f"score={r.get('score', 0.0):.3f} -> {'通过' if found else '未过'}")
        if not found:
            return "scan"
        if not self.vlm.enabled:
            return "report_found"
        self._target_phrase(observation)
        decision = self.vlm.verify_arrival(
            observation.goal_text, self.target_spec,
            self._decision_state(observation), observation.rgb,
            self._selected_evidence)
        if decision is None:
            return "report_found"
        print(f"[NavAgent] VLM 到达复核: {decision.decision} "
              f"conf={decision.confidence:.2f} reason={decision.reason}")
        if decision.decision == "report_found" and \
                decision.confidence < self.vlm_verify_conf:
            return "scan"
        return decision.decision

    def _handle_scan(self, observation):
        """360° 原地扫描：每步先确认当前帧，未过继续转；
        转满一圈仍未确认则拉黑当前目标点、退回探索。"""
        arrival = self._arrival_decision(observation)
        if arrival == "report_found":
            print(f"[NavAgent] step={observation.step_count} 扫描第 "
                  f"{self._scan_steps + 1} 帧确认通过，TARGET_FOUND "
                  f"(目标='{self.target_text}')")
            return self._report_found()
        if arrival == "reject":
            return self._reject_current_target(observation)
        self._scan_steps += 1
        if self._scan_steps < 12:
            return int(Action.TURN_LEFT)
        self._scanning = False
        self._verify_failures += 1
        print(f"[NavAgent] step={observation.step_count} 360° 扫描未确认，"
              f"拉黑目标点（累计 {self._verify_failures} 次）")
        if self.target_point is not None:
            self._bad_points.append(self.target_point.copy())
        self._clear_current_target()
        self.mode = "explore"
        return self._explore_action(observation)

    def _is_bad_point(self, point):
        """目标点是否在黑名单附近（1.5m 内）。"""
        scale = self.calibrator.current_scale() or 1.0
        for bp in self._bad_points + self._found_points:
            if np.linalg.norm(point - bp) * scale < 1.5:
                return True
        return False

    def _maybe_query_target(self, observation):
        step = observation.step_count
        if step < self.warmup_steps or step - self._last_query_step < \
                self.query_interval:
            return
        self._last_query_step = step

        # 已有目标点：不再重复查询，按间隔重试规划（地图在增长）；
        # 连续失败过多说明目标点可能不可靠，丢弃后重新查询。
        if self.target_point is not None:
            if self._plan_failures >= 5:
                print("[NavAgent] 规划连续失败，丢弃当前目标点，重新查询")
                self.target_point = None
                self._plan_failures = 0
            elif step - self._last_plan_step >= self.replan_interval:
                if self._plan_to_target(observation):
                    self.mode = "nav"
            return

        phrase = self._target_phrase(observation)
        self.target_text = phrase
        try:
            results = self.client.ground_object(
                phrase, top_k=self.ground_top_k)
        except Exception as e:          # server 忙/异常不应杀死 episode
            print(f"[NavAgent] ground_object 失败: {e}")
            return
        hits = [r for r in results if r.get("found")]
        if not hits:
            self._no_hit_queries += 1
            self._vlm_candidate_decision(observation, [])
            print(f"[NavAgent] step={step} 未定位到 '{phrase}'")
            return
        # 按 SAM 分数排序，跳过黑名单（扫描确认失败过）的目标点
        hits.sort(key=lambda r: r.get("sam_score", 0.0), reverse=True)
        eligible = []
        for h in hits:
            pt = np.asarray(h["point"], dtype=np.float64)
            if h.get("sam_score", 0.0) >= self.min_sam and \
                    not self._is_bad_point(pt):
                eligible.append(h)
        if not eligible:
            self._no_hit_queries += 1
            self._vlm_candidate_decision(observation, [])
            print(f"[NavAgent] step={step} '{phrase}' 命中均被拉黑或分数不足")
            return
        best, evidence = self._vlm_candidate_decision(
            observation, eligible[:self.vlm_candidate_limit])
        if best is None:
            self._no_hit_queries += 1
            print(f"[NavAgent] step={step} VLM 建议继续探索")
            return
        self.target_point = np.asarray(best["point"], dtype=np.float64)
        self.target_candidate_id = best.get("candidate_id")
        self._selected_evidence = evidence
        self._no_hit_queries = 0
        print(f"[NavAgent] step={step} 定位 '{phrase}' sam="
              f"{best['sam_score']:.3f} point={self.target_point.round(2)}")
        if self._plan_to_target(observation):
            self.mode = "nav"

    def _vlm_candidate_decision(self, observation, candidates):
        """返回 (selected candidate, evidence bytes)；无 VLM 时取最高 SAM。"""
        if not self.vlm.enabled:
            return (candidates[0], None) if candidates else (None, None)
        self._target_phrase(observation)
        evidence_by_id = {}
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            if not candidate_id:
                continue
            try:
                meta, payload = self.client.get_candidate_evidence(candidate_id)
                if meta.get("found") and payload:
                    evidence_by_id[candidate_id] = payload
            except Exception as exc:
                print(f"[NavAgent] candidate evidence 失败: {exc}")
        decision = self.vlm.choose_candidate(
            observation.goal_text, self.target_spec,
            self._decision_state(observation), observation.rgb,
            candidates, evidence_by_id)
        if decision is None:
            return (candidates[0], evidence_by_id.get(
                candidates[0].get("candidate_id"))) if candidates else (None, None)
        self._set_explore_hint(decision.exploration_hint)
        by_id = {c.get("candidate_id"): c for c in candidates}
        for candidate_id in decision.rejected_candidate_ids:
            rejected = by_id.get(candidate_id)
            if rejected is not None:
                self._bad_points.append(np.asarray(
                    rejected["point"], dtype=np.float64))
        print(f"[NavAgent] VLM 候选判断: {decision.decision} "
              f"candidate={decision.candidate_id} conf={decision.confidence:.2f} "
              f"reason={decision.reason}")
        if decision.decision != "navigate" or \
                decision.confidence < self.vlm_candidate_conf or \
                decision.candidate_id not in by_id or \
                decision.candidate_id in decision.rejected_candidate_ids:
            return None, None
        selected = by_id[decision.candidate_id]
        return selected, evidence_by_id.get(decision.candidate_id)

    # ------------------------------------------------------------------
    # 规划
    # ------------------------------------------------------------------
    def _refresh_anchor(self, poses, frame_ids):
        """锚定到最新关键帧位姿，并重放锚点之后的动作做航位推算。"""
        order = np.argsort(frame_ids)
        fid = int(np.asarray(frame_ids)[order][-1])
        pose = np.asarray(poses, dtype=np.float64)[order][-1]
        if self.align_R is None:
            self.align_R = nav.gravity_alignment(
                np.asarray(poses, dtype=np.float64))
        scale = self.calibrator.current_scale()
        if self.follower is None:
            self.follower = nav.PathFollower(scale=scale, reach_m=self.reach_m)
        self.follower.scale = scale
        if fid <= self.follower.anchor_frame:
            return
        self.follower.update_anchor(pose, self.align_R, fid)
        # 位姿对应第 fid 帧（即 actions[:fid-1] 执行完）的状态，
        # 之后的动作重放外推到现在
        for a in self.calibrator.actions[fid - 1:]:
            self.follower.dead_reckon(a)

    def _plan_to_target(self, observation):
        """重建栅格并规划到目标点。成功返回 True。"""
        if self.target_point is None:
            return False
        if self.target_candidate_id:
            try:
                resolved = self.client.resolve_candidate(
                    self.target_candidate_id)
                if not resolved.get("found"):
                    print("[NavAgent] 候选在最新地图中无法重投影，重新查询")
                    self._clear_current_target()
                    return False
                self.target_point = np.asarray(
                    resolved["point"], dtype=np.float64)
            except Exception as e:
                print(f"[NavAgent] 候选重投影失败: {e}")
                return False
        scale = self.calibrator.current_scale()
        if scale is None:
            return False
        self._last_plan_step = observation.step_count
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is None or len(poses) < 5:
                return False
            self._refresh_anchor(poses, frame_ids)
        except Exception as e:
            print(f"[NavAgent] 拉取位姿失败: {e}")
            return False

        cam_centers = np.asarray(poses, dtype=np.float64)[:, :3, 3] \
            @ self.align_R.T
        # 面包屑栅格：目标必然在走过的地方附近（从历史关键帧检出），
        # 沿轨迹网络规划，不依赖脆弱的地板/障碍高度分层
        self.grid = nav.OccupancyGrid.from_trajectory(cam_centers)
        if self.grid is None:
            print("[NavAgent] 栅格构建失败（点数不足）")
            return False

        goal_xy = (self.align_R @ self.target_point)[:2]
        # 起点投影回走廊：机器人物理上一定在自己走过的轨迹附近，
        # 只有跟随器估计会被撞墙/尺度误差带偏。直接把估计位置吸附到
        # 走廊上，消除"强制自由产生的孤岛起点"导致的 A* 失败。
        # 吸附不动（估计漂出 60 格）则硬重置到最新已定位关键帧。
        sc = self.grid.nearest_traversable(
            self.grid.world_to_cell((self.follower.x, self.follower.y)), 60)
        if sc is None:
            self.follower.x, self.follower.y = \
                float(cam_centers[-1][0]), float(cam_centers[-1][1])
            print("[NavAgent] 跟随器漂出走廊，硬重置到最新关键帧位置")
        else:
            self.follower.x, self.follower.y = \
                [float(v) for v in self.grid.cell_to_world(sc)]
        start_xy = (self.follower.x, self.follower.y)
        path = self.grid.astar(start_xy, goal_xy, snap_radius=30)
        if path is None:
            self._plan_failures += 1
            sc_snap = self.grid.nearest_traversable(sc, 30) \
                if sc is not None else None
            gc_snap = self.grid.nearest_traversable(
                self.grid.world_to_cell(goal_xy), 30)
            print(f"[NavAgent] 规划失败（第 {self._plan_failures} 次） "
                  f"follower=({self.follower.x:.2f},{self.follower.y:.2f}) "
                  f"anchor_fid={self.follower.anchor_frame} "
                  f"goal=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) "
                  f"free={self.grid.free.sum()} res={self.grid.res:.3f} "
                  f"u={getattr(self.grid, 'unit_per_m', 0):.3f} "
                  f"start_snap={sc_snap} goal_snap={gc_snap} "
                  f"scale={scale:.2f}")
            return False
        path = self.grid.shortcut(path)
        self.follower.set_path(path)
        self._plan_failures = 0
        dist_m = sum(np.linalg.norm(np.asarray(path[i + 1]) - np.asarray(path[i]))
                     for i in range(len(path) - 1)) * scale
        print(f"[NavAgent] 规划成功: {len(path)} 航点, 路径长约 {dist_m:.1f}m")
        return True

    # ------------------------------------------------------------------
    # NAV：跟随路径
    # ------------------------------------------------------------------
    def _nav_action(self, observation):
        scale = self.calibrator.current_scale()
        if scale is None or self.follower is None:
            return None, False
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 3:
                self._refresh_anchor(poses, frame_ids)
        except Exception:
            pass

        # 到达判定：到原始目标点的水平距离（评测阈值 1.0m，默认 0.8m 留裕量）
        goal_xy = (self.align_R @ self.target_point)[:2]
        dist_m = math.hypot(goal_xy[0] - self.follower.x,
                            goal_xy[1] - self.follower.y) * scale
        if dist_m < self.reach_m:
            return None, True

        # 定期重规划（地图增长 / 回环改写位姿 / 卡死兜底）
        if observation.step_count - self._last_plan_step >= self.replan_interval:
            if not self._plan_to_target(observation):
                return None, False      # 地图反倒变差，退回探索

        action, arrived = self.follower.next_action()
        if arrived:
            return None, True
        if action is None:
            return None, False
        return int(action), False
