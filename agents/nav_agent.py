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
from agents import planner
from agents import skeleton as skel
from agents.belief import BeliefMap
from agents.evidence import ObservationLedger
from agents.mapping_agent import MappingAgent
from agents.memory import InstanceMemory
from decision import (DecisionLoop, DecisionTraceLogger, TargetSpec,
                      VLMDecisionClient)


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
        self.finish_frontier_patience = int(os.environ.get(
            "NAV_FINISH_FRONTIER_PATIENCE", "3"))
        self.finish_map_stable_steps = int(os.environ.get(
            "NAV_FINISH_MAP_STABLE_STEPS", "100"))
        self.instance_merge_m = float(os.environ.get(
            "NAV_INSTANCE_MERGE_M", "0.75"))
        self.ground_top_k = int(os.environ.get("NAV_GROUND_TOP_K", "5"))
        # semantic_memory 后端（Phase 3）：pointing + 分级置信度 + 视觉伺服
        self.semantic_backend = os.environ.get(
            "NAV_SEMANTIC_BACKEND", "clip_sam")
        self.point_min_conf = float(os.environ.get("NAV_POINT_MIN_CONF", "0.5"))
        self.confirm_min_obs = int(os.environ.get("NAV_CONFIRM_MIN_OBS", "2"))
        self.min_target_pixels = float(os.environ.get(
            "NAV_MIN_TARGET_PIXELS", "32"))
        self.max_depth_std_m = float(os.environ.get(
            "NAV_MAX_DEPTH_STD_M", "0.5"))
        self.servo_max_steps = int(os.environ.get("NAV_SERVO_MAX_STEPS", "8"))
        self.servo_area_ratio = float(os.environ.get(
            "NAV_SERVO_AREA_RATIO", "0.04"))
        self.servo_center_tol = float(os.environ.get(
            "NAV_SERVO_CENTER_TOL", "0.25"))
        if self.semantic_backend == "semantic_memory":
            print("[NavAgent] semantic_memory 后端：NAV_MIN_SAM 阈值已弃用，"
                  "命中准入由 pointing 置信度与分级观测决定")
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
        # Phase 4 决策层：NAV_DECIDER=vlm 时事件驱动接管高层规划，
        # rules（默认）保持纯规则；FINISH 硬条件由状态机强制。
        self.decider_mode = os.environ.get("NAV_DECIDER", "rules")
        self.decision_loop = None
        if self.decider_mode == "vlm":
            if self.vlm.enabled:
                self.decision_loop = DecisionLoop(
                    chat_fn=self.vlm.agentic_chat,
                    tools={"query_memory": self._tool_query_memory,
                           "look_at": self._tool_look_at},
                    logger=DecisionTraceLogger(os.environ.get(
                        "NAV_DECIDER_LOG",
                        os.path.join(self.output_dir,
                                     "decision_trace.jsonl"))),
                    max_tool_rounds=int(os.environ.get(
                        "NAV_DECIDER_MAX_TOOL_ROUNDS", "3")),
                    finish_unexplored_max=float(os.environ.get(
                        "NAV_FINISH_UNEXPLORED_MAX", "0.15")))
            else:
                print("[NavAgent] WARNING: NAV_DECIDER=vlm 但 VLM API 未配置，"
                      "回退规则决策")
                self.decider_mode = "rules"
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
        self.memory = InstanceMemory()  # 多目标实例记忆（确认/访问/拉黑）
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
        # 分级置信度账本（semantic_memory 后端）：单帧观测->belief 锚点，
        # 独立多帧观测->confirmed；clip_sam 后端不用。
        self.ledger = ObservationLedger(min_obs=self.confirm_min_obs)
        # 末端视觉伺服状态
        self._servo_active = False
        self._servo_steps = 0
        self._servo_last_bbox = None
        # 决策层状态：近期事件流 + 最近一次探索规划的 frontier 缓存
        self._events = []
        self._last_frontier_clusters = []
        self._explore_grid = None
        self._last_finish_decision_step = -10 ** 9
        # 前沿引导探索状态
        self._explore_follower = None
        self._last_explore_plan = -10 ** 9
        self.explore_replan_interval = int(
            os.environ.get("NAV_EXPL_REPLAN_INTERVAL", "25"))
        self.explore_enabled = os.environ.get(
            "NAV_FRONTIER_EXPLORE", "1") == "1"
        # 节点信念（CLIP 先验）引导探索排序
        self.belief = BeliefMap()
        self.explore_belief_weight = float(
            os.environ.get("NAV_BELIEF_WEIGHT", "1.0"))
        self._frontier_empty_streak = 0
        self._last_frontier_count = None
        self._last_frontier_step = -10 ** 9
        self._recent_frontiers = []
        self._last_map_submaps = 0
        self._last_map_growth_step = 0
        self.frontier_cooldown_steps = int(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_STEPS", "100"))
        self.frontier_cooldown_m = float(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_M", "1.0"))

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
        # Quantifiers describe benchmark completion semantics, not the visual
        # category passed to CLIP/SAM.  This matters when VLM parsing is not
        # configured (for example, "exactly two baskets" -> "baskets").
        text = re.sub(
            r"^(?:(?:exactly|at\s+least|at\s+most)\s+)?"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+",
            "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:all|every)\s+(?:the\s+)?", "", text,
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
        """前沿引导探索：活跃探索路径优先，其次 VLM 短宏提示，
        碰撞恢复最优先，构建失败回退随机游走。"""
        if self._last_motion_failed:
            self._explore_hint_steps = 0
            self._explore_follower = None      # 碰撞后旧路径不可信
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
        if not self.explore_enabled:
            return super()._explore_action(observation)
        # 活跃探索路径：跟随；走完后立即尝试选新目标
        action = self._explore_follow(observation)
        if action is not None:
            return action
        if observation.step_count - self._last_explore_plan \
                >= self.explore_replan_interval:
            self._plan_exploration(observation)
            action = self._explore_follow(observation)
            if action is not None:
                return action
        return super()._explore_action(observation)

    # ------------------------------------------------------------------
    # 前沿引导探索
    # ------------------------------------------------------------------
    def _explore_follow(self, observation):
        """跟随探索路径一步。返回动作；无路径/走完/走丢返回 None。"""
        fl = self._explore_follower
        if fl is None:
            return None
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 3:
                order = np.argsort(frame_ids)
                fid = int(np.asarray(frame_ids)[order][-1])
                pose = np.asarray(poses, dtype=np.float64)[order][-1]
                if fid > fl.anchor_frame:
                    fl.update_anchor(pose, self.align_R, fid)
                    for a in self.calibrator.actions[fid - 1:]:
                        fl.dead_reckon(a)
        except Exception:
            pass
        action, arrived = fl.next_action()
        if arrived or action is None:
            self._explore_follower = None
            # 到达短 frontier 后保留正常重规划间隔，避免每一步重新选择
            # 同一个已经到达的边界。
            self._last_explore_plan = observation.step_count
            return None
        fl.dead_reckon(int(action))
        return int(action)

    def _plan_exploration(self, observation):
        """重建自由空间栅格，选 frontier 目标并规划探索路径。"""
        self._last_explore_plan = observation.step_count
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is None or len(poses) < 8:
                return
            poses64 = np.asarray(poses, dtype=np.float64)
            if self.align_R is None:
                self.align_R = nav.gravity_alignment(
                    poses64, cam_up=nav.mount_compensated_cam_up())
            frames = self.client.get_frame_points(stride=6)
            if not frames:
                return
            grid = nav.OccupancyGrid.from_frame_points(
                frames, self.align_R)
            if grid is None:
                return
            self._explore_grid = grid
        except Exception as e:
            print(f"[NavAgent] 探索栅格构建失败: {e}")
            return

        order = np.argsort(frame_ids)
        cur = poses64[order][-1][:3, 3] @ self.align_R.T
        # 尺度：栅格尺规（相机离地 1.5m）比在线标定更直接
        scale = 1.0 / grid.unit_per_m if grid.unit_per_m > 0 else None

        # 骨架拓扑：实例记忆挂载 + 节点信念更新
        graph = skel.build_skeleton_graph(grid)
        if graph is not None:
            self.memory.attach_to_skeleton(graph)
            self.belief.update(self.client, graph, self.target_text,
                               self.align_R, observation.step_count)
        if self.semantic_backend == "semantic_memory":
            # 分级置信度 belief 锚点作为探索先验（打分骨架不动）
            self.belief.set_external_anchors([
                (a.point[:2], a.score)
                for a in self.ledger.belief_anchors(self.target_text)])

        clusters = skel.frontier_clusters(grid, min_size=5)
        try:
            num_submaps = int(self.client.get_state().get("num_submaps", 0))
            if num_submaps > self._last_map_submaps:
                self._last_map_growth_step = observation.step_count
                self._last_map_submaps = num_submaps
        except Exception:
            pass
        self._last_frontier_count = len(clusters)
        self._last_frontier_step = observation.step_count
        self._last_frontier_clusters = list(clusters)
        if clusters:
            self._frontier_empty_streak = 0
        else:
            self._frontier_empty_streak += 1
        self._recent_frontiers = [
            item for item in self._recent_frontiers
            if observation.step_count - item[1] <= self.frontier_cooldown_steps
        ]
        # 过滤太近（<1m，已看过）并打分：
        # 大小/(1+距离m) × (1+信念权重×节点信念)
        cands = []
        for c in clusters:
            d_units = math.hypot(c["world"][0] - cur[0],
                                 c["world"][1] - cur[1])
            d_m = d_units * scale if scale else d_units
            if scale and d_m < 1.0:
                continue
            if scale and any(
                    math.hypot(c["world"][0] - old_xy[0],
                               c["world"][1] - old_xy[1]) * scale
                    < self.frontier_cooldown_m
                    for old_xy, _old_step in self._recent_frontiers):
                continue
            belief = self.belief.belief_at(c["world"], graph)
            score = c["size"] / (1.0 + d_m) * \
                (1.0 + self.explore_belief_weight * belief)
            cands.append((score, c))
        cands.sort(key=lambda t: -t[0])

        for _, c in cands[:5]:
            path = grid.astar(cur[:2], c["world"])
            if path is None or len(path) < 2:
                continue
            path = grid.shortcut(path)
            fl = nav.PathFollower(
                scale=scale or 1.0, reach_m=self.reach_m)
            fl.set_path(path)
            self._explore_follower = fl
            self._recent_frontiers.append((
                np.asarray(c["world"], dtype=np.float64)[:2],
                observation.step_count))
            print(f"[NavAgent] step={observation.step_count} 探索目标 "
                  f"frontier size={c['size']} 路径 {len(path)} 点")
            return

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

        if self._servo_active:
            # 末端视觉伺服中：不看坐标看图像，直到"近且居中"或超限
            action = self._servo_step(observation)
            self._record_and_update(observation, action)
            return action

        if self.mode == "explore" and self._should_finish(observation):
            action = int(Action.FINISH)
        elif self.mode == "reported":
            if self._should_finish(observation):
                action = int(Action.FINISH)
            else:
                self._clear_current_target()
                self.mode = "explore"
                action = None
                if getattr(self, "decision_loop", None) is not None:
                    # 事件驱动：新实例确认后由决策层定下一步
                    action = self._decider_next(
                        observation, "instance_confirmed")
                if action is None:
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
                        action = self._confirm_and_report(observation)
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
        self._explore_follower = None
        self._plan_failures = 0
        self._scanning = False
        self._selected_evidence = None
        self._servo_active = False
        self._servo_steps = 0
        self._servo_last_bbox = None

    def _merge_dist(self):
        """实例合并/黑名单距离（统一对齐地图单位）。"""
        scale = self.calibrator.current_scale() or 1.0
        return getattr(self, "instance_merge_m", 0.75) / scale

    def _ensure_alignment(self):
        """每个 episode 固定一套重力对齐坐标，供地图、记忆、TSP 共用。"""
        if self.align_R is not None:
            return True
        try:
            poses, _frame_ids = self.client.get_all_poses()
            if poses is None or len(poses) < 3:
                return False
            self.align_R = nav.gravity_alignment(
                np.asarray(poses, dtype=np.float64),
                cam_up=nav.mount_compensated_cam_up())
            return True
        except Exception:
            return False

    def _aligned_point(self, point):
        point = np.asarray(point, dtype=np.float64)
        align_R = getattr(self, "align_R", None)
        return align_R @ point if align_R is not None else point

    def _raw_point(self, point):
        point = np.asarray(point, dtype=np.float64)
        align_R = getattr(self, "align_R", None)
        return align_R.T @ point if align_R is not None else point

    def _refresh_memory_candidates(self):
        """回环/图优化后按 candidate_id 刷新持久实例坐标。"""
        if not self._ensure_alignment():
            return
        for node in self.memory.nodes:
            if not node.candidate_id:
                continue
            try:
                resolved = self.client.resolve_candidate(node.candidate_id)
                if resolved.get("found"):
                    self.memory.refresh_point(
                        node, self._aligned_point(resolved["point"]))
            except Exception:
                pass

    def _report_found(self):
        if self.target_point is not None:
            node, _ = self.memory.add_or_merge(
                self.target_text, self._aligned_point(self.target_point), 1.0,
                merge_dist=self._merge_dist(), status="confirmed",
                step=getattr(self, "_last_report_step", 0),
                candidate_id=self.target_candidate_id)
            self.memory.mark_visited(node)
        self._reported_count += 1
        self._no_hit_queries = 0
        self._log_event(f"reported TARGET_FOUND '{self.target_text}' "
                        f"(total {self._reported_count})")
        self.mode = "reported"
        self._scanning = False
        return int(Action.TARGET_FOUND)

    def _reject_current_target(self, observation):
        if self.target_point is not None:
            node, is_new = self.memory.add_or_merge(
                self.target_text, self._aligned_point(self.target_point), 0.0,
                merge_dist=self._merge_dist(), status="rejected")
            if not is_new:
                self.memory.mark_rejected(node)
            self.ledger.discard_near(
                self.target_text, self._aligned_point(self.target_point),
                self._merge_dist())
        self._log_event(f"rejected candidate '{self.target_text}'")
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
            "rejected_or_duplicate_location_count": sum(
                1 for n in self.memory.nodes if n.status == "rejected"),
            "has_agent_estimated_metric_scale":
                self.calibrator.current_scale() is not None,
            **mapping_state,
        }

    # ------------------------------------------------------------------
    # Phase 4：VLM 决策层（NAV_DECIDER=vlm，事件驱动，不进控制回路）
    # ------------------------------------------------------------------
    def _log_event(self, message):
        self._events.append(str(message))
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def _tool_query_memory(self, text):
        """决策层只读工具：caption 语义记忆检索。"""
        try:
            results = self.client.retrieve_captions(text, top_k=5)
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return [{"frame_id": r.get("frame_id"),
                 "score": round(float(r.get("score", 0.0)), 3),
                 "caption": str(r.get("caption", ""))[:300]}
                for r in results]

    def _tool_look_at(self, frame_id):
        """决策层只读工具：取关键帧图像（JPEG bytes）。"""
        try:
            meta, payload = self.client.get_frame_image(frame_id)
        except Exception:
            return None
        return payload if meta.get("found") else None

    def _build_decider_input(self, observation):
        """组装决策输入：世界状态 JSON + 俯视标注地图 PNG（编号一致）。"""
        from agents.decision_state import build_world_state
        grid = self.grid if self.grid is not None else self._explore_grid
        frontiers = self._last_frontier_clusters
        state = build_world_state(self, observation, grid=grid,
                                  frontiers=frontiers)
        map_png = None
        if grid is not None:
            try:
                from agents.map_render import render_topdown
                trajectory = None
                poses, frame_ids = self.client.get_all_poses()
                if poses is not None and len(poses) and \
                        self.align_R is not None:
                    centers = np.asarray(poses, dtype=np.float64)[:, :3, 3] \
                        @ self.align_R.T
                    trajectory = [tuple(p[:2]) for p in centers[::4]]
                pose = None
                if self.follower is not None and \
                        self.follower.anchor_frame >= 0:
                    pose = (self.follower.x, self.follower.y,
                            self.follower.yaw)
                map_png = render_topdown(
                    grid, trajectory=trajectory, pose=pose,
                    instances=[{"id": nd.iid, "xy": tuple(nd.point[:2]),
                                "visited": nd.status == "visited"}
                               for nd in self.memory.nodes
                               if nd.status != "rejected"],
                    anchors=[{"id": f"b{i}", "xy": tuple(a.point[:2])}
                             for i, a in enumerate(
                                 self.ledger.belief_anchors(
                                     self.target_text))],
                    frontiers=[{"id": f"f{i}", "xy": tuple(c["world"][:2])}
                               for i, c in enumerate(frontiers)])
            except Exception as exc:
                print(f"[NavAgent] 俯视地图渲染失败: {exc}")
                map_png = None
        return state, map_png

    def _apply_decider_steering(self, observation, result):
        """把决策结果映射到现有状态机动作（GOTO_INSTANCE/GOTO_FRONTIER）。
        底层跟随/避障/重规划仍由确定性模块执行。"""
        if result.action == "GOTO_INSTANCE" and result.target_id is not None:
            for nd in self.memory.unvisited(self.target_text):
                if str(nd.iid) == str(result.target_id):
                    self.target_point = self._raw_point(nd.point)
                    self.target_candidate_id = nd.candidate_id
                    self._explore_follower = None
                    self._log_event(
                        f"decider -> GOTO_INSTANCE {nd.iid}")
                    if self._plan_to_target(observation):
                        self.mode = "nav"
                    return
        if result.action == "GOTO_FRONTIER" and result.target_id is not None:
            cluster = None
            for i, c in enumerate(self._last_frontier_clusters):
                if f"f{i}" == str(result.target_id):
                    cluster = c
                    break
            grid = self._explore_grid
            cur = self._current_aligned_xy()
            if cluster is None or grid is None or cur is None:
                return
            path = grid.astar(cur, cluster["world"])
            if not path or len(path) < 2:
                return
            scale = self.calibrator.current_scale() or 1.0
            fl = nav.PathFollower(scale=scale, reach_m=self.reach_m)
            fl.set_path(grid.shortcut(path))
            self._explore_follower = fl
            self._recent_frontiers.append((
                np.asarray(cluster["world"], dtype=np.float64)[:2],
                observation.step_count))
            self._log_event(
                f"decider -> GOTO_FRONTIER {result.target_id}")

    def _decider_should_finish(self, observation):
        """all 模式终止决策（VLM 决策层）。True/False；模型不可用返回
        None（调用方回退规则）。FINISH 硬条件由 DecisionLoop 强制。"""
        step = observation.step_count
        if self._reported_count <= 0:
            return False
        if self.memory.unvisited(self.target_text):
            return False                      # 还有已确认未访问实例
        if step - self._last_finish_decision_step < self.query_interval:
            return False
        late = step >= int(0.5 * observation.max_steps)
        frontier_quiet = self._frontier_empty_streak >= \
            self.finish_frontier_patience
        if not (late or frontier_quiet):
            return False
        self._last_finish_decision_step = step
        state, map_png = self._build_decider_input(observation)
        result = self.decision_loop.decide("finish_check", state, map_png)
        if result is None:
            return None                       # 回退规则
        print(f"[NavAgent] 决策层 finish_check: {result}")
        if result.action == "FINISH":
            return True
        self._apply_decider_steering(observation, result)
        return False

    def _decider_next(self, observation, event):
        """事件驱动咨询决策层下一步。返回 action 或 None（回退默认流程）。"""
        try:
            state, map_png = self._build_decider_input(observation)
            result = self.decision_loop.decide(event, state, map_png)
        except Exception as exc:
            print(f"[NavAgent] 决策层调用失败，回退规则: {exc}")
            return None
        if result is None:
            return None
        print(f"[NavAgent] 决策层 {event}: {result}")
        self._log_event(
            f"decider {event} -> {result.action} {result.target_id}")
        if result.action == "FINISH":
            # FINISH 硬条件已在 DecisionLoop 内强制（many 计数 / all 终止账本）
            return int(Action.FINISH)
        if result.action in ("GOTO_INSTANCE", "GOTO_FRONTIER"):
            self._apply_decider_steering(observation, result)
            if self.mode == "nav":
                action, arrived = self._nav_action(observation)
                if action is not None:
                    return action
                self.mode = "explore"
            return self._explore_action(observation)
        return None                             # VERIFY 等：默认流程处理

    def _should_finish(self, observation):
        if self._target_mode == "many" and self._target_count is not None:
            return self._reported_count >= int(self._target_count)
        if self._target_mode == "all":
            if getattr(self, "decision_loop", None) is not None:
                decided = self._decider_should_finish(observation)
                if decided is not None:
                    return decided
                # 模型不可用：落回下面的确定性规则
            late = observation.step_count >= int(0.8 * observation.max_steps)
            frontier_fresh = observation.step_count - self._last_frontier_step \
                <= 2 * self.explore_replan_interval
            no_pending = not self.memory.unvisited(self.target_text)
            geometric_ready = self._reported_count > 0 and late and no_pending and \
                self._no_hit_queries >= self.finish_patience and \
                frontier_fresh and self._last_frontier_count == 0 and \
                self._frontier_empty_streak >= self.finish_frontier_patience
            fallback_late = observation.step_count >= int(
                0.95 * observation.max_steps)
            map_stable = observation.step_count - self._last_map_growth_step \
                >= self.finish_map_stable_steps
            fallback_ready = geometric_ready and fallback_late and map_stable
            if not self.vlm.enabled:
                return fallback_ready
            if not geometric_ready:
                return False
            if observation.step_count - self._last_finish_vlm_step < \
                    self.query_interval:
                return False
            self._last_finish_vlm_step = observation.step_count
            self._target_phrase(observation)
            decision = self.vlm.decide_finish(
                observation.goal_text, self.target_spec,
                self._decision_state(observation), observation.rgb)
            if decision is None:
                return fallback_ready
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
        """当前帧视觉确认（clip_sam: SAM3；semantic_memory: pointing+VQA
        复核）后，再由 VLM 战略层核对完整语义属性。"""
        try:
            r = self.client.ground_frame(observation.rgb, self.target_text)
        except Exception as e:
            print(f"[NavAgent] ground_frame 失败: {e}")
            return "scan"
        min_conf = self.point_min_conf \
            if self.semantic_backend == "semantic_memory" else self.verify_min
        found = bool(r.get("found")) and \
            r.get("score", 0.0) >= min_conf
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
            return self._confirm_and_report(observation)
        if arrival == "reject":
            return self._reject_current_target(observation)
        self._scan_steps += 1
        if self._scan_steps < 12:
            return int(Action.TURN_LEFT)
        self._scanning = False
        self._verify_failures += 1
        self._log_event(f"360 scan failed for '{self.target_text}', "
                        f"blacklisted ({self._verify_failures} total)")
        print(f"[NavAgent] step={observation.step_count} 360° 扫描未确认，"
              f"拉黑目标点（累计 {self._verify_failures} 次）")
        if self.target_point is not None:
            node, is_new = self.memory.add_or_merge(
                self.target_text, self._aligned_point(self.target_point), 0.0,
                merge_dist=self._merge_dist(), status="rejected")
            if not is_new:
                self.memory.mark_rejected(node)
            self.ledger.discard_near(
                self.target_text, self._aligned_point(self.target_point),
                self._merge_dist())
        self._clear_current_target()
        self.mode = "explore"
        return self._explore_action(observation)

    def _is_bad_point(self, point):
        """目标点是否接近已拒绝/已访问实例（默认 0.75m，可配置）。"""
        dist = self._merge_dist()
        aligned = self._aligned_point(point)
        return self.memory.is_rejected(self.target_text, aligned, dist) or \
            self.memory.is_visited(self.target_text, aligned, dist)

    def _order_by_route(self, eligible):
        """候选实例按开路径 TSP 重排（欧氏距离近似），失败时保持原序。"""
        try:
            start = None
            if self.follower is not None and self.follower.anchor_frame >= 0:
                start = (self.follower.x, self.follower.y)
            else:
                poses, frame_ids = self.client.get_all_poses()
                if poses is not None and len(poses) and self.align_R is not None:
                    order = np.argsort(frame_ids)
                    cur = np.asarray(poses, dtype=np.float64)[order][-1]
                    cur = cur[:3, 3] @ self.align_R.T
                    start = (float(cur[0]), float(cur[1]))
            if start is None:
                return eligible
            goals = [tuple(self._aligned_point(h["point"])[:2])
                     for h in eligible]

            def euclid(a, b):
                return math.hypot(a[0] - b[0], a[1] - b[1])

            order = planner.route_order(start, goals, euclid)
            return [eligible[i] for i in order]
        except Exception as e:
            print(f"[NavAgent] TSP 排序失败，保持原序: {e}")
            return eligible

    def _current_aligned_xy(self):
        if not self._ensure_alignment():
            return None
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is None or not len(poses):
                return None
            order = np.argsort(frame_ids)
            point = np.asarray(poses, dtype=np.float64)[order][-1][:3, 3]
            aligned = self._aligned_point(point)
            return float(aligned[0]), float(aligned[1])
        except Exception:
            return None

    def _ordered_memory_nodes(self):
        """从持久 confirmed memory 产生当前模式的真实规划序列。"""
        instances = self.memory.unvisited(self.target_text)
        if not instances:
            return []
        start = self._current_aligned_xy()
        if start is None:
            return instances

        def euclid(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        if self._target_mode == "many":
            need = max(int(self._target_count or 0) - self._reported_count, 0)
            ordered, _gap = planner.plan_multi(
                start, instances, euclid, need=need)
            return ordered
        if self._target_mode == "all":
            ordered, _gap = planner.plan_multi(
                start, instances, euclid, need=len(instances))
            return ordered
        selected = planner.select_goal_any(start, instances, euclid)
        return [selected] if selected is not None else []

    def _activate_memory_target(self, observation):
        """选择持久实例、经 VLM/fallback 审阅，并规划第一段。"""
        nodes = self._ordered_memory_nodes()
        if not nodes:
            return False
        candidates = [{
            "candidate_id": nd.candidate_id,
            "point": self._raw_point(nd.point).tolist(),
            "sam_score": nd.score,
            "frame_id": nd.frame_id,
            "memory_iid": nd.iid,
        } for nd in nodes[:self.vlm_candidate_limit]]
        best, evidence = self._vlm_candidate_decision(observation, candidates)
        if best is None:
            return False
        self.target_point = np.asarray(best["point"], dtype=np.float64)
        self.target_candidate_id = best.get("candidate_id")
        self._selected_evidence = evidence
        self._no_hit_queries = 0
        self._explore_follower = None
        print(f"[NavAgent] 从实例记忆选择 #{best.get('memory_iid')} "
              f"sam={best.get('sam_score', 0.0):.3f}")
        if self._plan_to_target(observation):
            self.mode = "nav"
        return True

    # ------------------------------------------------------------------
    # EXPLORE：semantic_memory 命中准入（分级置信度）
    # ------------------------------------------------------------------
    def _ingest_semantic_hits(self, observation, hits):
        """pointing 命中 -> 观测账本：单帧 belief 锚点 / 多帧独立观测升级
        confirmed 进 TSP；小/远目标强制留 belief 当探索先验。"""
        step = observation.step_count
        scale = self.calibrator.current_scale() or 1.0
        self.ledger.min_pose_sep = 0.5 / scale
        hits.sort(key=lambda r: r.get("point_score", r.get("sam_score", 0.0)),
                  reverse=True)
        any_confirmed = False
        n_belief = 0
        for h in hits:
            conf = h.get("point_score", h.get("sam_score", 0.0))
            if conf < self.point_min_conf:
                continue
            pt = np.asarray(h["point"], dtype=np.float64)
            if self._is_bad_point(pt):
                continue
            aligned = self._aligned_point(pt)
            outcome, anchor = self.ledger.add_observation(
                self.target_text, aligned, conf,
                merge_dist=self._merge_dist(),
                frame_id=h.get("frame_id"), step=step,
                obs_xy=self._hit_obs_xy(h),
                force_belief=self._is_small_or_far(h, scale))
            if outcome == "confirmed":
                node, _ = self.memory.add_or_merge(
                    self.target_text, anchor.point, anchor.score,
                    merge_dist=self._merge_dist(), status="confirmed",
                    frame_id=h.get("frame_id"), step=step,
                    candidate_id=h.get("candidate_id"))
                node.n_obs = anchor.n_obs
                self.ledger.discard(anchor)
                any_confirmed = True
                print(f"[NavAgent] step={step} 实例经 {anchor.n_obs} 帧独立"
                      f"观测升级 confirmed（conf={anchor.score:.2f}）")
            elif outcome == "belief":
                n_belief += 1
        if not self._activate_memory_target(observation):
            if not any_confirmed:
                self._no_hit_queries += 1
            print(f"[NavAgent] step={step} '{self.target_text}' "
                  f"新增 belief 锚点 {n_belief} 个"
                  f"（confirmed={'有' if any_confirmed else '无'}），继续探索")

    def _hit_obs_xy(self, hit):
        """命中帧的拍照位姿（对齐地图坐标 xy），用于独立观测判定。"""
        pose = hit.get("pose")
        if pose is None or self.align_R is None:
            return None
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            return None
        return (pose[:3, 3] @ self.align_R.T)[:2]

    def _is_small_or_far(self, hit, scale):
        """小/远目标两段式：目标像素占比小或 patch 深度方差大时，
        不登记发现，降级 belief 锚点，逼近后复核。"""
        bbox = hit.get("bbox")
        if bbox is not None:
            side = max(float(bbox[2]) - float(bbox[0]),
                       float(bbox[3]) - float(bbox[1]))
            if side < self.min_target_pixels:
                return True
        depth_std = hit.get("depth_std")
        if depth_std is not None and depth_std * scale > self.max_depth_std_m:
            return True
        return False

    # ------------------------------------------------------------------
    # 末端视觉伺服（semantic_memory 后端）
    # ------------------------------------------------------------------
    def _confirm_and_report(self, observation):
        """确认通过 -> TARGET_FOUND。semantic_memory 后端先进入末端视觉
        伺服：最后一段逼近不看坐标看图像，把 benchmark 0.25m 判定与
        SLAM 0.33-1.0m 误差解耦；超限/异常退回坐标判定。"""
        if self.semantic_backend == "semantic_memory" and \
                not self._servo_active:
            self._servo_active = True
            self._servo_steps = 0
            self._servo_last_bbox = None
            return self._servo_step(observation)
        return self._report_found()

    def _servo_step(self, observation):
        """视觉伺服一步：目标近且居中 -> TARGET_FOUND；否则对中/逼近；
        超 NAV_SERVO_MAX_STEPS 步退回坐标判定（已在 0.8m 到达圈内）。"""
        self._servo_steps += 1
        if self._servo_steps > self.servo_max_steps:
            print(f"[NavAgent] step={observation.step_count} "
                  f"伺服超 {self.servo_max_steps} 步，退回坐标判定")
            self._servo_active = False
            return self._report_found()
        try:
            r = self.client.ground_frame(observation.rgb, self.target_text)
        except Exception as e:
            print(f"[NavAgent] 伺服 ground_frame 失败: {e}，退回坐标判定")
            self._servo_active = False
            return self._report_found()
        bbox = r.get("bbox")
        if r.get("found") and bbox:
            self._servo_last_bbox = bbox
            h, w = np.asarray(observation.rgb).shape[:2]
            x0, y0, x1, y1 = (float(v) for v in bbox)
            area_ratio = max(0.0, (x1 - x0) * (y1 - y0)) / float(h * w)
            center_off = abs((x0 + x1) / 2 - w / 2) / float(w)
            if area_ratio >= self.servo_area_ratio and \
                    center_off <= self.servo_center_tol:
                print(f"[NavAgent] step={observation.step_count} 伺服确认："
                      f"占比 {area_ratio:.3f} 居中 {center_off:.2f}，"
                      f"TARGET_FOUND")
                self._servo_active = False
                return self._report_found()
            if center_off > 0.15:
                return int(Action.TURN_LEFT if (x0 + x1) / 2 < w / 2
                           else Action.TURN_RIGHT)
            return int(Action.MOVE_FORWARD)
        # 目标暂时丢出视野：朝最后已知方向找回（计入步数上限）
        last = self._servo_last_bbox
        if last is not None:
            w = np.asarray(observation.rgb).shape[1]
            return int(Action.TURN_LEFT if (last[0] + last[2]) / 2 < w / 2
                       else Action.TURN_RIGHT)
        return int(Action.TURN_LEFT)

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
        self._ensure_alignment()
        self._refresh_memory_candidates()
        # 先完成已经看见但尚未访问的持久实例，再继续扩大搜索空间。
        if self._activate_memory_target(observation):
            return
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
        if self.semantic_backend == "semantic_memory":
            self._ingest_semantic_hits(observation, hits)
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
        # 所有当前命中先写入 confirmed memory；planner 随后对持久集合规划，
        # 而不是只对这一次查询的临时候选排序。
        for hit in eligible:
            self.memory.add_or_merge(
                self.target_text, self._aligned_point(hit["point"]),
                hit.get("sam_score", 0.0), merge_dist=self._merge_dist(),
                status="confirmed", frame_id=hit.get("frame_id"),
                step=step, candidate_id=hit.get("candidate_id"))
        if not self._activate_memory_target(observation):
            self._no_hit_queries += 1
            print(f"[NavAgent] step={step} VLM 建议继续探索")

    def _vlm_candidate_decision(self, observation, candidates):
        """返回 (selected candidate, evidence bytes)；无 VLM 时取最高 SAM。"""
        if not self.vlm.enabled:
            return (candidates[0], None) if candidates else (None, None)
        # This helper is also used on no-hit/VLM-only paths, so do not rely on
        # _maybe_query_target having initialized the memory category first.
        self.target_text = self._target_phrase(observation)
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
                node, is_new = self.memory.add_or_merge(
                    self.target_text,
                    self._aligned_point(rejected["point"]),
                    0.0, merge_dist=self._merge_dist(), status="rejected")
                if not is_new:
                    self.memory.mark_rejected(node)
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
                np.asarray(poses, dtype=np.float64),
                cam_up=nav.mount_compensated_cam_up())
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
                    print("[NavAgent] 候选缓存已失效，使用记忆中的最近坐标")
                    self.target_candidate_id = None
                else:
                    self.target_point = np.asarray(
                        resolved["point"], dtype=np.float64)
            except Exception as e:
                print(f"[NavAgent] 候选重投影失败，使用记忆坐标: {e}")
                self.target_candidate_id = None
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
        # 自由空间栅格：优先逐帧局部地板锚定投票（对抗子图间漂移），
        # 其次全局点云双切片，最后回退面包屑走廊（保底）。
        self.grid = None
        try:
            frames = self.client.get_frame_points(stride=6)
            if frames:
                self.grid = nav.OccupancyGrid.from_frame_points(
                    frames, self.align_R)
        except Exception as e:
            print(f"[NavAgent] 逐帧栅格构建异常: {e}")
            self.grid = None
        if self.grid is None:
            try:
                pts, _cols = self.client.get_map_points(max_points=800000)
                if pts is not None and len(pts) > 1000:
                    pts_aligned = np.asarray(pts, dtype=np.float64) \
                        @ self.align_R.T
                    self.grid = nav.OccupancyGrid.build(
                        pts_aligned, cam_centers)
            except Exception as e:
                print(f"[NavAgent] 点云栅格构建异常: {e}")
                self.grid = None
        if self.grid is None:
            print("[NavAgent] 点云栅格不可用，回退轨迹走廊")
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
