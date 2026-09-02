"""多目标导航 agent：语义记忆 -> 实例定位 -> 栅格规划 -> 路径跟随。

流程（多目标状态机）：
1. EXPLORE：持续建图；决策 VLM 自主调用 propose_candidates（SAM 全分割）
   挑选编号 mask → som_pick 注册候选 → commit_candidates 裁决，经 3D
   几何验证后写入实例记忆。
2. 拿到目标点后用点云构建 2D 占据栅格（agents/navigator.py），A* 规划，
   进入 NAV 模式沿路径输出离散动作。位姿锚定最新关键帧 + 航位推算，
   定期重建栅格并重规划（地图随探索增长，回环也会改写历史位姿）。
3. 到达候选点后把当前观测和候选证据交给决策 VLM，由它直接选择报告、
   扫描、探索或更换目标；不再自动执行当前帧 pointing、verify 或视觉伺服。

目标短语只来自公开 instruction（或显式 NAV_TARGET 调试覆盖），不读取
query_program、GPS、深度或仿真器姿态。

运行方式与 MappingAgent 相同：
    --agent agents.nav_agent:NavAgent
"""

import json
import math
import os
import re
import time

import numpy as np

from benchmark_api import Action
from agents import navigator as nav
from agents import planner
from agents import skeleton as skel
from agents.mapping_agent import MappingAgent
from agents.memory import InstanceMemory
from decision import (DecisionLoop, DecisionResult, DecisionTraceLogger,
                      VLMDecisionClient)
from runtime_paths import env_debug_path

class NavAgent(MappingAgent):
    def __init__(self):
        super().__init__()
        self.query_interval = int(os.environ.get("NAV_QUERY_INTERVAL", "20"))
        self.replan_interval = int(os.environ.get("NAV_REPLAN_INTERVAL", "20"))
        self.nav_collision_limit = max(1, int(os.environ.get(
            "NAV_NAV_COLLISION_LIMIT", "3")))
        self.nav_escape_turns = max(1, int(os.environ.get(
            "NAV_NAV_ESCAPE_TURNS", "1")))
        self.nav_block_radius_m = max(0.1, float(os.environ.get(
            "NAV_NAV_BLOCK_RADIUS_M", "0.35")))
        self.nav_block_ttl_steps = max(1, int(os.environ.get(
            "NAV_NAV_BLOCK_TTL_STEPS", "80")))
        self.reach_m = float(os.environ.get("NAV_REACH_M", "0.8"))
        self.finish_patience = int(os.environ.get("NAV_FINISH_PATIENCE", "5"))
        self.finish_frontier_patience = int(os.environ.get(
            "NAV_FINISH_FRONTIER_PATIENCE", "3"))
        self.finish_map_stable_steps = int(os.environ.get(
            "NAV_FINISH_MAP_STABLE_STEPS", "100"))
        self.ground_top_k = int(os.environ.get("NAV_GROUND_TOP_K", "2"))
        self.relevant_frame_top_k = max(1, min(20, int(os.environ.get(
            "NAV_RELEVANT_FRAME_TOP_K", "5"))))
        self.adjust_max_steps = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_STEPS", "10")))
        self.adjust_max_tilt_steps = max(0, int(os.environ.get(
            "NAV_ADJUST_MAX_TILT_STEPS", "1")))
        self.adjust_max_sessions_per_target = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_SESSIONS_PER_TARGET", "2")))
        self.adjust_max_total_steps_per_target = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_TOTAL_STEPS_PER_TARGET", "8")))
        self.adjust_max_turns_per_target = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_TURNS_PER_TARGET", "4")))
        # 一次 MOVE_FORWARD 决策允许连续执行的最大前进步数（每步 0.25m）；
        # VLM 必须显式选择步数，harness 逐步执行、碰撞即停。
        self.adjust_max_forward_steps = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_FORWARD_STEPS", "8")))
        self.adjust_map_radius_m = max(1.0, float(os.environ.get(
            "NAV_ADJUST_MAP_RADIUS_M", "4.0")))
        # 唯一候选生成语义链路：caption 检索 + pointing + 3D instance memory
        self.vlm = VLMDecisionClient.from_env()
        self.vlm.set_trace_path(env_debug_path(
            "NAV_VLM_TRACE", os.path.join(self.output_dir, "vlm_calls.jsonl")))
        # Save the byte-identical images sent to the decision API beside its
        # JSONL trace. This includes current RGB and the RGB point-cloud map.
        self.vlm.image_dir = env_debug_path(
            "NAV_VLM_IMAGE_DIR", os.path.join(self.output_dir, "vlm_inputs"))
        print(f"[NavAgent] VLM 战略层: "
              f"{'enabled (' + self.vlm.model + ')' if self.vlm.enabled else 'disabled'}")
        # Phase 4 决策层：NAV_DECIDER=vlm 时事件驱动接管高层规划，
        # rules（默认）保持纯规则；FINISH 硬条件由状态机强制。
        self.decider_mode = os.environ.get("NAV_DECIDER", "rules")
        self.decision_loop = None
        if self.decider_mode == "vlm":
            if self.vlm.enabled:
                require_vlm_probe = os.environ.get(
                    "NAV_REQUIRE_DECISION_PREFLIGHT", "1").strip().lower() in {
                        "1", "true", "yes", "on"}
                if require_vlm_probe and not self.vlm.probe():
                    raise RuntimeError(
                        "DECISION_BACKEND_UNAVAILABLE: live generation failed: "
                        f"{self.vlm._last_error or 'invalid response'}")
                self.decision_loop = DecisionLoop(
                    chat_fn=self.vlm.agentic_chat,
                    tools={"search_frames": self._tool_search_frames,
                           "search_instances": self._tool_search_instances,
                           "view_instance": self._tool_view_instance,
                           "get_instance": self._tool_get_instance,
                           "update_instance": self._tool_update_instance,
                           "view_frame": self._tool_view_frame,
                           "propose_candidates": self._tool_propose_candidates,
                           "commit_candidates": self._tool_commit_candidates,
                           "review_crosshair": self._tool_review_crosshair,
                           "instantiate_points": self._tool_instantiate_points,
                           "som_pick": self._tool_som_pick,
                           "resolve_duplicate": self._tool_resolve_duplicate,
                           "get_agent_status": self._tool_get_agent_status,
                           "set_notes": self._tool_set_notes,
                           "get_action_history": self._tool_get_action_history},
                    logger=DecisionTraceLogger(env_debug_path(
                        "NAV_DECIDER_LOG",
                        os.path.join(self.output_dir,
                                     "decision_trace.jsonl"))),
                    max_tool_rounds=int(os.environ.get(
                        "NAV_DECIDER_MAX_TOOL_ROUNDS", "15")))
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
        self.target_instance_id = None
        self.follower = None
        self.grid = None
        self.align_R = None
        # harness：VLM 工作记忆、动作流水与新关键帧通知水位
        self._notes = ""
        self._action_log = []          # [{step, action, target_id, outcome}]
        self._last_notified_frame_id = 0
        self._last_observation = None
        self._last_plan_step = -10 ** 9
        self._last_anchor_step = -10 ** 9
        self._plan_failures = 0
        # nav 模式卡死检测：连续碰撞（撞墙原地不动）达到 nav_collision_limit
        # 次后放弃当前目标并触发 nav_failed 决策。explore 模式已有碰撞恢复，
        # nav 模式此前没有——撞墙会永远重规划同一路径直到步数耗尽。
        self._nav_collision_streak = 0
        self._nav_stuck_replanned = False
        self._nav_recovery_queue = []
        self._nav_recovery_stage = 0
        self._nav_blocked_points = []
        self._metric_replan_required = False
        # 导航确认不可达的实例（episode 内排除出 VLM 候选表；REPORT_FOUND
        # 仍可用——agent 可能就停在目标旁边，直接观察可确认）。
        self._unreachable_instance_ids = set()
        self._scanning = False          # 到达后原地 360° 扫描确认中
        self._scan_steps = 0
        self._scan_images = []
        self.memory = InstanceMemory()
        self._reported_count = 0
        self._last_dup_reviews = []
        # 评测采集（get_target_pool）只读锚点状态：世界系 (gps, compass)
        # 在首个有效 act 观测记录一次；SLAM 侧锚点（重力对齐系中最早
        # 关键帧位姿）在每次重规划时刷新，跟随回环对历史位姿的改写。
        self._pool_world_anchor = None
        self._pool_slam_anchor = None
        self._no_hit_queries = 0
        self._target_mode = "any"
        self._target_count = None
        self._selected_evidence = None
        self._arrival_transition_active = False
        # VLM 显式 START_ADJUST/END_ADJUST 控制的通用微调状态。
        self._adjusting = False
        self._adjust_steps = 0
        self._adjust_source_event = None
        self._adjust_context_images = []
        self._adjust_pitch_steps = 0
        self._adjust_leveling = False
        self._adjust_end_reason = None
        # Prevent START_ADJUST -> END_ADJUST -> START_ADJUST recursion within
        # one benchmark observation when the VLM repeats the same request.
        self._adjust_reentry_blocked_step = None
        self._adjustment_key = None
        self._adjustment_budgets = {}
        self._adjust_goal = None
        self._adjust_start_frame_id = None
        # MOVE_FORWARD steps>1 的续执行计数：逐步执行，碰撞即停并交还 VLM。
        self._adjust_repeat_remaining = 0
        self._adjust_repeat_total = 0
        self._adjust_progress = {"successful_moves": 0, "fresh_views": 0,
                                 "new_keyframes": 0}
        self._last_decision_output = None
        # 十字证据与显式语义审核分开保存。证据已展示绝不等于审核通过：
        # 只有 Decision VLM 调用 review_crosshair(..., verdict="ACCEPT")
        # 的同一像素才可进入 3D 实例化。
        self._crosshair_evidence = {}
        self._crosshair_reviews = {}
        # Candidate transaction state.  A proposal is not a navigation target:
        # only explicitly accepted proposals are inserted into InstanceMemory.
        self._proposals = {}
        self._proposal_limit = 128
        # 被拒候选记忆：(frame_id, round(x), round(y)) -> {count, reason, step}。
        # molmo/VLM 反复在同一个像素报同一个目标时，硬过滤禁止再次 propose；
        # 全部被过滤时向 VLM 报 ALL_SPOTS_REJECTED，引导其先靠近再看。
        self._rejected_spots = {}
        # geometry 解析失败候选的重看导航：frame_id -> {point, attempts, step}。
        # 系统把 agent 导航到该帧拍摄位姿附近重新观察，不再直接丢弃。
        self._revisit_targets = {}
        self.revisit_max_attempts = max(1, int(os.environ.get(
            "NAV_REVISIT_MAX_ATTEMPTS", "3")))
        # 决策层状态：近期事件流 + 最近一次探索规划的 frontier 缓存
        self._events = []
        self._last_frontier_clusters = []
        self._explore_grid = None
        # frontier 候选、彩色点云与鸟瞰图必须来自同一原子快照。
        self._frontier_grid = None
        self._frontier_layers = None
        # RGB point cloud captured atomically with the grid/pose snapshot.  It
        # is the only base layer sent to the decision VLM.
        self._frontier_pointcloud = None
        self.decision_map_max_points = max(10000, int(os.environ.get(
            "NAV_DECISION_MAP_MAX_POINTS", "2000000")))
        self.decision_map_point_stride = max(1, int(os.environ.get(
            "NAV_DECISION_MAP_POINT_STRIDE", "3")))
        self._frontier_revision = 0
        self._frontier_trajectory = []
        self._frontier_pose = None
        self._frontier_slam_pose = None
        self._frontier_scale = None
        self._metric_snapshot = {"revision": 0, "scale": None,
                                 "source": None, "pending": None,
                                 "pending_count": 0}
        self._frontier_server_revision = None
        self._frontier_stats = {}
        self._last_decision_snapshot_step = -10 ** 9
        self._last_candidate_refresh_step = -10 ** 9
        self._semantic_coverage_warned = False
        self._last_finish_decision_step = -10 ** 9
        # 前沿引导探索状态
        self._explore_follower = None
        self._last_explore_plan = -10 ** 9
        self.explore_replan_interval = int(
            os.environ.get("NAV_EXPL_REPLAN_INTERVAL", "25"))
        # 途中决策间隔：follower 活跃时每 N 步咨询一次 VLM（en_route 事件），
        # continue 保持当前路径；选其他动作即结束本次导航规划。
        self.en_route_decision_interval = max(1, int(os.environ.get(
            "NAV_EN_ROUTE_INTERVAL", "5")))
        self._last_en_route_step = -10 ** 9
        self.decision_map_refresh_interval = max(1, int(os.environ.get(
            "NAV_DECISION_MAP_REFRESH_INTERVAL", "5")))
        self.map_max_instances = max(1, int(os.environ.get(
            "NAV_MAP_MAX_INSTANCES", "12")))
        # 实例化去重：新观测 3D 点该半径（米）内有已有实例时不直接新建，
        # 挂起为 duplicate_review，由决策 VLM 看证据图裁决（resolve_duplicate）。
        self.instance_dup_radius_m = float(os.environ.get(
            "NAV_INSTANCE_DUPLICATE_RADIUS_M", "3.0"))
        self.explore_enabled = os.environ.get(
            "NAV_FRONTIER_EXPLORE", "1") == "1"
        self._frontier_empty_streak = 0
        self._last_frontier_count = None
        self._last_reachable_frontier_count = None
        self._last_frontier_step = -10 ** 9
        self._recent_frontiers = []
        self._frontier_failures = {}
        self._active_frontier_key = None
        self._frontier_branches = []
        self._branch_outcomes = {}
        self._active_branch_key = None
        self._explore_recovery_queue = []
        self._explore_recovery_stage = 0
        self._explore_recovery_count = 0
        self._frontier_exhausted_reported = False
        self._last_map_submaps = 0
        self._last_map_growth_step = 0
        self.frontier_cooldown_steps = int(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_STEPS", "100"))
        self.frontier_cooldown_m = float(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_M", "1.0"))
        # 轮内合并半径：代表点距离小于该值的候选合并为一个物理目标。
        # 0 表示禁用合并。
        self.frontier_merge_radius_m = float(os.environ.get(
            "NAV_FRONTIER_MERGE_RADIUS_M", "1.5"))
        self.semantic_max_range_m = float(os.environ.get(
            "NAV_SEMANTIC_RANGE_M", "4.0"))
        self.semantic_close_range_m = float(os.environ.get(
            "NAV_SEMANTIC_CLOSE_RANGE_M", "2.0"))
        self.semantic_min_views = int(os.environ.get(
            "NAV_SEMANTIC_MIN_VIEWS", "2"))
        self.semantic_min_view_angle_deg = float(os.environ.get(
            "NAV_SEMANTIC_MIN_VIEW_ANGLE_DEG", "25"))
        self.semantic_min_view_baseline_m = float(os.environ.get(
            "NAV_SEMANTIC_MIN_VIEW_BASELINE_M", "0.5"))
        self.geometry_gain_weight = float(os.environ.get(
            "NAV_FRONTIER_GEOMETRY_WEIGHT", "1.0"))
        self.semantic_gain_weight = float(os.environ.get(
            "NAV_FRONTIER_SEMANTIC_WEIGHT", "1.0"))

    def reset(self):
        super().reset()
        self._nav_reset_state()

    def _finalize(self):
        """episode 收尾：先按原逻辑保存 SLAM 轨迹，再渲染最终带路径鸟瞰图。

        时机在 ``MappingAgent.reset`` 的 ``reset_map`` 之前（以及进程退出
        atexit），此时点云和位姿仍在 mapping server 上，可重建栅格并叠加
        完整轨迹。atexit 时 client 可能已断，整体兜住异常。
        """
        episode = self.episode_id
        super()._finalize()
        if episode is None:
            return
        try:
            self._render_final_trajectory_map(episode)
        except Exception as exc:
            print(f"[NavAgent] 最终轨迹图渲染失败: {exc}")

    def _render_final_trajectory_map(self, episode):
        """渲染 episode 全轨迹 + 占用覆盖诊断图并保存。

        与 ``render_pointcloud_topdown``（决策图，刻意无轨迹）不同，这里
        输出 occupancy 区域着色 + 红色 SLAM 关键帧轨迹 + 实例/前沿标注，
        供离线检查漫游空转、覆盖盲区和卡死形态。
        """
        from agents import navigator as nav
        from agents.map_render import render_topdown
        frames = self.client.get_frame_points(
            stride=self.decision_map_point_stride)
        if not frames:
            return
        pose_by_frame = {}
        for frame in frames:
            pose_by_frame[int(frame.get("frame_id", -1))] = np.asarray(
                frame["pose"], dtype=np.float64)
        frame_ids = np.asarray(sorted(pose_by_frame), dtype=np.int64)
        if len(frame_ids) < 8:
            return
        align_R = self.align_R
        if align_R is None:
            poses64 = np.stack(
                [pose_by_frame[int(fid)] for fid in frame_ids])
            align_R = nav.gravity_alignment(
                poses64, cam_up=nav.mount_compensated_cam_up())
        grid = nav.OccupancyGrid.from_frame_points(frames, align_R)
        if grid is None:
            return
        trajectory = list(self._frontier_trajectory)
        pose_xy = None
        try:
            poses, pose_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 1 and pose_ids is not None:
                order = np.argsort(pose_ids)
                centers = np.asarray(poses)[order][:, 0:3, 3] @ align_R.T
                trajectory = [tuple(point[:2]) for point in centers]
                slam_x, slam_y, slam_yaw = nav.pose_to_yaw_2d(
                    np.asarray(poses)[order][-1], align_R)
                pose_xy = (slam_x, slam_y, slam_yaw)
        except Exception as exc:
            print(f"[NavAgent] 最终轨迹位姿拉取失败: {exc}")
        instances = [{"id": nd.iid, "xy": tuple(nd.point[:2]),
                      "reported": nd.reported}
                     for nd in getattr(self.memory, "nodes", [])]
        png = render_topdown(
            grid, trajectory=trajectory, pose=pose_xy,
            instances=instances,
            frontier_stats=self._frontier_stats,
            step=getattr(self, "_last_frontier_step", None),
            map_revision=getattr(self, "_frontier_revision", None),
            show_legend=True)
        safe_episode = "".join(
            ch if ch.isalnum() or ch in "-_." else "_"
            for ch in str(episode))[:120] or "unknown"
        out = os.path.join(
            self.output_dir, f"final_topdown_{safe_episode}.png")
        with open(out, "wb") as fp:
            fp.write(png)
        print(f"[NavAgent] 最终轨迹图已保存: {out} "
              f"(trajectory_points={len(trajectory)})")

    def _target_phrase(self, observation):
        override = os.environ.get("NAV_TARGET")
        if override:
            return override
        source = str(observation.goal_text or "").strip()
        # 只从公开 instruction 确定性去掉命令和数量；完整原文仍进入统一
        # world_state.task.goal，供 caption/pointing/VLM 做属性与关系判断。
        text = source
        text = re.sub(
            r"^(?:please\s+)?(?:find|locate|look\s+for|navigate\s+to|go\s+to)\s+",
            "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:any|a|an|the)\s+", "", text,
                      flags=re.IGNORECASE)
        # 数量词描述完成条件，不属于视觉目标短语，例如
        # "exactly two baskets" -> "baskets"。
        text = re.sub(
            r"^(?:(?:exactly|at\s+least|at\s+most)\s+)?"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+",
            "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:all|every)\s+(?:the\s+)?", "", text,
                      flags=re.IGNORECASE)
        fallback = text.strip().rstrip(".!?") or source
        return fallback

    def _explore_action(self, observation):
        """前沿引导探索；碰撞恢复最优先，构建失败回退随机游走。"""
        if (not self._last_motion_failed and
                getattr(observation, "previous_action", None) ==
                int(Action.MOVE_FORWARD) and self._active_branch_key):
            outcome = self._branch_outcomes.setdefault(
                self._active_branch_key, {"collisions": 0, "moves": 0,
                                          "keyframes": 0, "last_step": 0})
            outcome["moves"] += 1
            outcome["last_step"] = int(observation.step_count)
            if self._last_feed_info.get("is_keyframe"):
                outcome["keyframes"] += 1
        if self._last_motion_failed:
            if self._active_frontier_key is not None:
                key = self._active_frontier_key
                self._frontier_failures[key] = \
                    self._frontier_failures.get(key, 0) + 1
                self._log_event(
                    f"frontier navigation failed {key} "
                    f"count={self._frontier_failures[key]}")
            if self._active_branch_key is not None:
                outcome = self._branch_outcomes.setdefault(
                    self._active_branch_key, {"collisions": 0, "moves": 0,
                                              "keyframes": 0, "last_step": 0})
                outcome["collisions"] += 1
                outcome["last_step"] = int(observation.step_count)
            self._block_failed_nav_direction(
                observation, follower=self._explore_follower)
            self._explore_recovery_count += 1
            turn = (int(Action.TURN_LEFT) if self._explore_recovery_count % 2
                    else int(Action.TURN_RIGHT))
            self._explore_recovery_queue = [turn] * self.nav_escape_turns
            self._explore_recovery_stage = 1
            self._active_frontier_key = None
            self._active_branch_key = None
            self._explore_follower = None      # 碰撞后旧路径不可信
        if self._explore_recovery_queue:
            return self._explore_recovery_queue.pop(0)
        if self._explore_recovery_stage == 1:
            self._explore_recovery_stage = 0
            self._plan_exploration(observation, select=True)
            action = self._explore_follow(observation)
            if action is not None:
                return action
        if not self.explore_enabled:
            return super()._explore_action(observation)
        # 途中决策：follower 活跃时每 en_route_decision_interval 步咨询一次
        # VLM。选 CONTINUE_NAVIGATION 保持当前路径；选其他动作即结束本次
        # 导航规划（follower 在 _en_route_decision 内清理）。
        if self._explore_follower is not None and self.decision_loop is not None:
            action = self._en_route_decision(observation)
            if action is not None:
                return action
        # 活跃探索路径：跟随；走完后立即尝试选新目标
        action = self._explore_follow(observation)
        if action is not None:
            return action
        if observation.step_count - self._last_explore_plan \
                >= self.explore_replan_interval:
            if self.decision_loop is not None:
                return self._choose_high_level_target(
                    observation, "world_state_updated")
            self._plan_exploration(observation)
            action = self._explore_follow(observation)
            if action is not None:
                return action
        return super()._explore_action(observation)

    # 前沿引导探索
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
        if action is None and not arrived and self._active_frontier_key is not None:
            key = self._active_frontier_key
            self._frontier_failures[key] = self._frontier_failures.get(key, 0) + 1
            self._log_event(
                f"frontier path lost {key} count={self._frontier_failures[key]}")
        if arrived or action is None:
            self._explore_follower = None
            self._active_frontier_key = None
            self._active_branch_key = None
            # 到达短 frontier 后保留正常重规划间隔，避免每一步重新选择
            # 同一个已经到达的边界。
            self._last_explore_plan = observation.step_count
            return None
        fl.dead_reckon(int(action))
        return int(action)

    def _en_route_decision(self, observation):
        """follower 活跃期间的周期性完整决策（event=en_route）。

        输入与正常决策一致（当前 RGB + 新 topdown 图；图中橙色 ACTIVE 星
        标记当前导航目标，即使该 frontier 已从 fN 候选表消失）。选
        CONTINUE_NAVIGATION 保持当前路径；选任何其他动作即放弃本次导航
        规划，按正常流程执行。决策层不可用/失败时静默继续跟随，绝不阻塞。
        """
        if observation.step_count - self._last_en_route_step \
                < self.en_route_decision_interval:
            return None
        self._last_en_route_step = observation.step_count
        try:
            result, action = self._decider_next(observation, "en_route")
        except Exception as exc:
            self._log_event(f"en_route decision failed: {exc}")
            return None
        if result is None:
            self._log_event("en_route decision unavailable; keep following")
            return None
        if result.action == "CONTINUE_NAVIGATION":
            self._log_event(
                "en_route continue -> "
                f"{result.target_id or 'active frontier'}")
            # 路径可能刚走完/被碰撞清理；continue 落空由外层重规划。
            return self._explore_follow(observation)
        # 选其他动作 = 结束本次导航规划。注意 GOTO_INSTANCE / GOTO_FRONTIER
        # / EXPLORE 已在 _decider_next 内部重建导航状态（GOTO_INSTANCE 清掉
        # 旧 follower 并切 mode="nav"；GOTO_FRONTIER / EXPLORE 激活了新的
        # frontier follower），这里再清会误杀刚选的新目标、丢回随机游走。
        if result.action in ("SCAN", "START_ADJUST", "FINISH",
                             "REPORT_FOUND"):
            self._explore_follower = None
            self._active_frontier_key = None
        self._last_explore_plan = observation.step_count
        if result.action == "SCAN":
            # SCAN 由 _decider_next 返回 action=None，在此显式启动环视。
            self._scanning = True
            self._scan_steps = 0
            self._scan_images = []
            return int(Action.TURN_LEFT)
        return action

    def _plan_exploration(self, observation, select=True):
        """重建栅格并刷新可达 frontier；select=False 时只更新候选。"""
        self._last_explore_plan = observation.step_count
        try:
            # 点云、位姿和 frame_id 必须来自同一个服务端锁内快照。过去
            # 先拉 poses 再拉 frame points，回环可能在两次 RPC 中间改写
            # 坐标系，导致轨迹穿墙、当前位置落在 unknown。
            # Stride 3 preserves four times as many image-grid samples as the
            # previous stride 6. The renderer bins them into an orthographic
            # RGB image, while this same atomic snapshot still drives geometry.
            frames = self.client.get_frame_points(
                stride=self.decision_map_point_stride)
            if not frames:
                return False
            pose_by_frame = {}
            for frame in frames:
                pose_by_frame[int(frame.get("frame_id", -1))] = np.asarray(
                    frame["pose"], dtype=np.float64)
            frame_ids = np.asarray(sorted(pose_by_frame), dtype=np.int64)
            if len(frame_ids) < 8:
                return False
            poses64 = np.stack([pose_by_frame[int(fid)] for fid in frame_ids])
            if self.align_R is None:
                self.align_R = nav.gravity_alignment(
                    poses64, cam_up=nav.mount_compensated_cam_up())
            self._update_pool_slam_anchor(poses64[0])
            grid = nav.OccupancyGrid.from_frame_points(
                frames, self.align_R)
            if grid is None:
                return False
            # Deduplicate overlap frames exactly as OccupancyGrid does, then
            # retain a bounded, color-aligned point snapshot for VLM rendering.
            colored_frames = {}
            for frame in frames:
                if frame.get("colors") is not None:
                    colored_frames[int(frame.get("frame_id", -1))] = frame
            point_parts, color_parts = [], []
            for fid in sorted(colored_frames):
                frame = colored_frames[fid]
                points = np.asarray(frame["points"], dtype=np.float64)
                colors = np.asarray(frame["colors"])
                finite = np.isfinite(points).all(axis=1)
                if len(colors) == len(points) and finite.any():
                    point_parts.append(points[finite] @ self.align_R.T)
                    color_parts.append(colors[finite])
            if point_parts:
                map_points = np.concatenate(point_parts)
                map_colors = np.concatenate(color_parts)
                if len(map_points) > self.decision_map_max_points:
                    keep = np.linspace(
                        0, len(map_points) - 1,
                        self.decision_map_max_points, dtype=np.int64)
                    map_points, map_colors = map_points[keep], map_colors[keep]
                self._frontier_pointcloud = (map_points, map_colors)
            else:
                self._frontier_pointcloud = None
            try:
                semantic_enabled, captioned_ids = \
                    self.client.get_captioned_frame_ids()
            except Exception as exc:
                # 新旧 mapping server 协议不匹配时显式记录；保留纯几何层，
                # 不把不存在的 caption 覆盖伪装成已完成语义探索。
                semantic_enabled, captioned_ids = False, []
                if not self._semantic_coverage_warned:
                    self._log_event(
                        f"semantic coverage unavailable: {exc}")
                    self._semantic_coverage_warned = True
            if semantic_enabled:
                grid.update_semantic_coverage(
                    frames, captioned_ids, self.align_R,
                    max_range_m=self.semantic_max_range_m,
                    close_range_m=self.semantic_close_range_m,
                    min_views=self.semantic_min_views,
                    min_view_angle_deg=self.semantic_min_view_angle_deg,
                    min_view_baseline_m=self.semantic_min_view_baseline_m)
            elif not self._semantic_coverage_warned:
                self._log_event("semantic coverage disabled by mapping server")
                self._semantic_coverage_warned = True
            self._explore_grid = grid
        except Exception as e:
            print(f"[NavAgent] 探索栅格构建失败: {e}")
            return False

        order = np.argsort(frame_ids)
        latest_fid = int(frame_ids[order][-1])
        latest_pose = poses64[order][-1]
        cur = latest_pose[:3, 3] @ self.align_R.T
        # 尺度：栅格尺规（相机离地 1.5m）比在线动作标定更直接，也必须
        # 与当前鸟瞰图使用同一份尺度。
        raw_scale = 1.0 / grid.unit_per_m if grid.unit_per_m > 0 else None
        self._frontier_scale = self._update_metric_snapshot(
            raw_scale or self.calibrator.current_scale() or 1.0,
            source="grid" if raw_scale else "calibrator")
        scale = self._frontier_scale
        slam_x, slam_y, slam_yaw = nav.pose_to_yaw_2d(
            latest_pose, self.align_R)
        self._frontier_slam_pose = (slam_x, slam_y, slam_yaw)
        self._frontier_pose = self._dead_reckon_snapshot_pose(
            latest_pose, latest_fid, scale=self._frontier_scale)
        centers = poses64[order, :3, 3] @ self.align_R.T
        self._frontier_trajectory = [tuple(point[:2])
                                     for point in centers[::4]]
        if self._frontier_pose is not None:
            live_xy = tuple(self._frontier_pose[:2])
            if not self._frontier_trajectory or np.linalg.norm(
                    np.asarray(live_xy) - np.asarray(
                        self._frontier_trajectory[-1])) > 1e-6:
                self._frontier_trajectory.append(live_xy)
        revision = getattr(
            self.client, "last_frame_snapshot_revision", None)
        self._frontier_server_revision = (
            dict(revision) if isinstance(revision, dict) else None)
        # 骨架拓扑只用于给实例附着可通行节点；语义判断交给 VLM。
        graph = skel.build_skeleton_graph(grid)
        if graph is not None:
            self.memory.attach_to_skeleton(graph)

        raw_clusters, frontier_layers = skel.frontier_clusters(
            grid, min_size=5, return_layers=True)
        # 轮内归并：同一物理位置（门洞内外环、几何/语义断簇）只保留
        # 一个候选，避免 VLM 看到一串冗余 frontier。
        merged_clusters = skel.merge_same_spot(
            raw_clusters, scale if scale else 1.0,
            radius_m=self.frontier_merge_radius_m)
        self._frontier_grid = grid
        self._frontier_layers = frontier_layers
        self._frontier_revision += 1
        try:
            num_submaps = int(self.client.get_state().get("num_submaps", 0))
            if num_submaps > self._last_map_submaps:
                self._last_map_growth_step = observation.step_count
                self._last_map_submaps = num_submaps
        except Exception:
            pass
        self._recent_frontiers = [
            item for item in self._recent_frontiers
            if observation.step_count - item[1] <= self.frontier_cooldown_steps
        ]
        # 全量候选先做距离/冷却/A* 可达性检查，再进入排序。对外暴露的
        # frontier、结束判断和 VLM 状态都只使用这个最终有效集合。
        reachable = []
        valid = []
        too_near_count = 0
        unreachable_count = 0
        cooldown_count = 0
        for c in merged_clusters:
            d_units = math.hypot(c["world"][0] - cur[0],
                                 c["world"][1] - cur[1])
            d_m = d_units * scale if scale else d_units
            if scale and d_m < 1.0:
                too_near_count += 1
                continue
            on_cooldown = bool(scale and any(
                    math.hypot(c["world"][0] - old_xy[0],
                               c["world"][1] - old_xy[1]) * scale
                    < self.frontier_cooldown_m
                    for old_xy, _old_step in self._recent_frontiers))
            path = grid.astar(cur[:2], c["world"])
            if path is None or len(path) < 2:
                unreachable_count += 1
                continue
            path_cost_units = sum(float(np.linalg.norm(
                np.asarray(path[i + 1]) - np.asarray(path[i])))
                for i in range(len(path) - 1))
            path_cost_m = path_cost_units * scale if scale else path_cost_units
            key = self._frontier_key(c["world"], scale)
            failures = self._frontier_failures.get(key, 0)
            gain = max(
                self.geometry_gain_weight * float(
                    c.get("geometry_gain", 0)) +
                self.semantic_gain_weight * float(
                    c.get("semantic_gain", 0)),
                1.0)
            score = gain / (1.0 + path_cost_m) / (1.0 + failures)
            item = dict(c)
            item.update({
                "path": grid.shortcut(path),
                "path_cost_m": float(path_cost_m),
                "failure_count": int(failures),
                "utility": float(score),
                "key": key,
                "on_cooldown": on_cooldown,
                "scale": float(scale or 1.0),
            })
            reachable.append(item)
            if not on_cooldown:
                valid.append(item)
            else:
                cooldown_count += 1
        valid.sort(key=lambda c: -c["utility"])
        self._frontier_branches = self._summarize_frontier_branches(
            valid, cur[:2], grid, scale, observation.step_count)

        self._last_frontier_count = len(valid)
        self._last_reachable_frontier_count = len(reachable)
        self._frontier_stats = {
            "raw_clusters": len(raw_clusters),
            "merged_clusters": len(merged_clusters),
            "raw_boundary_cells": int(np.asarray(
                frontier_layers.get("unified", []), dtype=bool).sum()),
            "reachable": len(reachable),
            "selectable": len(valid),
            "filtered_too_near": too_near_count,
            "filtered_unreachable": unreachable_count,
            "filtered_cooldown": cooldown_count,
        }
        self._last_frontier_step = observation.step_count
        self._last_frontier_clusters = valid
        if valid:
            self._frontier_empty_streak = 0
            self._frontier_exhausted_reported = False
        elif not reachable:
            self._frontier_empty_streak += 1
            if not self._frontier_exhausted_reported:
                self._log_event("frontier_exhausted: no reachable frontier")
                self._frontier_exhausted_reported = True
            return True
        else:
            # 仍有可达 frontier，只是处于短期冷却；不能计为探索耗尽。
            self._frontier_empty_streak = 0
            self._frontier_exhausted_reported = False
            return True

        if not select:
            return True

        return self._activate_frontier(
            observation, valid[0], source="deterministic explorer")

    def _frontier_key(self, world_xy, scale):
        """稳定的米制空间桶，用于累计 frontier 导航失败惩罚。"""
        bucket_m = max(self.frontier_cooldown_m, 0.25)
        x_m = float(world_xy[0]) * (scale or 1.0)
        y_m = float(world_xy[1]) * (scale or 1.0)
        return (int(round(x_m / bucket_m)), int(round(y_m / bucket_m)))

    def _update_metric_snapshot(self, candidate, source):
        """Accept only stable metric-scale changes for all nav consumers."""
        try:
            candidate = float(candidate)
        except (TypeError, ValueError):
            return self._metric_scale_value() or 1.0
        if not math.isfinite(candidate) or candidate <= 0:
            return self._metric_scale_value() or 1.0
        snapshot = self._metric_snapshot
        current = snapshot.get("scale")
        if current is None:
            snapshot.update(scale=candidate, source=str(source), revision=1,
                            pending=None, pending_count=0)
            return candidate
        relative = abs(candidate - current) / max(abs(current), 1e-6)
        if relative <= 0.12:
            snapshot.update(scale=candidate, source=str(source),
                            pending=None, pending_count=0)
            return candidate
        pending = snapshot.get("pending")
        if pending is not None and abs(candidate - pending) / max(
                abs(pending), 1e-6) <= 0.12:
            snapshot["pending_count"] = int(snapshot.get("pending_count", 0)) + 1
        else:
            snapshot["pending"] = candidate
            snapshot["pending_count"] = 1
        if snapshot["pending_count"] >= 3:
            snapshot.update(scale=candidate, source=str(source),
                            revision=int(snapshot.get("revision", 0)) + 1,
                            pending=None, pending_count=0)
            self._invalidate_metric_navigation()
            self._log_event(
                f"metric scale snapshot switched to {candidate:.3f} ({source})")
        else:
            self._log_event(
                f"metric scale candidate deferred {candidate:.3f}; "
                f"current={current:.3f}")
        return float(snapshot["scale"])

    def _invalidate_metric_navigation(self):
        """Discard geometry whose metres-per-unit contract just changed."""
        # Paths, temporary obstacle radii and follower reach checks all embed
        # a scale. Keep the semantic target, but rebuild every geometric
        # artifact from one fresh mapping snapshot before issuing another move.
        self.follower = None
        self.grid = None
        self._nav_recovery_queue = []
        self._nav_recovery_stage = 0
        self._nav_blocked_points = []
        self._explore_follower = None
        self._explore_recovery_queue = []
        self._explore_recovery_stage = 0
        self._active_frontier_key = None
        self._active_branch_key = None
        self._metric_replan_required = True

    def _metric_scale_value(self):
        scale = self._metric_snapshot.get("scale")
        if scale is not None:
            return float(scale)
        try:
            scale = self.calibrator.current_scale()
            return float(scale) if scale is not None else None
        except Exception:
            return None

    def _summarize_frontier_branches(self, frontiers, start_xy, grid, scale,
                                     step):
        """Group current frontiers by their local A* prefix, not room labels.

        VGGT global reconstruction is not trusted enough for persistent room
        segmentation.  A branch is an ephemeral route option on this snapshot;
        its short-lived ledger prevents choosing the same blocked corridor over
        and over without inventing semantic place names or fixed map markers.
        """
        groups = {}
        prefix_units = 1.0 / max(float(scale or 1.0), 1e-6)
        for frontier in frontiers:
            path = frontier.get("path") or []
            if len(path) < 2:
                continue
            walked, pivot = 0.0, np.asarray(path[-1], dtype=np.float64)
            previous = np.asarray(path[0], dtype=np.float64)
            for point in path[1:]:
                point = np.asarray(point, dtype=np.float64)
                walked += float(np.linalg.norm(point - previous))
                previous = point
                if walked >= prefix_units:
                    pivot = point
                    break
            try:
                row, col = grid.world_to_cell(tuple(pivot[:2]))
                key = f"b{int(row // 3)}_{int(col // 3)}"
            except Exception:
                angle = math.atan2(pivot[1] - start_xy[1],
                                   pivot[0] - start_xy[0])
                key = f"b_angle_{int(round(angle / (math.pi / 4)))}"
            outcome = self._branch_outcomes.get(key, {})
            frontier["branch_id"] = key
            frontier["recently_attempted"] = bool(
                int(step) - int(outcome.get("last_step", -10 ** 9)) <=
                self.frontier_cooldown_steps)
            frontier["novelty"] = (
                "untried" if not outcome else
                ("blocked" if outcome.get("collisions", 0) else
                 "revisited"))
            bucket = groups.setdefault(key, {"id": key, "frontiers": [],
                                              "geometry_gain": 0,
                                              "semantic_gain": 0,
                                              "min_cost_m": None})
            bucket["frontiers"].append(frontier)
            bucket["geometry_gain"] += int(frontier.get("geometry_gain", 0))
            bucket["semantic_gain"] += int(frontier.get("semantic_gain", 0))
            cost = frontier.get("path_cost_m")
            if cost is not None:
                bucket["min_cost_m"] = (float(cost) if
                    bucket["min_cost_m"] is None else
                    min(float(cost), bucket["min_cost_m"]))
        rows = []
        for key, bucket in groups.items():
            outcome = self._branch_outcomes.get(key, {})
            rows.append({"id": key,
                         "frontier_ids": [f"f{i}" for i, item in
                                          enumerate(frontiers)
                                          if item.get("branch_id") == key],
                         "path_cost_m": (round(bucket["min_cost_m"], 2)
                                         if bucket["min_cost_m"] is not None else None),
                         "geometry_gain": bucket["geometry_gain"],
                         "semantic_gain": bucket["semantic_gain"],
                         "collision_count": int(outcome.get("collisions", 0)),
                         "successful_moves": int(outcome.get("moves", 0)),
                         "new_keyframes": int(outcome.get("keyframes", 0)),
                         "recently_attempted": bool(outcome),
                         "novelty": ("untried" if not outcome else
                                     ("blocked" if outcome.get("collisions", 0)
                                      else "revisited"))})
        return sorted(rows, key=lambda row: (
            row["novelty"] != "untried", row["path_cost_m"] is None,
            row["path_cost_m"] or float("inf")))

    def _activate_frontier(self, observation, cluster, source):
        """用生成该路径时的尺度启动 frontier follower。"""
        path = cluster.get("path")
        if not path or len(path) < 2:
            return False
        self._clear_current_target()
        self.mode = "explore"
        follower = nav.PathFollower(
            scale=cluster.get("scale", 1.0), reach_m=self.reach_m)
        follower.set_path(path)
        self._explore_follower = follower
        self._active_frontier_key = cluster.get("key")
        self._active_branch_key = cluster.get("branch_id")
        if self._active_branch_key is not None:
            outcome = self._branch_outcomes.setdefault(
                self._active_branch_key, {"collisions": 0, "moves": 0,
                                          "keyframes": 0, "last_step": 0})
            outcome["last_step"] = int(observation.step_count)
        self._recent_frontiers.append((
            np.asarray(cluster["world"], dtype=np.float64)[:2],
            observation.step_count))
        self._log_event(
            f"{source} -> frontier key={self._active_frontier_key} "
            f"reason={cluster.get('reason')} "
            f"cost={cluster.get('path_cost_m')} "
            f"utility={cluster.get('utility')}")
        return True

    def act(self, observation):
        self._feed_frame(observation)
        self._last_observation = observation
        self._settle_action_outcomes()
        if self._adjusting and not self._last_motion_failed and \
                getattr(observation, "previous_action", None) == \
                int(Action.MOVE_FORWARD):
            self._adjust_progress["successful_moves"] += 1
        frame_id = self._last_feed_info.get("frame_id")
        if self._adjusting and frame_id is not None and \
                self._adjust_start_frame_id is not None and \
                int(frame_id) > int(self._adjust_start_frame_id):
            self._adjust_progress["fresh_views"] += 1
            if self._last_feed_info.get("is_keyframe"):
                self._adjust_progress["new_keyframes"] += 1
        if hasattr(self.vlm, "set_trace_context"):
            self.vlm.set_trace_context(
                episode=str(observation.episode_id),
                step=int(observation.step_count),
                goal_text=str(observation.goal_text or ""))
        self._target_mode = str(observation.target_mode or "any").lower()
        self._target_count = observation.target_count
        self._capture_pool_world_anchor(observation)
        if self._last_motion_failed and \
                observation.previous_action == int(Action.MOVE_FORWARD):
            if self.follower is not None and self.follower.anchor_frame >= 0:
                self.follower.undo_dead_reckon(int(Action.MOVE_FORWARD))
            if self._adjusting and self._explore_follower is not None and \
                    self._explore_follower.anchor_frame >= 0:
                self._explore_follower.undo_dead_reckon(
                    int(Action.MOVE_FORWARD))

        if self._adjusting:
            action = self._adjustment_action(observation)
        elif self.mode == "explore" and self._should_finish(observation):
            action = int(Action.FINISH)
        elif self.mode == "reported":
            if self._should_finish(observation):
                action = int(Action.FINISH)
            else:
                self._clear_current_target()
                self.mode = "explore"
                action = self._choose_high_level_target(
                    observation, "world_state_updated")
        elif self.mode == "nav":
            if self._scanning:
                action = self._handle_scan(observation)
            else:
                if self._metric_replan_required:
                    self._metric_replan_required = False
                    if self._plan_to_target(observation):
                        action, arrived, stuck = self._nav_action(observation)
                    else:
                        action, arrived, stuck = None, False, False
                else:
                    action, arrived, stuck = self._nav_action(observation)
                if stuck:
                    # 连续碰撞无法到达：登记不可达并让决策 VLM 介入换目标
                    action = self._nav_failed_recovery(observation)
                elif arrived:
                    action = self._handle_arrival_transition(observation)
                elif action is None:    # 路径走丢，退回探索
                    self.mode = "explore"
                    action = self._explore_action(observation)
        else:
            self._periodic_anchor(observation)
            if self._scanning:
                # en_route 决策可启动 SCAN（explore 模式下同样消费转圈）。
                action = self._handle_scan(observation)
            else:
                action = self._explore_action(observation)

        self._record_and_update(observation, action)
        if self.follower is not None and self.follower.anchor_frame >= 0:
            self.follower.dead_reckon(action)
        elif self._adjusting and self._explore_follower is not None and \
                self._explore_follower.anchor_frame >= 0:
            # Adjustment bypasses _explore_follow(), so keep its pose estimate
            # synchronized with the atomic action executed by the VLM.
            self._explore_follower.dead_reckon(action)
        return action

    def _clear_current_target(self):
        self.target_point = None
        self.target_candidate_id = None
        self.target_instance_id = None
        self.follower = None
        self.grid = None
        self._explore_follower = None
        self._active_frontier_key = None
        self._active_branch_key = None
        self._explore_recovery_queue = []
        self._explore_recovery_stage = 0
        self._plan_failures = 0
        self._nav_recovery_queue = []
        self._nav_recovery_stage = 0
        self._nav_collision_streak = 0
        self._nav_stuck_replanned = False
        self._nav_blocked_points = []
        self._scanning = False
        self._scan_steps = 0
        self._scan_images = []
        self._selected_evidence = None

    def _handle_arrival_transition(self, observation):
        """Enter the arrival decision exactly once from every nav call path."""
        if self._arrival_transition_active:
            self._log_event(
                "suppressed recursive immediate arrival; returning to explore")
            self._clear_current_target()
            self.mode = "explore"
            return self._autonomous_explore_action(observation)
        self._mark_goto_arrived()
        self._arrival_transition_active = True
        try:
            result, decided_action = self._arrival_vlm_decision(observation)
        finally:
            self._arrival_transition_active = False
        if result is None:
            self._log_event(
                "arrival decision unavailable; leaving candidate without report")
            self._clear_current_target()
            self.mode = "explore"
            return self._explore_action(observation)
        if result.action == "REPORT_FOUND":
            return self._report_found(result.target_id)
        if result.action == "SCAN":
            self._scanning = True
            self._scan_steps = 0
            self._scan_images = []
            return int(Action.TURN_LEFT)
        if result.action == "EXPLORE":
            # _decider_next has already activated the selected frontier.
            return (decided_action if decided_action is not None
                    else self._autonomous_explore_action(observation))
        # GOTO_INSTANCE/GOTO_FRONTIER/FINISH are mapped by _decider_next.
        return (decided_action if decided_action is not None
                else self._explore_action(observation))

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

    # ------------------------------------------------------------------
    # 评测采集接口（只读旁路；benchmark 评测器在每步 act() 之后调用，
    # 统计已实例化但未上报目标 U_t 与发现池质量）。不写任何导航/
    # 决策状态，不发起网络/磁盘 IO，不改变 act() 行为。
    # ------------------------------------------------------------------
    def _capture_pool_world_anchor(self, observation):
        """记录 episode 首个有效 (gps, compass) 作为世界系锚点。

        gps 为 habitat 世界系绝对位置（米，y-up）；compass 为绕世界
        +Y 的右手 yaw（弧度），compass=0 时 agent 面向世界 -Z（与
        evaluation/main/evaluator.py 的 _agent_compass 四元数转 yaw
        约定一致）。只写 _pool_world_anchor，不影响其他状态。
        """
        if self._pool_world_anchor is not None:
            return
        gps = getattr(observation, "gps", None)
        compass = getattr(observation, "compass", None)
        if gps is None or compass is None:
            return
        try:
            gps = np.asarray(gps, dtype=np.float64).reshape(-1)
            compass = float(compass)
        except (TypeError, ValueError):
            return
        if gps.size < 3 or not np.isfinite(gps[:3]).all() \
                or not math.isfinite(compass):
            return
        self._pool_world_anchor = (gps[:3].copy(), compass)

    def _update_pool_slam_anchor(self, first_pose):
        """用最早关键帧位姿刷新 SLAM 侧锚点（重力对齐系，z-up）。

        位姿来自调用方已有的 RPC 快照（本方法自身不做 IO）。每次
        重规划刷新一次：回环改写历史位姿后，实例点与锚点仍处在
        同一版 SLAM 坐标系里。
        """
        align_R = getattr(self, "align_R", None)
        if align_R is None or first_pose is None:
            return
        try:
            pose = np.asarray(first_pose, dtype=np.float64)
            pos = align_R @ pose[:3, 3]
            _x, _y, yaw = nav.pose_to_yaw_2d(pose, align_R)
        except Exception:
            return
        if not np.isfinite(pos).all() or not math.isfinite(yaw):
            return
        self._pool_slam_anchor = (float(pos[0]), float(pos[1]),
                                  float(pos[2]), float(yaw))

    def _pool_metric_scale(self):
        """地图单位 -> 米。无可靠估计返回 None（此时池坐标无意义）。"""
        scale = self._metric_scale_value()
        if scale is not None and math.isfinite(float(scale)) \
                and float(scale) > 0:
            return float(scale)
        grid = getattr(self, "_frontier_grid", None)
        unit_per_m = float(getattr(grid, "unit_per_m", 0.0) or 0.0)
        if math.isfinite(unit_per_m) and unit_per_m > 0:
            return 1.0 / unit_per_m
        return None

    def get_target_pool(self):
        """评测采集接口：当前 episode 全部 canonical instance 的世界坐标。

        契约（与 benchmark 评测器约定）：list[dict]，每项
        {"position": [x, y, z], "reported": bool, "label": str}。
        position 为 habitat 世界系坐标（米，y-up）；label 为
        InstanceNode.text 截断 100 字符；包含已上报实例（reported
        标志区分）。变换未建立（无锚点或无尺度）时返回 []，绝不
        抛异常。每次调用现算（实例点随回环刷新），O(实例数)，无
        任何网络/磁盘 IO。

        坐标变换（近似相似变换，推导见 AGENT_ARCHITECTURE.md §12）：
        对齐 SLAM 系为右手 z-up，habitat 世界系为右手 y-up；两者
        水平面基序 (x_s, y_s) 与 (x_w, z_w) 手性相反，因此平面
        映射是反射+旋转而非纯旋转：
            ψ_w = atan2(−cos c0, −sin c0)   # 世界 forward 的平面角
            α   = ψ_w + yaw_s0
            dx, dy = p_xy − anchor_xy
            wx = g0x + s·(cosα·dx + sinα·dy)
            wy = g0y + s·(p_z − anchor_z)
            wz = g0z + s·(sinα·dx − cosα·dy)
        """
        try:
            world_anchor = getattr(self, "_pool_world_anchor", None)
            slam_anchor = getattr(self, "_pool_slam_anchor", None)
            if world_anchor is None or slam_anchor is None:
                return []
            scale = self._pool_metric_scale()
            if scale is None:
                return []
            g0, compass0 = world_anchor
            ax, ay, az, yaw_s0 = slam_anchor
            psi_w = math.atan2(-math.cos(compass0), -math.sin(compass0))
            alpha = psi_w + yaw_s0
            cos_a, sin_a = math.cos(alpha), math.sin(alpha)
            pool = []
            for node in self.memory.nodes:
                point = np.asarray(node.point, dtype=np.float64).reshape(-1)
                if point.size < 3 or not np.isfinite(point[:3]).all():
                    continue
                dx, dy = point[0] - ax, point[1] - ay
                wx = g0[0] + scale * (cos_a * dx + sin_a * dy)
                wy = g0[1] + scale * (point[2] - az)
                wz = g0[2] + scale * (sin_a * dx - cos_a * dy)
                pool.append({
                    "position": [float(wx), float(wy), float(wz)],
                    "reported": bool(node.reported),
                    "label": str(node.text or "")[:100],
                })
            return pool
        except Exception:
            return []

    def _refresh_memory_candidates(self, instance_ids=None):
        """回环后刷新 Observation 坐标，再重选 canonical 导航点。"""
        if not self._ensure_alignment():
            return
        selected = None if instance_ids is None else {
            int(iid) for iid in instance_ids}
        nodes = [node for node in self.memory.nodes
                 if selected is None or node.iid in selected]
        if not nodes:
            return
        candidate_to_observations = {}
        legacy_candidates = {}
        for node in nodes:
            observations = [self.memory.get_observation(oid)
                            for oid in node.observation_ids]
            observations = [obs for obs in observations
                            if obs is not None and obs.candidate_id]
            for observed in observations:
                candidate_to_observations.setdefault(
                    str(observed.candidate_id), []).append(observed)
            if not observations and node.candidate_id:
                legacy_candidates.setdefault(
                    str(node.candidate_id), []).append(node)
        candidate_ids = list(dict.fromkeys(
            [*candidate_to_observations, *legacy_candidates]))
        if not candidate_ids:
            return

        def apply_resolved(candidate_id, resolved):
            if not resolved.get("found"):
                return
            point = self._aligned_point(resolved["point"])
            for observed in candidate_to_observations.get(
                    str(candidate_id), []):
                self.memory.refresh_observation_point(observed, point)
            for node in legacy_candidates.get(str(candidate_id), []):
                self.memory.refresh_point(node, point)

        batch = getattr(self.client, "resolve_candidates", None)
        if callable(batch):
            try:
                resolved_rows = batch(candidate_ids)
                for candidate_id, resolved in resolved_rows.items():
                    apply_resolved(candidate_id, resolved)
                return
            except Exception:
                # 兼容尚未更新的 mapping server，退回逐个协议。
                pass
        for candidate_id in candidate_ids:
            try:
                resolved = self.client.resolve_candidate(candidate_id)
                apply_resolved(candidate_id, resolved)
            except Exception:
                pass

    def _report_found(self, instance_id=None):
        """Atomically claim and report the active canonical instance."""
        node = self.memory.get(instance_id)
        if node is None or node.iid != self.target_instance_id:
            self._log_event("ignored REPORT_FOUND not bound to active instance")
            return int(Action.TURN_LEFT)
        step = int(getattr(self._last_observation, "step_count", 0) or 0)
        claim = self.memory.claim(node, step=step)
        if claim is None:
            self._log_event("ignored duplicate/invalid TARGET_FOUND report")
            self._clear_current_target()
            self.mode = "explore"
            self._scanning = False
            return int(Action.TURN_LEFT)
        self._reported_count += 1
        self._no_hit_queries = 0
        self._log_event(
            f"ReportClaim {claim.claim_id}: instance {node.iid} -> "
            f"TARGET_FOUND (total {self._reported_count})")
        self.mode = "reported"
        self._scanning = False
        return int(Action.TARGET_FOUND)

    # Phase 4：VLM 决策层（NAV_DECIDER=vlm，事件驱动，不进控制回路）
    def _log_event(self, message):
        self._events.append(str(message))
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def _tool_search_frames(self, query, top_k=5):
        """决策层只读工具：caption 语义记忆检索（纯检索，不 ground）。"""
        query = str(query or "").strip()
        if not query:
            return {"error": "query must not be empty"}
        try:
            limit = min(20, max(1, int(top_k)))
        except (TypeError, ValueError):
            return {"error": "top_k must be an integer"}
        try:
            results = self.client.retrieve_captions(query, top_k=limit)
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return [{"frame_id": r.get("frame_id"),
                 "score": round(float(r.get("score", 0.0)), 3),
                 "caption": str(r.get("caption", ""))[:300]}
                for r in results]

    def _tool_search_instances(self, query, reported=None, top_k=10):
        """按VLM给出的关键词直接检索实例text；OR匹配，命中数排序。"""
        if isinstance(query, str):
            raw = query.replace(",", " ").split()
        elif isinstance(query, (list, tuple)):
            raw = query
        else:
            return {"error": "query must be a non-empty string or string array"}
        terms = []
        for value in raw:
            term = str(value or "").strip().lower()
            if term and term not in terms:
                terms.append(term)
        if not terms:
            return {"error": "query must not be empty"}
        if reported is not None and not isinstance(reported, bool):
            return {"error": "reported must be true, false, or null"}
        try:
            limit = min(20, max(1, int(top_k)))
        except (TypeError, ValueError):
            return {"error": "top_k must be an integer"}
        rows = []
        for node in self.memory.nodes:
            if reported is not None and node.reported != reported:
                continue
            haystack = node.text.lower()
            matched = [term for term in terms if term in haystack]
            if not matched:
                continue
            rows.append({
                "id": node.iid,
                "text": node.text,
                "reported": node.reported,
                "observation_count": len(node.observation_ids),
                "report_claim_id": node.report_claim_id,
                "matched_keywords": matched,
                "evidence_count": len(node.evidence),
                "frame_ids": [item.get("frame_id")
                              for item in node.evidence
                              if item.get("frame_id") is not None],
            })
        rows.sort(key=lambda row: (-len(row["matched_keywords"]), row["id"]))
        return rows[:limit]

    def _tool_view_instance(self, instance_id):
        """返回实例最相关的证据图；优先pointing overlay，回退关键帧。"""
        node = self.memory.get(instance_id)
        if node is None:
            return None
        candidate_ids = [node.candidate_id] + [
            item.get("candidate_id") for item in reversed(node.evidence)]
        seen = set()
        for candidate_id in candidate_ids:
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            try:
                meta, payload = self.client.get_candidate_evidence(candidate_id)
                if meta.get("found") and payload:
                    return payload
            except Exception:
                pass
        frame_ids = [node.frame_id] + [
            item.get("frame_id") for item in reversed(node.evidence)]
        seen.clear()
        for frame_id in frame_ids:
            if frame_id is None or frame_id in seen:
                continue
            seen.add(frame_id)
            try:
                meta, payload = self.client.get_frame_image(frame_id)
                if meta.get("found") and payload:
                    return payload
            except Exception:
                pass
        return None

    @staticmethod
    def _instance_tool_view(node):
        if node is None:
            return None
        return {
            "id": node.iid,
            "point": [round(float(value), 3) for value in node.point],
            "text": node.text,
            "reported": node.reported,
            "frame_id": node.frame_id,
            "candidate_id": node.candidate_id,
            "evidence": list(node.evidence),
            "observation_ids": list(node.observation_ids),
            "report_claim_id": node.report_claim_id,
        }

    def _tool_get_instance(self, instance_id):
        node = self.memory.get(instance_id)
        if node is None:
            return {"error": f"instance {instance_id!r} not found"}
        return self._instance_tool_view(node)

    def _tool_update_instance(self, instance_id, text):
        node = self.memory.update_text(instance_id, text)
        if node is None:
            return {"error": f"instance {instance_id!r} not found"}
        self._log_event(f"VLM updated instance {node.iid}: {node.text[:120]}")
        return self._instance_tool_view(node)

    # harness：动作流水（决策结果被应用时记录；执行层在下一步结算 outcome）
    def _record_action(self, action, target_id=None):
        obs = self._last_observation
        step = int(getattr(obs, "step_count", 0) or 0)
        self._action_log.append({
            "step": step,
            "action": str(action),
            "target_id": (None if target_id is None else str(target_id)),
            "outcome": None,
        })
        if len(self._action_log) > 500:
            self._action_log = self._action_log[-500:]

    def _settle_action_outcomes(self):
        """act() 开头结算所有未收尾记录：collision 优先，否则 ok。"""
        pending = [e for e in self._action_log if e.get("outcome") is None]
        if not pending:
            return
        for entry in pending:
            entry["outcome"] = "ok"
        if self._last_motion_failed:
            pending[-1]["outcome"] = "collision"

    def _mark_goto_arrived(self):
        """到达事件发生：把最近一条 GOTO_* 记录结算为 arrived。"""
        for entry in reversed(self._action_log):
            if entry["action"] in ("GOTO_INSTANCE", "GOTO_FRONTIER"):
                entry["outcome"] = "arrived"
                break

    # harness：新增决策工具
    def _tool_view_frame(self, frame_id):
        """查看指定关键帧的原始 RGB（JPEG bytes；失败返回 None）。"""
        try:
            meta, payload = self.client.get_frame_image(int(frame_id))
        except Exception:
            return None
        if not meta.get("found") or not payload:
            return None
        return payload

    @staticmethod
    def _ground_rows(changed):
        return [{
            "instance_id": row["instance_id"],
            "observation_id": row.get("observation_id"),
            "frame_id": row["frame_id"],
            "confidence": row["confidence"],
            "association": row.get("association"),
            "reported": bool(row.get("reported", False)),
        } for row in changed]

    @staticmethod
    def _pointing_error(response):
        """Preserve a stable backend error code across mapping RPC/tools."""
        if not isinstance(response, dict) or not response.get("error"):
            return None
        return {"error": {
            "code": str(response.get("error_code") or "TOOL_ERROR"),
            "message": str(response.get("error"))[:300],
        }}

    @staticmethod
    def _crosshair_key(frame_id, pixel_norm):
        """归一化十字坐标的稳定 key；允许 VLM 一位小数重发。"""
        return (int(frame_id), round(float(pixel_norm[0]), 1),
                round(float(pixel_norm[1]), 1))

    def _find_crosshair_key(self, frame_id, pixel_norm, source):
        """查找同一证据点，容忍 VLM 对 0--1000 坐标的 ±2 四舍五入。"""
        try:
            fid = int(frame_id)
            x, y = float(pixel_norm[0]), float(pixel_norm[1])
        except (TypeError, ValueError, IndexError):
            return None
        for key in source:
            if key[0] == fid and abs(key[1] - x) <= 2.0 and \
                    abs(key[2] - y) <= 2.0:
                return key
        return None

    def _record_crosshair_evidence(self, frame_id, pixel_norm):
        """记录已展示给 VLM 的十字图；不授予实例化权限。"""
        key = self._crosshair_key(frame_id, pixel_norm)
        self._crosshair_evidence[key] = {
            "frame_id": key[0], "pixel": [key[1], key[2]],
        }
        return key

    # ------------------------------------------------------------------
    # 被拒候选记忆：molmo 反复报同一错误像素、VLM 连续 REJECT 时，系统
    # 硬过滤禁止再 propose；目标在视野中过小/过远时提示 VLM 先靠近。
    # ------------------------------------------------------------------
    def _record_rejected_spot(self, frame_id, pixel_norm, reason=""):
        """记录被语义审核拒绝的像素点；返回拒绝 key 或 None。"""
        try:
            key = (int(frame_id), int(round(float(pixel_norm[0]))),
                   int(round(float(pixel_norm[1]))))
        except (TypeError, ValueError, IndexError):
            return None
        entry = self._rejected_spots.setdefault(
            key, {"count": 0, "reason": "", "step": 0})
        entry["count"] += 1
        if reason:
            entry["reason"] = str(reason)[:200]
        entry["step"] = int(getattr(self._last_observation, "step_count", 0))
        return key

    def _spot_rejected(self, frame_id, pixel_norm):
        """pixel 距任一被拒点 40/1000 以内视为同一位置（硬过滤）。"""
        try:
            fid = int(frame_id)
            x, y = float(pixel_norm[0]), float(pixel_norm[1])
        except (TypeError, ValueError, IndexError):
            return False
        for (fkey, kx, ky) in self._rejected_spots:
            if fkey == fid and math.hypot(kx - x, ky - y) <= 40.0:
                return True
        return False

    def _schedule_revisit(self, frame_id):
        """geometry 解析失败候选：导航到该帧拍摄位姿附近重新观察。

        目标帧位姿由 mapping server 按 frame_id 查询（世界系 4x4）。
        超过 NAV_REVISIT_MAX_ATTEMPTS 次后放弃该帧，避免循环导航。
        """
        fid = int(frame_id)
        try:
            pose = self.client.get_frame_pose(fid)
        except Exception as exc:
            print(f"[NavAgent] revisit frame {fid}: pose query failed: {exc}")
            pose = None
        if pose is None:
            return {"frame_id": fid, "attempts": 0, "navigating": False,
                    "error": "no pose for this frame; re-observe from a "
                             "fresh viewpoint"}
        entry = self._revisit_targets.setdefault(
            fid, {"point": None, "attempts": 0, "step": 0})
        entry["attempts"] += 1
        entry["step"] = int(getattr(self._last_observation, "step_count", 0))
        if entry["attempts"] > self.revisit_max_attempts:
            return {"frame_id": fid, "attempts": entry["attempts"],
                    "navigating": False,
                    "error": "max revisit attempts reached for this frame"}
        entry["point"] = np.asarray(pose[:3, 3], dtype=np.float64)
        # 复用实例导航链：设世界系目标点 + 取消候选重投影，规划后切入 nav。
        self.target_instance_id = None
        self.target_candidate_id = None
        self.target_point = entry["point"]
        self._selected_evidence = None
        self._explore_follower = None
        self._active_frontier_key = None
        self._log_event(f"geometry revisit frame {fid} "
                        f"attempt {entry['attempts']}")
        if not self._plan_to_target(self._last_observation):
            self._clear_current_target()
            return {"frame_id": fid, "attempts": entry["attempts"],
                    "navigating": False,
                    "error": "no navigable path to the frame pose"}
        self.mode = "nav"
        return {"frame_id": fid, "attempts": entry["attempts"],
                "navigating": True}

    def _tool_review_crosshair(self, frame_id, pixel_1000, verdict,
                               reason=""):
        """记录 VLM 对已展示十字图的显式三值语义审核。"""
        verdict = str(verdict or "").strip().upper()
        if verdict not in {"ACCEPT", "REJECT", "UNCERTAIN"}:
            return {"error": "verdict must be ACCEPT, REJECT, or UNCERTAIN"}
        key = self._find_crosshair_key(
            frame_id, pixel_1000, self._crosshair_evidence)
        if key is None:
            return {"error": "no crosshair evidence was shown for this pixel; "
                    "call instantiate_points first"}
        review = {"verdict": verdict, "reason": str(reason or "")[:300]}
        self._crosshair_reviews[key] = review
        if verdict == "REJECT":
            # 同一像素被否 → 进入被拒记忆；propose 硬过滤防止 molmo
            # 反复报同一个错误位置，逼 VLM 靠近后从新视角再试。
            self._record_rejected_spot(key[0], key[1:], reason)
        return {"frame_id": key[0], "pixel": [key[1], key[2]], **review,
                "instantiation_allowed": verdict == "ACCEPT"}

    def _tool_propose_candidates(self, frame_id, query=""):
        """SAM 全分割整帧 → 编号 mask 表 + overlay；VLM 选编号走 som_pick。

        molmo 点指已废弃：分割只依赖 SAM（不依赖 pointing 模型），把
        "生成坐标"变成"选编号"。目标太小/太远时 SAM 会漏分割——先
        START_ADJUST 靠近当前帧再调用，近处 mask 才完整可靠。
        """
        try:
            meta, payload = self.client.som_segment(int(frame_id))
        except Exception as exc:
            return {"error": str(exc)[:200]}
        if meta.get("error"):
            return self._pointing_error(meta)
        if not meta.get("found"):
            return {"masks": [], "error": "unknown or unsegmentable frame"}
        fid = int(frame_id)
        masks = []
        for row in meta.get("masks") or []:
            centroid = row.get("centroid") or []
            if len(centroid) != 2:
                continue
            # 被拒记忆硬过滤：同一帧同一区域已被语义审核否定过。
            if self._spot_rejected(fid, centroid):
                continue
            masks.append({"mask_id": int(row["mask_id"]),
                          "centroid": [round(float(centroid[0]), 1),
                                       round(float(centroid[1]), 1)],
                          "bbox": row.get("bbox"),
                          "area_frac": row.get("area_frac")})
        out = {"frame_id": fid, "masks": masks,
               "next": "the numbered labels on the attached overlay match "
                       "mask_id; call som_pick(frame_id, mask_ids, query) "
                       "with the ids of the regions matching the target, "
                       "then review and commit them with commit_candidates"}
        if payload:
            out["_tool_images"] = [(f"som_{fid}", payload)]
        if not masks:
            out["all_spots_rejected"] = True
            out["next"] = ("every candidate region on this frame has already "
                           "been rejected; do NOT re-propose here. If the "
                           "target is small or far away, START_ADJUST with "
                           "MOVE_FORWARD (or go to a frontier) to get "
                           "closer, then propose from the new viewpoint")
        return out

    def _tool_som_pick(self, frame_id, mask_ids, query=""):
        """把 SoM 选中的 mask 注册为待审核候选（复用 commit 流程）。

        每个 mask 的质心成为候选像素、mask 本体用于深度采样与证据图。
        返回后需照常检查证据面板并用 commit_candidates 提交裁决。
        """
        if not isinstance(mask_ids, (list, tuple)) or not mask_ids:
            return {"error": "mask_ids must be a non-empty list of ints"}
        try:
            resp = self.client.som_pick(int(frame_id), list(mask_ids))
        except Exception as exc:
            return {"error": str(exc)[:200]}
        if resp.get("error"):
            return self._pointing_error(resp)
        images, rows = [], []
        for candidate in resp.get("candidates") or []:
            cid = str(candidate.get("candidate_id") or "")
            fid = candidate.get("frame_id")
            pixel = candidate.get("pixel_norm")
            if not cid or fid is None or not pixel:
                continue
            self._proposals[cid] = {
                "candidate_id": cid, "frame_id": int(fid),
                "pixel_norm": list(pixel), "bbox": candidate.get("bbox"),
                "query": str(query)[:300]
                         or f"som mask {candidate.get('mask_id')}",
                "step": int(
                    getattr(self._last_observation, "step_count", 0)),
                "status": "pending",
            }
            self._record_crosshair_evidence(fid, pixel)
            try:
                meta, payload = self.client.get_candidate_evidence(cid)
            except Exception:
                meta, payload = {"found": False}, b""
            if meta.get("found") and payload:
                images.append((f"proposal_{cid}", payload))
            rows.append({"candidate_id": cid, "frame_id": int(fid),
                         "mask_id": candidate.get("mask_id"),
                         "pixel": list(pixel)})
        if len(self._proposals) > self._proposal_limit:
            oldest = sorted(self._proposals.values(),
                            key=lambda row: row["step"])[:-self._proposal_limit]
            for row in oldest:
                self._proposals.pop(row["candidate_id"], None)
        out = {"proposals": rows, "next":
               "inspect panels, then commit any reviewed subset with one "
               "verdict per submitted candidate; other proposals stay pending"}
        if images:
            out["_tool_images"] = images
        return out

    def _tool_commit_candidates(self, reviews, label=""):
        """Commit a reviewed proposal batch; only ACCEPT becomes active."""
        if self._last_observation is None:
            return {"error": "no observation yet"}
        if not isinstance(reviews, (list, tuple)) or not reviews:
            return {"error": "reviews must be non-empty candidate verdict rows"}
        accepted, rejected, unresolved = [], [], []
        seen = set()
        for row in reviews[:16]:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("candidate_id") or "")
            verdict = str(row.get("verdict") or "UNCERTAIN").upper()
            proposal = self._proposals.get(cid)
            if not cid or cid in seen or proposal is None or \
                    proposal.get("status") != "pending":
                continue
            seen.add(cid)
            if verdict not in {"ACCEPT", "REJECT", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            proposal["status"] = verdict.lower()
            proposal["reason"] = str(row.get("reason") or "")[:300]
            if verdict == "ACCEPT":
                accepted.append(proposal)
            elif verdict == "REJECT":
                rejected.append(cid)
                self._record_rejected_spot(
                    proposal["frame_id"], proposal["pixel_norm"],
                    row.get("reason"))
            else:
                unresolved.append(cid)
        if not accepted:
            return {"instances": [], "accepted": [], "rejected": rejected,
                    "uncertain": unresolved}
        resolved = self.client.resolve_candidates(
            [row["candidate_id"] for row in accepted])
        resolved_rows = (resolved.get("candidates", {})
                         if isinstance(resolved, dict) else {})
        # MappingClient returns {candidates: {...}}; accept the older flat
        # shape in tests/rolling deployments as well.
        if not resolved_rows and isinstance(resolved, dict):
            resolved_rows = resolved
        valid_hits, geometry_rejections = [], []
        for proposal in accepted:
            cid = proposal["candidate_id"]
            row = (resolved_rows.get(cid) or {})
            if not row.get("found"):
                geometry_rejections.append({"candidate_id": cid,
                    "reason": str(row.get("error") or "no valid 3D depth")[:240]})
                proposal["status"] = "geometry_rejected"
                # 不丢弃：导航到该帧拍摄位姿附近重新观察，近处深度更可靠。
                self._schedule_revisit(proposal["frame_id"])
                continue
            proposal["status"] = "active"
            valid_hits.append({**row, "found": True,
                               "frame_id": proposal["frame_id"],
                               "candidate_id": cid,
                               "pixel": proposal["pixel_norm"],
                               "bbox": proposal.get("bbox"),
                               "text": str(label or proposal["query"])})
        changed = self._ingest_semantic_hits(
            self._last_observation, valid_hits, select=False) or []
        dup_rows, dup_images = self._dup_review_payload()
        out = {"instances": self._ground_rows(changed),
               "accepted": [row["candidate_id"] for row in accepted],
               "rejected": rejected, "uncertain": unresolved,
               "geometry_rejections": geometry_rejections}
        if dup_rows:
            out["duplicate_review"] = dup_rows
            out["next"] = ("duplicate_review lists new observations near "
                           "existing instances; compare the attached evidence "
                           "images and call resolve_duplicate for each: "
                           "DUPLICATE merges into the existing instance, "
                           "NEW creates a separate one")
            if dup_images:
                out["_tool_images"] = dup_images
        return out

    def _tool_instantiate_points(self, frame_id, pixels_1000, label=""):
        """按像素坐标实例化 3D 目标（0-1000 归一化坐标）。

        两段式确认：像素必须先经过"十字证据图给 VLM 看过"（本工具第一次
        调用渲染返回并登记）。VLM 看图后原样重发
        同坐标即确认，跳过语义审核直接做 3D 几何验证；未确认的坐标返回
        pending_confirmation + 十字图，不注册任何实例。
        """
        if self._last_observation is None:
            return {"error": "no observation yet"}
        if not isinstance(pixels_1000, (list, tuple)) or not pixels_1000:
            return {"error": "pixels_1000 must be a non-empty list of "
                             "[x, y] in 0-1000 normalized coordinates"}
        # 先准备候选以生成十字证据；随后只解析被显式 ACCEPT 的像素。
        prepare = getattr(self.client, "prepare_pixels", None)
        if prepare is not None:
            try:
                prepared = prepare(int(frame_id), pixels_1000, normalized=True)
            except Exception as exc:
                return {"error": str(exc)[:200]}
            if prepared.get("error"):
                return self._pointing_error(prepared)
            candidates = prepared.get("candidates") or []
            if not candidates:
                return {"instances": [], "semantic_rejections": [],
                        "geometry_rejections": []}
            desc = str(label or "").strip()
            hits = [{"found": True, "frame_id": row.get("frame_id"),
                     "pixel": row.get("pixel"),
                     "pixel_norm": row.get("pixel_norm"),
                     "candidate_id": row.get("candidate_id"),
                     "bbox": row.get("bbox"),
                     "point_score": row.get("point_score", 1.0),
                     "text": desc}
                    for row in candidates]
            pending, accepted, semantic_rejections = [], [], []
            for hit in hits:
                fid, pixel = hit.get("frame_id"), hit.get("pixel_norm")
                evidence_key = self._find_crosshair_key(
                    fid, pixel, self._crosshair_evidence)
                review_key = self._find_crosshair_key(
                    fid, pixel, self._crosshair_reviews)
                review = (self._crosshair_reviews.get(review_key)
                          if review_key is not None else None)
                if evidence_key is None or review is None:
                    pending.append(hit)
                elif review["verdict"] == "ACCEPT":
                    accepted.append(hit)
                else:
                    semantic_rejections.append({
                        "candidate_id": hit.get("candidate_id"),
                        "frame_id": fid,
                        "pixel": pixel or hit.get("pixel"),
                        "verdict": review["verdict"],
                        "reason": review.get("reason", ""),
                    })
                    if review["verdict"] == "REJECT":
                        self._record_rejected_spot(
                            fid, pixel or hit.get("pixel"),
                            review.get("reason", ""))
            if pending:
                # 第一段：渲染十字图返回给 VLM 看图确认，不注册实例。
                images = []
                for h in pending:
                    cid = h.get("candidate_id")
                    try:
                        meta, payload = self.client.get_candidate_evidence(cid)
                    except Exception:
                        meta, payload = {"found": False}, b""
                    if not (meta.get("found") and payload):
                        continue
                    images.append((f"confirm_{cid}", payload))
                    try:
                        self._record_crosshair_evidence(
                            h["frame_id"], h["pixel_norm"])
                    except (TypeError, ValueError, IndexError):
                        pass
                return {"instances": [],
                        "semantic_rejections": semantic_rejections,
                        "geometry_rejections": [],
                        "pending_confirmation": [
                            {"candidate_id": h.get("candidate_id"),
                             "frame_id": h.get("frame_id"),
                             "pixel": h.get("pixel_norm") or h.get("pixel")}
                            for h in pending],
                        "_tool_images": images}
            if not accepted:
                return {"instances": [],
                        "semantic_rejections": semantic_rejections,
                        "geometry_rejections": []}
            ids = [h.get("candidate_id") for h in accepted
                   if h.get("candidate_id")]
            resolved = self.client.resolve_candidates(ids)
            geometry_rejections = []
            valid_hits = []
            for hit in accepted:
                cid = str(hit.get("candidate_id"))
                row = resolved.get(cid) or {}
                if not row.get("found"):
                    geometry_rejections.append({
                        "candidate_id": cid, "frame_id": hit.get("frame_id"),
                        "pixel": hit.get("pixel_norm") or hit.get("pixel"),
                        "reason":
                        str(row.get("error") or "no valid 3D depth")[:240]})
                    # 不丢弃：导航到该帧拍摄位姿附近重新观察。
                    self._schedule_revisit(hit.get("frame_id"))
                    continue
                hit.update(row)
                valid_hits.append(hit)
            changed = self._ingest_semantic_hits(
                self._last_observation, valid_hits, select=False) or []
            return {"instances": self._ground_rows(changed),
                    "semantic_rejections": semantic_rejections,
                    "geometry_rejections": geometry_rejections}
        return {"error": "mapping server lacks prepare_pixels; refusing "
                "unsafe instantiation without explicit semantic audit"}

    def _tool_get_agent_status(self):
        """覆盖/预算快照：服务端建图状态 + 最新 caption 帧号 + 实例计数。"""
        try:
            state = self.client.get_state()
            status = {
                "num_frames": state.get("num_frames"),
                "num_submaps": state.get("num_submaps"),
                "num_loop_closures": state.get("num_loop_closures"),
                "caption_pending": state.get("caption_pending"),
                "semantic": state.get("semantic"),
            }
            _enabled, frame_ids = self.client.get_captioned_frame_ids()
            status["latest_captioned_frame_ids"] = [
                int(fid) for fid in frame_ids][-5:]
        except Exception as exc:
            return {"error": str(exc)[:200]}
        status["instances_total"] = len(self.memory.nodes)
        status["unreported_instances"] = len(self.memory.available())
        obs = self._last_observation
        if obs is not None:
            status["steps_remaining"] = max(
                0, int(obs.max_steps) - int(obs.step_count))
        return status

    def _tool_set_notes(self, text):
        """覆盖 VLM 自己的跨决策工作记忆（上限 500 字，随决策回传）。"""
        self._notes = str(text)[:500]
        return {"notes": self._notes}

    def _tool_get_action_history(self, before_step=None, limit=20):
        """分页查询更早的动作流水（不含 outcome 为 None 的进行中条目）。"""
        try:
            limit = max(1, min(100, int(limit)))
        except (TypeError, ValueError):
            limit = 20
        entries = [e for e in self._action_log if e.get("outcome") is not None]
        if before_step is not None:
            try:
                before = int(before_step)
                entries = [e for e in entries if int(e["step"]) < before]
            except (TypeError, ValueError):
                pass
        return [dict(e) for e in entries[-limit:]]

    def _dead_reckon_snapshot_pose(self, pose, frame_id, scale=None):
        """从同一地图快照的末帧位姿重放尚未进入 SLAM 的动作。"""
        try:
            x, y, yaw = nav.pose_to_yaw_2d(pose, self.align_R)
            tracker = nav.PathFollower(
                scale=scale or self._metric_scale_value() or 1.0,
                reach_m=self.reach_m)
            tracker.x, tracker.y, tracker.yaw = x, y, yaw
            tracker.anchor_frame = int(frame_id)
            start = max(int(frame_id) - 1, 0)
            for action in self.calibrator.actions[start:]:
                tracker.dead_reckon(action)
            return float(tracker.x), float(tracker.y), float(tracker.yaw)
        except Exception:
            return None

    @staticmethod
    def _map_revision_tuple(state):
        if not isinstance(state, dict):
            return None
        try:
            return (int(state.get("num_frames", 0)),
                    int(state.get("num_submaps", 0)),
                    int(state.get("num_loop_closures", 0)))
        except (TypeError, ValueError):
            return None

    def _ensure_decision_map_snapshot(self, observation, force=False):
        """在决策前刷新地图；同一步的多轮工具调用复用一次快照。"""
        step = int(observation.step_count)
        if self._last_decision_snapshot_step == step and \
                self._frontier_grid is not None:
            return
        refresh = self._frontier_grid is None or force or \
            step - self._last_frontier_step >= self.decision_map_refresh_interval
        if not refresh and self._frontier_server_revision is not None:
            try:
                current = self._map_revision_tuple(self.client.get_state())
                cached = self._map_revision_tuple(
                    self._frontier_server_revision)
                refresh = current is not None and current != cached
            except Exception:
                pass
        if refresh:
            self._plan_exploration(observation, select=False)
        if self._frontier_grid is not None:
            self._last_decision_snapshot_step = step

    def _estimated_current_pose(self):
        """Return the best current aligned (x, y, yaw) estimate.

        Prefer a live path follower because it already includes dead reckoning
        after the newest keyframe. Otherwise reconstruct the same estimate from
        the newest SLAM pose plus actions executed since that frame.
        """
        for follower in (self.follower, self._explore_follower):
            if follower is not None and follower.anchor_frame >= 0:
                return (float(follower.x), float(follower.y),
                        float(follower.yaw))
        if not self._ensure_alignment():
            return None
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is None or not len(poses):
                return None
            order = np.argsort(frame_ids)
            fid = int(np.asarray(frame_ids)[order][-1])
            pose = np.asarray(poses, dtype=np.float64)[order][-1]
            x, y, yaw = nav.pose_to_yaw_2d(pose, self.align_R)
            scale = self._metric_scale_value() or 1.0
            tracker = nav.PathFollower(scale=scale, reach_m=self.reach_m)
            tracker.x, tracker.y, tracker.yaw = x, y, yaw
            tracker.anchor_frame = fid
            for action in self.calibrator.actions[fid - 1:]:
                tracker.dead_reckon(action)
            return float(tracker.x), float(tracker.y), float(tracker.yaw)
        except Exception:
            return None

    def _active_target_info(self, pose, frontiers, scale=None):
        """Return JSON metadata and renderer metadata for the active target."""
        target_type, target_id, target_xy, target_text = None, None, None, None
        if self.target_instance_id is not None:
            node = self.memory.get(self.target_instance_id)
            target_type = "instance"
            target_id = self.target_instance_id
            if node is not None:
                target_xy = tuple(np.asarray(node.point, dtype=np.float64)[:2])
                target_text = node.text
            elif self.target_point is not None:
                target_xy = tuple(self._aligned_point(self.target_point)[:2])
                target_text = self.target_text
        elif self._active_frontier_key is not None:
            target_type = "frontier"
            for i, cluster in enumerate(frontiers or []):
                if cluster.get("key") == self._active_frontier_key:
                    target_id = f"f{i}"
                    target_xy = tuple(np.asarray(
                        cluster["world"], dtype=np.float64)[:2])
                    break
            if target_xy is None and self._explore_follower is not None and \
                    self._explore_follower.path:
                target_id = "active_frontier"
                target_xy = tuple(np.asarray(
                    self._explore_follower.path[-1], dtype=np.float64)[:2])
        if target_xy is None:
            return None, None

        scale = scale or self._metric_scale_value() or 1.0
        distance_m = None
        if pose is not None:
            distance_m = math.hypot(
                target_xy[0] - pose[0], target_xy[1] - pose[1]) * scale
        info = {
            "type": target_type,
            "id": target_id,
            "text": target_text,
            "distance_m": (round(float(distance_m), 2)
                           if distance_m is not None else None),
        }
        render_info = {"type": target_type, "id": target_id,
                       "xy": target_xy}
        return info, render_info

    def _build_decider_input(self, observation, local_map=False):
        """组装世界状态和带候选标记的 RGB 点云鸟瞰图。"""
        from agents.decision_state import build_world_state
        self._ensure_decision_map_snapshot(observation, force=local_map)
        self._wait_for_decision_captions()
        # 底图、轨迹和 frontier 都来自 _plan_exploration 的同一原子 frame
        # snapshot；不能再临时拉一份已被回环改写的新 poses 覆盖。
        grid = self._frontier_grid
        if grid is None:
            grid = self._explore_grid if self._explore_grid is not None \
                else self.grid
        frontiers = self._last_frontier_clusters
        pose = self._frontier_pose or self._frontier_slam_pose or \
            self._estimated_current_pose()
        snapshot_scale = self._frontier_scale or \
            self._metric_scale_value() or 1.0
        # 只刷新将进入有界 world-state 的候选，避免长 episode 对上百个
        # 历史实例逐个 RPC；刷新后重建状态，距离和 A* 代价保持一致。
        state = build_world_state(
            self, observation, grid=grid, frontiers=frontiers,
            start_xy=(pose[:2] if pose is not None else None),
            scale=snapshot_scale)
        if self._last_candidate_refresh_step != observation.step_count:
            refresh_ids = [row["id"] for row in state.get("instances", [])]
            if self.target_instance_id is not None:
                refresh_ids.append(self.target_instance_id)
            self._refresh_memory_candidates(refresh_ids)
            self._last_candidate_refresh_step = observation.step_count
            state = build_world_state(
                self, observation, grid=grid, frontiers=frontiers,
                start_xy=(pose[:2] if pose is not None else None),
                scale=snapshot_scale)
        active_target, render_target = self._active_target_info(
            pose, frontiers, scale=snapshot_scale)
        scale = snapshot_scale
        state["navigation"] = {
            "mode": self.mode,
            "current_pose": ({
                "x_m": round(float(pose[0]) * scale, 3),
                "y_m": round(float(pose[1]) * scale, 3),
                "yaw_deg": round(math.degrees(float(pose[2])), 1),
            } if pose is not None else None),
            # 用"已入 submap 可检索的最大帧号"，而不是最新 feed 帧号：
            # 最新帧可能还在关键帧缓冲/子图处理中，direct 查询会 unknown。
            # 旧 server 不返回 last_available_frame_id 时回退 feed 帧号。
            "current_frame_id": (
                int(self._last_feed_info["last_available_frame_id"])
                if self._last_feed_info.get("last_available_frame_id")
                else self._last_feed_info.get("frame_id")),
            "active_target": active_target,
            "snapshot_age_steps": max(
                0, int(observation.step_count) - int(self._last_frontier_step)),
        }
        # 新关键帧编号通知：只交付编号 + caption 摘要，图像经 view_frame 按需取。
        # 服务端不支持这些 RPC 时静默跳过（state 里不加该字段）。
        try:
            _enabled, captioned_ids = self.client.get_captioned_frame_ids()
            new_ids = [int(fid) for fid in captioned_ids
                       if int(fid) > self._last_notified_frame_id]
            if new_ids:
                captions = {
                    int(row["frame_id"]): str(row.get("caption", ""))
                    for row in self.client.get_captions(new_ids)
                    .get("captions", [])}
                state["new_keyframes"] = [
                    {"frame_id": fid,
                     "caption": captions.get(fid, "")[:200]}
                    for fid in new_ids]
                self._last_notified_frame_id = max(new_ids)
        except Exception:
            pass
        # Always expose task-directed retrieval candidates.  They remain
        # hypotheses until their RGB is viewed and the explicit tri-state
        # semantic review accepts a marked point.
        query = self.target_text or self._target_phrase(observation)
        if query:
            try:
                relevant = self.client.retrieve_captions(
                    query, top_k=self.relevant_frame_top_k)
                state["relevant_frames"] = [
                    {"frame_id": int(row["frame_id"]),
                     "score": round(float(row.get("score", 0.0)), 3),
                     "caption": str(row.get("caption", ""))[:300]}
                    for row in relevant if row.get("frame_id") is not None]
            except Exception as exc:
                state["semantic_retrieval"] = {
                    "available": False, "error": str(exc)[:160]}
        map_png = None
        if grid is not None:
            try:
                from agents.map_render import render_pointcloud_topdown
                crop_radius = None
                if local_map and pose is not None:
                    crop_radius = self.adjust_map_radius_m / scale
                visible_ids = [row["id"]
                               for row in state.get("instances", [])
                               ][:self.map_max_instances]
                if self.target_instance_id is not None and \
                        self.target_instance_id not in visible_ids:
                    visible_ids.append(self.target_instance_id)
                visible_ids = set(visible_ids)
                pointcloud = self._frontier_pointcloud
                if pointcloud is None:
                    raise RuntimeError(
                        "mapping snapshot has no RGB point-cloud colors")
                map_png = render_pointcloud_topdown(
                    pointcloud[0], pointcloud[1], pose=pose,
                    instances=[{"id": nd.iid, "xy": tuple(nd.point[:2]),
                                "reported": nd.reported}
                               for nd in self.memory.nodes
                               if nd.iid in visible_ids],
                    frontiers=[{"id": f"f{i}", "xy": tuple(c["world"][:2]),
                                "reason": c.get("reason", "geometry")}
                               for i, c in enumerate(frontiers)],
                    active_target=render_target,
                    crop_center=(pose[:2] if local_map and pose is not None
                                 else None),
                    crop_radius=crop_radius,
                    floor_z=getattr(grid, "floor_z", None),
                    unit_per_m=getattr(grid, "unit_per_m", None),
                    max_plot_points=self.decision_map_max_points)
            except Exception as exc:
                print(f"[NavAgent] 俯视地图渲染失败: {exc}")
                map_png = None
        return state, map_png

    def _apply_decider_steering(self, observation, result):
        """把决策结果映射到现有状态机动作（GOTO_INSTANCE/GOTO_FRONTIER）。
        底层跟随/避障/重规划仍由确定性模块执行。"""
        if result.action == "GOTO_INSTANCE" and result.target_id is not None:
            nd = self.memory.get(result.target_id)
            if nd is not None and not nd.reported and \
                    nd.iid not in self._unreachable_instance_ids:
                self.target_instance_id = nd.iid
                self.target_point = self._raw_point(nd.point)
                self.target_candidate_id = nd.candidate_id
                self._selected_evidence = None
                if nd.candidate_id:
                    try:
                        meta, payload = self.client.get_candidate_evidence(
                            nd.candidate_id)
                        if meta.get("found"):
                            self._selected_evidence = payload
                    except Exception:
                        pass
                self._explore_follower = None
                self._active_frontier_key = None
                self._log_event(f"decider -> GOTO_INSTANCE {nd.iid}")
                if self._plan_to_target(observation):
                    self.mode = "nav"
                    return True
                # Planning failure must not leave an instance displayed as an
                # active target while control has already returned to explore.
                failed_id = nd.iid
                self._clear_current_target()
                self.mode = "explore"
                self._log_event(
                    f"GOTO_INSTANCE {failed_id} plan failed; target cleared")
                return False
        if result.action == "GOTO_FRONTIER" and result.target_id is not None:
            cluster = None
            for i, c in enumerate(self._last_frontier_clusters):
                if f"f{i}" == str(result.target_id):
                    cluster = c
                    break
            if cluster is None:
                return False
            self._activate_frontier(
                observation, cluster,
                source=f"decider GOTO_FRONTIER {result.target_id}")
            return self._explore_follower is not None
        return False

    def _autonomous_explore_action(self, observation):
        """Map EXPLORE to the best deterministic frontier, not random motion.

        The frontier table is refreshed when stale and is already sorted by
        geometric utility. Random/basic exploration remains only the explicit
        recovery path when no reachable non-cooled frontier can be activated.
        """
        self._clear_current_target()
        self.mode = "explore"
        if self._last_frontier_step != observation.step_count:
            self._plan_exploration(observation, select=False)
        if self._last_frontier_clusters:
            selected = self._last_frontier_clusters[0]
            self._activate_frontier(
                observation, selected, source="autonomous EXPLORE")
            action = self._explore_follow(observation)
            if action is not None:
                self._log_event(
                    "EXPLORE -> autonomous frontier f0 "
                    f"utility={selected.get('utility')}")
                return action
        self._log_event(
            "EXPLORE -> basic recovery (no reachable non-cooled frontier)")
        return super()._explore_action(observation)

    def _decider_should_finish(self, observation):
        """all 模式终止决策（VLM 决策层）。True/False；模型不可用返回
        None（调用方回退规则）。FINISH 硬条件由 DecisionLoop 强制。"""
        step = observation.step_count
        if self._reported_count <= 0:
            return False
        if step - self._last_finish_decision_step < self.query_interval:
            return False
        late = step >= int(0.5 * observation.max_steps)
        frontier_quiet = self._frontier_empty_streak >= \
            self.finish_frontier_patience
        if not (late or frontier_quiet):
            return False
        self._last_finish_decision_step = step
        if hasattr(self.vlm, "set_trace_context"):
            self.vlm.set_trace_context(
                episode=str(getattr(observation, "episode_id", "")),
                step=int(getattr(observation, "step_count", 0)),
                goal_text=str(getattr(observation, "goal_text", "") or ""),
                event="finish_check")
        state, map_png = self._build_decider_input(observation)
        result = self.decision_loop.decide(
            "finish_check", state, map_png,
            state_fn=lambda: self._build_decider_input(observation))
        if result is None:
            return None                       # 回退规则
        self._record_action(result.action, result.target_id)
        print(f"[NavAgent] 决策层 finish_check: {result}")
        if result.action == "FINISH":
            return True
        self._apply_decider_steering(observation, result)
        return False

    def _decider_next(self, observation, event, images=None, state_fn=None):
        """事件驱动咨询决策层。返回 (DecisionResult|None, action|None)。"""
        try:
            if hasattr(self.vlm, "set_trace_context"):
                self.vlm.set_trace_context(
                    episode=str(getattr(observation, "episode_id", "")),
                    step=int(getattr(observation, "step_count", 0)),
                    goal_text=str(getattr(observation, "goal_text", "") or ""),
                    event=str(event))
            images = list(images or [])
            if event in ("world_state_updated", "arrival", "scan_complete",
                         "adjustment", "en_route", "nav_failed") and not any(
                             label == "current_observation"
                             for label, _payload in images):
                images.insert(0, (
                    "current_observation", self.vlm.encode_rgb(observation.rgb)))
            state_fn = state_fn or (lambda: self._build_decider_input(observation))
            state, map_png = state_fn()
            result = self.decision_loop.decide(
                event, state, map_png, images=images,
                state_fn=state_fn)
        except Exception as exc:
            print(f"[NavAgent] 决策层调用失败，回退规则: {exc}")
            return None, None
        if result is None:
            return None, None
        print(f"[NavAgent] 决策层 {event}: {result}")
        self._last_decision_output = {
            "step": observation.step_count,
            "event": str(event),
            "output": result.as_dict(),
        }
        self._record_action(result.action, result.target_id)
        self._log_event(
            f"decider {event} -> {result.action} {result.target_id}")
        if result.action == "FINISH":
            # FINISH 硬条件已在 DecisionLoop 内强制（many 计数 / all 终止账本）
            return result, int(Action.FINISH)
        if result.action == "START_ADJUST":
            if self._adjust_reentry_blocked_step == observation.step_count:
                self._adjust_reentry_blocked_step = None
                self._log_event(
                    "suppress repeated START_ADJUST in the same observation")
                # END_ADJUST already consumed the decision turn. Return one
                # safe atomic action so the simulator advances to a fresh RGB
                # observation before adjustment can be requested again.
                return result, int(Action.MOVE_FORWARD)
            return result, self._start_adjustment(
                observation, source_event=event, context_images=images)
        if result.action in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
                             "LOOK_UP", "LOOK_DOWN"):
            action_map = {
                "MOVE_FORWARD": int(Action.MOVE_FORWARD),
                "TURN_LEFT": int(Action.TURN_LEFT),
                "TURN_RIGHT": int(Action.TURN_RIGHT),
                "LOOK_UP": int(Action.LOOK_UP),
                "LOOK_DOWN": int(Action.LOOK_DOWN),
            }
            return result, action_map[result.action]
        if result.action in ("GOTO_INSTANCE", "GOTO_FRONTIER"):
            self._apply_decider_steering(observation, result)
            if self.mode == "nav":
                action, arrived, stuck = self._nav_action(observation)
                if stuck:
                    return result, self._nav_failed_recovery(observation)
                if arrived:
                    return result, self._handle_arrival_transition(observation)
                if action is not None:
                    return result, action
                self._clear_current_target()
                self.mode = "explore"
            action = self._explore_follow(observation)
            return result, (action if action is not None else
                            super()._explore_action(observation))
        if result.action == "EXPLORE":
            return result, self._autonomous_explore_action(observation)
        return result, None

    def _start_adjustment(self, observation, source_event,
                          context_images=None):
        """Enter VLM-controlled single-step visual adjustment."""
        key = ("instance", int(self.target_instance_id)) \
            if self.target_instance_id is not None else \
            ("candidate", str(self.target_candidate_id or source_event))
        budget = self._adjustment_budgets.setdefault(key, {
            "sessions": 0, "steps": 0, "turns": 0,
            "turn_streak": 0, "last_turn": None,
        })
        if budget["sessions"] >= self.adjust_max_sessions_per_target or \
                budget["steps"] >= self.adjust_max_total_steps_per_target:
            self._log_event(
                f"adjustment budget exhausted for {key}: {budget}")
            return self._abandon_adjustment_target(
                observation, "target_adjustment_budget_exhausted")
        budget["sessions"] += 1
        self._adjustment_key = key
        self._adjusting = True
        self._adjust_steps = 0
        self._adjust_source_event = str(source_event)
        self._adjust_goal = (
            "verify_instance" if self.target_instance_id is not None else
            ("clear_path" if self._last_motion_failed else "inspect_sector"))
        self._adjust_start_frame_id = self._last_feed_info.get("frame_id")
        self._adjust_progress = {"successful_moves": 0, "fresh_views": 0,
                                 "new_keyframes": 0}
        self._adjust_pitch_steps = 0
        self._adjust_leveling = False
        self._adjust_end_reason = None
        self._adjust_repeat_remaining = 0
        self._adjust_repeat_total = 0
        # Current RGB must be regenerated after every action. Preserve only
        # historical/context evidence across adjustment rounds.
        self._adjust_context_images = [
            (label, payload) for label, payload in (context_images or [])
            if label != "current_observation"]
        self._log_event(
            f"adjustment started from {self._adjust_source_event}; "
            f"goal={self._adjust_goal}; target_budget={budget}")
        return self._adjustment_action(observation)

    def _abandon_adjustment_target(self, observation, reason):
        """调整无法产生新证据时冷却当前候选，避免 END/START 循环。"""
        if self.target_instance_id is not None:
            self._unreachable_instance_ids.add(self.target_instance_id)
        self._log_event(
            f"adjustment abandon target={self.target_instance_id} "
            f"candidate={self.target_candidate_id}: {reason}")
        self._adjusting = False
        self._adjustment_key = None
        self._adjust_steps = 0
        self._adjust_source_event = None
        self._adjust_context_images = []
        self._adjust_pitch_steps = 0
        self._adjust_leveling = False
        self._adjust_end_reason = None
        self._adjust_goal = None
        self._adjust_start_frame_id = None
        self._adjust_repeat_remaining = 0
        self._adjust_repeat_total = 0
        self._clear_current_target()
        self.mode = "explore"
        return self._explore_action(observation)

    def _adjustment_state(self, observation):
        state, map_png = self._build_decider_input(
            observation, local_map=True)
        previous = getattr(observation, "previous_action", None)
        try:
            previous_id = int(previous)
            previous_name = Action(previous_id).name
        except Exception:
            previous_id = None
            previous_name = None
        collision_detected = bool(
            self._last_motion_failed and
            previous_id == int(Action.MOVE_FORWARD))
        navigation = state.get("navigation", {})
        target_budget = dict(self._adjustment_budgets.get(
            self._adjustment_key, {}))
        state["adjustment"] = {
            "active": True,
            "source_event": self._adjust_source_event,
            "goal": self._adjust_goal,
            "success_conditions": {
                "verify_instance": "obtain at least one fresh post-adjustment view, then END_ADJUST",
                "clear_path": "make one successful move or reveal an alternate route",
                "inspect_sector": "obtain a new mapping keyframe of unseen local space",
            }.get(self._adjust_goal, "obtain new actionable evidence"),
            "progress": dict(self._adjust_progress),
            "steps_used": self._adjust_steps,
            "max_steps": self.adjust_max_steps,
            "steps_remaining": max(
                0, self.adjust_max_steps - self._adjust_steps),
            "max_forward_steps": self.adjust_max_forward_steps,
            "forward_repeat_remaining": self._adjust_repeat_remaining,
            "target_budget": {
                **target_budget,
                "max_sessions": self.adjust_max_sessions_per_target,
                "max_total_steps": self.adjust_max_total_steps_per_target,
                "max_turns": self.adjust_max_turns_per_target,
            },
            "pitch_offset_steps": self._adjust_pitch_steps,
            "pitch_offset_degrees": 30 * self._adjust_pitch_steps,
            "max_pitch_offset_steps": self.adjust_max_tilt_steps,
            "camera_leveling": self._adjust_leveling,
            "last_motion_failed": bool(self._last_motion_failed),
            "previous_action": {"id": previous_id, "name": previous_name},
            "collision": {
                "detected": collision_detected,
                "method": "rgb_motion_delta",
                "applies_to_previous_action": (
                    previous_name == "MOVE_FORWARD"),
            },
            "target_instance_id": self.target_instance_id,
            "target_text": self.target_text,
            "active_target": navigation.get("active_target"),
            "current_pose": navigation.get("current_pose"),
            "current_frame_id": (
                int(self._last_feed_info["last_available_frame_id"])
                if self._last_feed_info.get("last_available_frame_id")
                else self._last_feed_info.get("frame_id")),
            "local_topdown_map": {
                "attached": map_png is not None,
                "radius_m": self.adjust_map_radius_m,
                "base_layer": "RGB point-cloud projection; no occupancy colors or trajectory",
                "robot_marker": "blue arrow labeled AGENT",
                "frontier_marker": "purple diamond labeled fN",
                "target_marker": "green circle labeled tN",
                "active_target_marker": "orange star labeled ACTIVE <id>",
            },
        }
        return state, map_png

    def _trace_adjustment_execution(self, observation, result, action):
        logger = getattr(getattr(self, "decision_loop", None), "logger", None)
        if logger is None:
            return
        try:
            actual_name = Action(int(action)).name
        except Exception:
            actual_name = str(action)
        logger.log({
            "step": observation.step_count,
            "event": "action_executed",
            "source_event": "adjustment",
            "decision_output": result.as_dict(),
            "actual_action": {"id": int(action), "name": actual_name},
            "adjustment_step": self._adjust_steps,
        })

    def _adjustment_action(self, observation):
        """Ask the VLM for exactly one atomic motion, then re-observe."""
        budget = self._adjustment_budgets.get(self._adjustment_key, {})
        if budget.get("steps", 0) >= self.adjust_max_total_steps_per_target:
            return self._abandon_adjustment_target(
                observation, "target_total_adjustment_steps_exhausted")
        if budget.get("turns", 0) >= self.adjust_max_turns_per_target:
            return self._abandon_adjustment_target(
                observation, "target_turn_budget_exhausted")
        if self._adjust_leveling:
            if self._adjust_pitch_steps:
                return self._level_adjustment_camera(observation)
            reason = self._adjust_end_reason or "camera_leveled"
            self._adjust_leveling = False
            self._adjust_end_reason = None
            return self._end_adjustment_and_resume(observation, reason)
        if self._adjust_steps >= self.adjust_max_steps:
            self._adjust_repeat_remaining = 0
            self._adjust_repeat_total = 0
            self._log_event(
                f"adjustment safety limit reached ({self.adjust_max_steps})")
            if self._adjust_pitch_steps:
                self._adjust_leveling = True
                self._adjust_end_reason = "safety_limit"
                return self._level_adjustment_camera(observation)
            return self._end_adjustment_and_resume(observation, "safety_limit")
        if self._adjust_repeat_remaining > 0:
            if self._last_motion_failed:
                # 连续前进途中碰撞：中断剩余步数，交还 VLM 用新观测重新决策。
                self._log_event(
                    f"adjustment forward repeat aborted by collision "
                    f"({self._adjust_repeat_total - self._adjust_repeat_remaining}"
                    f"/{self._adjust_repeat_total} done)")
                self._adjust_repeat_remaining = 0
                self._adjust_repeat_total = 0
            else:
                done = (self._adjust_repeat_total
                        - self._adjust_repeat_remaining)
                self._adjust_repeat_remaining -= 1
                budget["steps"] = budget.get("steps", 0) + 1
                budget["turn_streak"] = 0
                budget["last_turn"] = None
                self._adjust_steps += 1
                trace_result = DecisionResult(
                    "MOVE_FORWARD",
                    reason=(f"forward repeat {done + 1}/"
                            f"{self._adjust_repeat_total}"),
                    validation="forward_repeat")
                self._trace_adjustment_execution(
                    observation, trace_result, int(Action.MOVE_FORWARD))
                self._log_event(
                    f"adjustment step {self._adjust_steps}/"
                    f"{self.adjust_max_steps}: MOVE_FORWARD "
                    f"(repeat {done + 1}/{self._adjust_repeat_total})")
                return int(Action.MOVE_FORWARD)
        result, action = self._decider_next(
            observation, "adjustment",
            images=self._adjust_context_images,
            state_fn=lambda: self._adjustment_state(observation))
        if result is None:
            self._log_event("adjustment decision unavailable")
            if self._adjust_pitch_steps:
                self._adjust_leveling = True
                self._adjust_end_reason = "unavailable"
                return self._level_adjustment_camera(observation)
            return self._end_adjustment_and_resume(observation, "unavailable")
        if result.action == "END_ADJUST":
            if not self._adjustment_goal_met():
                # END is a completion claim, not an escape hatch.  Advance one
                # bounded observation so the VLM can re-evaluate with evidence.
                action = (int(Action.TURN_LEFT) if self._adjust_steps % 2 == 0
                          else int(Action.TURN_RIGHT))
                budget["steps"] = budget.get("steps", 0) + 1
                budget["turns"] = budget.get("turns", 0) + 1
                budget["turn_streak"] = 0
                budget["last_turn"] = None
                self._adjust_steps += 1
                self._trace_adjustment_execution(observation, result, action)
                self._log_event(
                    "ignored END_ADJUST before goal progress; "
                    f"goal={self._adjust_goal} progress={self._adjust_progress}")
                return action
            if self._adjust_pitch_steps:
                # This is direct visual evidence captured at the current robot
                # position. Keep it for the resumed arrival/global decision,
                # then restore the neutral sensor pose before navigation.
                evidence = self.vlm.encode_rgb(observation.rgb)
                self._adjust_context_images = [
                    (label, payload)
                    for label, payload in self._adjust_context_images
                    if label != "adjustment_direct_evidence"]
                self._adjust_context_images.append(
                    ("adjustment_direct_evidence", evidence))
                self._adjust_leveling = True
                self._adjust_end_reason = "vlm"
                return self._level_adjustment_camera(observation, result)
            return self._end_adjustment_and_resume(observation, "vlm")
        if action is None:
            self._log_event(
                f"adjustment produced no executable action: {result.action}")
            return self._end_adjustment_and_resume(observation, "invalid")
        if result.action in ("TURN_LEFT", "TURN_RIGHT"):
            same_turn = budget.get("last_turn") == result.action
            if same_turn and budget.get("turn_streak", 0) >= 2:
                return self._abandon_adjustment_target(
                    observation, "repeated_same_turn")
            budget["turns"] = budget.get("turns", 0) + 1
            budget["turn_streak"] = budget.get("turn_streak", 0) + 1 \
                if same_turn else 1
            budget["last_turn"] = result.action
        else:
            budget["turn_streak"] = 0
            budget["last_turn"] = None
        budget["steps"] = budget.get("steps", 0) + 1
        self._adjust_steps += 1
        if result.action == "LOOK_UP":
            self._adjust_pitch_steps += 1
        elif result.action == "LOOK_DOWN":
            self._adjust_pitch_steps -= 1
        self._adjust_repeat_remaining = 0
        self._adjust_repeat_total = 0
        if result.action == "MOVE_FORWARD":
            # VLM 必须显式给出 steps；本步已执行并计数，其余步数逐步续执行，
            # 碰撞即停。上限同时受会话/目标预算钳制。
            requested = max(1, int(result.steps or 1))
            allowed = max(1, min(
                requested, self.adjust_max_forward_steps,
                self.adjust_max_steps - self._adjust_steps,
                self.adjust_max_total_steps_per_target
                - budget.get("steps", 0) + 1))
            self._adjust_repeat_total = allowed
            self._adjust_repeat_remaining = allowed - 1
        self._trace_adjustment_execution(observation, result, action)
        if result.action == "MOVE_FORWARD" and self._adjust_repeat_total > 1:
            self._log_event(
                f"adjustment step {self._adjust_steps}/"
                f"{self.adjust_max_steps}: MOVE_FORWARD x"
                f"{self._adjust_repeat_total} (requested "
                f"{int(result.steps or 1)})")
        else:
            self._log_event(
                f"adjustment step {self._adjust_steps}/{self.adjust_max_steps}: "
                f"{result.action}")
        return action

    def _adjustment_goal_met(self):
        """Minimal harness-side evidence required before END_ADJUST."""
        progress = self._adjust_progress
        if self._adjust_goal == "clear_path":
            return progress.get("successful_moves", 0) >= 1
        if self._adjust_goal == "inspect_sector":
            return progress.get("new_keyframes", 0) >= 1
        if self._adjust_goal == "verify_instance":
            return progress.get("fresh_views", 0) >= 1
        return progress.get("fresh_views", 0) >= 1

    def _level_adjustment_camera(self, observation, result=None):
        """Return one legal pitch action toward the neutral mapping pose."""
        if not self._adjust_pitch_steps:
            self._adjust_leveling = False
            return self._end_adjustment_and_resume(
                observation, self._adjust_end_reason or "camera_leveled")
        if self._adjust_pitch_steps > 0:
            action_name, action = "LOOK_DOWN", int(Action.LOOK_DOWN)
            self._adjust_pitch_steps -= 1
        else:
            action_name, action = "LOOK_UP", int(Action.LOOK_UP)
            self._adjust_pitch_steps += 1
        budget = self._adjustment_budgets.get(self._adjustment_key)
        if budget is not None:
            budget["steps"] = budget.get("steps", 0) + 1
        self._adjust_steps += 1
        trace_result = result or DecisionResult(
            action_name, reason="automatic camera leveling",
            validation="camera_leveling")
        self._trace_adjustment_execution(observation, trace_result, action)
        self._log_event(
            f"adjustment camera leveling: {action_name}; "
            f"pitch_offset_steps={self._adjust_pitch_steps}")
        return action

    def _end_adjustment_and_resume(self, observation, reason):
        """Exit adjustment and immediately resume its originating decision."""
        if reason in {"safety_limit", "unavailable", "invalid"}:
            return self._abandon_adjustment_target(
                observation, f"adjustment_{reason}")
        source_event = self._adjust_source_event or "world_state_updated"
        context_images = list(self._adjust_context_images)
        self._adjusting = False
        self._adjust_steps = 0
        self._adjust_source_event = None
        self._adjust_context_images = []
        self._adjust_pitch_steps = 0
        self._adjust_leveling = False
        self._adjust_end_reason = None
        self._adjust_goal = None
        self._adjust_start_frame_id = None
        self._adjustment_key = None
        self._adjust_repeat_remaining = 0
        self._adjust_repeat_total = 0
        self._adjust_reentry_blocked_step = observation.step_count
        self._log_event(f"adjustment ended ({reason}); resume {source_event}")

        if source_event == "arrival":
            result, action = self._arrival_vlm_decision(observation)
        else:
            # Local active exploration has fed fresh RGB into SLAM. Rebuild
            # frontier state before resuming the global decision so the new
            # observation can affect the next target rather than leaving the
            # pre-adjustment map frozen for a full replan interval.
            if source_event in ("world_state_updated", "scan_complete"):
                self._plan_exploration(observation, select=False)
            result, action = self._decider_next(
                observation, source_event, images=context_images)
        if result is None:
            if source_event == "arrival":
                self._clear_current_target()
                self.mode = "explore"
            return self._explore_action(observation)
        if result.action == "REPORT_FOUND":
            return self._report_found(result.target_id)
        if result.action == "SCAN":
            self._scanning = True
            self._scan_steps = 0
            self._scan_images = []
            return int(Action.TURN_LEFT)
        # EXPLORE has already activated the autonomous frontier follower in
        # _decider_next. Clearing here would silently turn it back into random
        # walking after END_ADJUST.
        return action if action is not None else self._explore_action(observation)

    def _arrival_vlm_decision(self, observation):
        """到达候选点后只咨询决策 VLM，不调用当前帧 pointing/verify。"""
        if self.decision_loop is None:
            return None, None
        arrival_info = {
            "at_candidate": True,
            "target_instance_id": self.target_instance_id,
            "target_candidate_id": self.target_candidate_id,
            "target_text": self.target_text,
        }

        def arrival_state():
            state, map_png = self._build_decider_input(observation)
            state["arrival"] = dict(arrival_info)
            return state, map_png

        images = [("current_observation",
                   self.vlm.encode_rgb(observation.rgb))]
        if self._selected_evidence:
            images.append(("selected_candidate", self._selected_evidence))
        return self._decider_next(
            observation, "arrival", images=images, state_fn=arrival_state)

    def _block_failed_nav_direction(self, observation, follower=None):
        """把最近撞上的前向小段临时作为障碍，避免 A* 原样重走。"""
        follower = follower or self.follower
        if follower is None:
            return
        scale = self._metric_scale_value() or 1.0
        distance = 0.25 / scale
        point = (float(follower.x + math.cos(follower.yaw) * distance),
                 float(follower.y + math.sin(follower.yaw) * distance))
        expiry = int(observation.step_count) + self.nav_block_ttl_steps
        self._nav_blocked_points.append((point, expiry))
        self._nav_blocked_points = self._nav_blocked_points[-8:]
        self._log_event(
            f"nav temporary block at ({point[0]:.2f},{point[1]:.2f}) "
            f"until step={expiry}")

    def _apply_nav_temporary_blocks(self, observation):
        """将临时撞墙位置投影到当前新栅格；地图重建后仍保留恢复约束。"""
        if self.grid is None:
            return
        step = int(observation.step_count)
        self._nav_blocked_points = [
            item for item in self._nav_blocked_points if item[1] >= step]
        if not self._nav_blocked_points:
            return
        radius_cells = max(1, int(math.ceil(
            self.nav_block_radius_m / max(float(self.grid.res), 1e-6))))
        height, width = self.grid.free.shape
        for point, _expiry in self._nav_blocked_points:
            row, col = self.grid.world_to_cell(point)
            r0, r1 = max(0, row - radius_cells), min(height, row + radius_cells + 1)
            c0, c1 = max(0, col - radius_cells), min(width, col + radius_cells + 1)
            yy, xx = np.ogrid[r0:r1, c0:c1]
            mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius_cells ** 2
            free_view = self.grid.free[r0:r1, c0:c1]
            obstacle_view = self.grid.obstacle[r0:r1, c0:c1]
            free_view[mask] = False
            obstacle_view[mask] = True

    def _nav_failed_vlm_decision(self, observation):
        """导航目标不可达（连续碰撞）：只咨询决策 VLM（event=nav_failed）。

        world_state 的 navigation.blocked_target 记录被放弃的目标与碰撞
        次数；current_observation 由 _decider_next 自动附加。
        """
        if self.decision_loop is None:
            return None, None

        def stuck_state():
            state, map_png = self._build_decider_input(observation)
            navigation = state.get("navigation", {})
            navigation["blocked_target"] = {
                "instance_id": self.target_instance_id,
                "collision_streak": self._nav_collision_streak,
            }
            return state, map_png

        return self._decider_next(observation, "nav_failed",
                                  state_fn=stuck_state)

    def _nav_failed_recovery(self, observation):
        """nav 卡死出口：登记不可达实例，咨询决策 VLM 换目标/探索/结束。

        决策不可用或未映射出动作时，放弃当前目标退回探索（世界状态每
        25 步刷新，探索期间仍能通过 world_state_updated 决策接回）。
        """
        if self.target_instance_id is not None:
            self._unreachable_instance_ids.add(self.target_instance_id)
            self._log_event(
                f"nav stuck: instance {self.target_instance_id} "
                "marked unreachable")
        result, decided_action = self._nav_failed_vlm_decision(observation)
        if result is None:
            self._clear_current_target()
            self.mode = "explore"
            return self._explore_action(observation)
        if result.action == "REPORT_FOUND":
            # 卡在目标旁边但能直接确认的情况（报告校验仍要求 active id）
            return self._report_found(result.target_id)
        if result.action == "SCAN":
            self._scanning = True
            self._scan_steps = 0
            self._scan_images = []
            return int(Action.TURN_LEFT)
        if result.action == "EXPLORE":
            return (decided_action if decided_action is not None
                    else self._autonomous_explore_action(observation))
        # GOTO_INSTANCE/GOTO_FRONTIER/FINISH/START_ADJUST 已由 _decider_next
        # 映射到底层动作（新目标已重建导航状态，不可达实例已被校验拦截）。
        if decided_action is not None:
            return decided_action
        self._clear_current_target()
        self.mode = "explore"
        return self._explore_action(observation)

    def _choose_high_level_target(self, observation,
                                  event="world_state_updated", images=None):
        """统一高层选择：同时向 VLM 暴露 instances/frontiers。

        只有 VLM 不可用或输出非法时，才确定性回退到最近实例，再回退到
        最高 utility frontier。该回退不参与正常 VLM 决策。
        """
        self._plan_exploration(observation, select=False)
        if self.decision_loop is not None:
            result, action = self._decider_next(
                observation, event, images=images)
            if result is not None:
                return (action if action is not None else
                        super()._explore_action(observation))
            # 只有模型不可用/非法才进入确定性保底。
        if self._activate_memory_target(observation):
            if self.mode == "nav":
                action, _arrived, _stuck = self._nav_action(observation)
                if action is not None:
                    return action
        if self._last_frontier_clusters:
            self._activate_frontier(
                observation, self._last_frontier_clusters[0],
                source="deterministic fallback")
            action = self._explore_follow(observation)
            if action is not None:
                return action
        return self._explore_action(observation)

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
            no_pending = not self.memory.available()
            geometric_ready = \
                self._reported_count > 0 and late and no_pending and \
                self._no_hit_queries >= self.finish_patience and frontier_fresh and \
                getattr(self, "_last_reachable_frontier_count",
                        self._last_frontier_count) == 0 and \
                self._frontier_empty_streak >= self.finish_frontier_patience
            fallback_late = observation.step_count >= int(
                0.95 * observation.max_steps)
            map_stable = observation.step_count - self._last_map_growth_step \
                >= self.finish_map_stable_steps
            fallback_ready = geometric_ready and fallback_late and map_stable
            return fallback_ready
        return False

    # EXPLORE：查询目标
    def _periodic_anchor(self, observation):
        """探索期间也定期刷新位姿锚点——否则锚点只在上次规划时更新，
        随机探索撞墙的航位推算误差会无界累积（实测漂数米导致假到达）。"""
        step = observation.step_count
        if step - self._last_anchor_step < 5:
            return
        if self._metric_scale_value() is None:
            return
        self._last_anchor_step = step
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 3:
                self._refresh_anchor(poses, frame_ids)
        except Exception:
            pass

    def _handle_scan(self, observation):
        """通用 360° 环视：只建图并均匀保存图像，不验证当前实例。"""
        if self._scan_steps % 3 == 0 and len(self._scan_images) < 4:
            self._scan_images.append(self.vlm.encode_rgb(observation.rgb))
        self._scan_steps += 1
        if self._scan_steps < 12:
            return int(Action.TURN_LEFT)
        self._scanning = False
        return self._scan_complete_decision(observation)

    def _scan_complete_decision(self, observation):
        """环视结束后刷新 task-relevant 实例，再进行一次全局高层决策。"""
        images = [(f"panorama_view_{i}", value)
                  for i, value in enumerate(self._scan_images)]
        origin_instance = self.target_instance_id
        self._log_event(
            f"panoramic scan complete near instance {origin_instance}; "
            f"captured {len(images)} views")
        self._clear_current_target()
        self.mode = "explore"
        phrase = self.target_text or self._target_phrase(observation)
        self.target_text = phrase
        try:
            # Caption 只为已进入子图的帧排队；先提交环视产生的尾部关键帧。
            self.client.flush_map()
        except Exception as exc:
            self._log_event(f"post-scan map flush unavailable: {exc}")
        self._ensure_alignment()
        self._refresh_memory_candidates()
        # 环视期间的新关键帧可能还在 caption 队列里；先等语义记忆追上，
        # 否则紧接的检索会漏掉刚看到的场景（有界等待，超时继续）。
        self._wait_for_captions()
        # A panoramic scan improves caption retrieval, but it must never
        # inject pointing hits directly into InstanceMemory.  That historical
        # shortcut bypassed the explicit ACCEPT/REJECT/UNCERTAIN gate and was
        # a major source of navigable wall/background false positives.  The
        # panorama is supplied below; the VLM may open a frame and start a
        # reviewable proposal transaction if it sees a target.
        self._log_event("post-scan: captions refreshed; no auto-instantiation")
        return self._choose_high_level_target(
            observation, "scan_complete", images=images)

    def _wait_for_decision_captions(self):
        """决策前等待 caption 队列清空（有界，每 2 秒 poll 一次）。

        保证随后下发的新关键帧编号 caption 已就绪；超时或服务端不支持
        该状态时静默继续，绝不阻塞决策。"""
        wait_s = float(os.environ.get("NAV_CAPTION_WAIT_S", "30"))
        if wait_s <= 0:
            return
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                pending = int(
                    self.client.get_state().get("caption_pending") or 0)
            except Exception:
                return
            if pending <= 0:
                return
            time.sleep(2.0)

    def _wait_for_captions(self):
        """检索前等待 caption worker 消化已入队关键帧（有界等待）。

        环视/建图产生的新关键帧由异步 caption worker 处理；不等待直接
        检索会漏掉刚看到的场景。超时或服务端不支持该状态时继续。"""
        wait_s = float(os.environ.get("NAV_CAPTION_WAIT_S", "30"))
        if wait_s <= 0:
            return
        try:
            if not self.client.wait_captions(timeout=wait_s):
                self._log_event(
                    "caption backlog not drained before retrieval")
        except Exception:
            pass

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
        """从未报告实例记忆产生确定性回退规划序列。"""
        instances = self.memory.available()
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
        """确定性回退：按现有规划器选择持久实例并规划第一段。"""
        nodes = [nd for nd in self._ordered_memory_nodes()
                 if nd.iid not in self._unreachable_instance_ids]
        if not nodes:
            return False
        selected = nodes[0]
        self.target_instance_id = selected.iid
        self.target_point = self._raw_point(selected.point)
        self.target_candidate_id = selected.candidate_id
        self._selected_evidence = None
        self._no_hit_queries = 0
        self._explore_follower = None
        print(f"[NavAgent] 从实例记忆选择 #{selected.iid}: "
              f"{selected.text[:80]}")
        if self._plan_to_target(observation):
            self.mode = "nav"
        return True

    # EXPLORE：pointing observation 经 3m 邻域预筛；疑似重复挂起交决策 VLM
    def _ingest_semantic_hits(self, observation, hits, select=True):
        """Observation 幂等 -> 3m 邻域预筛 -> canonical instance。

        去重策略：新观测的 3D 点 instance_dup_radius_m 内没有已有实例才
        直接新建；有邻居则挂起为 duplicate_review（不建实例、不可导航），
        由决策 VLM 通过 resolve_duplicate 工具看证据图后裁决 NEW（新建）
        或 DUPLICATE（并入既有实例）。复核记录经 self._last_dup_reviews
        交给工具层随返回附上证据图。
        """
        step = observation.step_count
        hits.sort(key=lambda r: r.get("point_score", 0.0), reverse=True)
        changed = []
        dup_reviews = []
        self._last_dup_reviews = dup_reviews
        for h in hits:
            if "point" not in h:
                continue
            pt = np.asarray(h["point"], dtype=np.float64)
            if pt.shape != (3,) or not np.all(np.isfinite(pt)):
                continue
            aligned = self._aligned_point(pt)
            caption = str(h.get("text") or h.get("caption") or "").strip()
            initial_text = caption or (
                f"Pointed candidate retrieved for task query: {self.target_text}")
            evidence = {
                "frame_id": h.get("frame_id"),
                "candidate_id": h.get("candidate_id"),
                "source": "semantic_pointing",
                "point_score": round(float(h.get("point_score", 0.0)), 3),
                "pixel": h.get("pixel"),
                "bbox": h.get("bbox"),
                "depth_std": h.get("depth_std"),
            }
            replay_observation = self.memory.find_replay_observation(
                candidate_id=h.get("candidate_id"),
                frame_id=h.get("frame_id"), pixel=h.get("pixel"),
                bbox=h.get("bbox"))
            replay = (self.memory.instance_for_observation(
                replay_observation.oid)
                if replay_observation is not None else None)
            is_new = False
            association = None
            if replay is not None:
                node = self.memory.register_replay(
                    replay, candidate_id=h.get("candidate_id"),
                    evidence=evidence, point=aligned, step=step)
                association = "observation_replay"
                observation_id = (node.observation_ids[-1]
                                  if node.observation_ids else None)
            elif replay_observation is not None:
                # 同一原始证据此前已挂起（duplicate_review/未决）：commit
                # 重试不得再次新建实例。
                node = None
                observation_id = replay_observation.oid
            else:
                observed = self.memory.new_observation(
                    aligned, text=initial_text, evidence=evidence,
                    frame_id=h.get("frame_id"), step=step,
                    candidate_id=h.get("candidate_id"),
                    pixel=h.get("pixel"), bbox=h.get("bbox"))
                observation_id = observed.oid
                scale = self._metric_scale_value() or 1.0
                neighbors = self.memory.nearby(
                    observed.point, scale, self.instance_dup_radius_m,
                    top_k=4)
                if neighbors:
                    node = None
                    dup_reviews.append({
                        "observation_id": observed.oid,
                        "candidate_id": h.get("candidate_id"),
                        "frame_id": h.get("frame_id"),
                        "text": initial_text[:200],
                        "neighbors": [
                            {"instance_id": nd.iid,
                             "dist_m": round(float(dist), 2),
                             "text": str(nd.text)[:120],
                             "reported": bool(nd.reported)}
                            for dist, nd in neighbors],
                    })
                else:
                    node = self.memory.create_instance(
                        observed, text=initial_text)
                    is_new = True
                    association = "new"
            if node is None:
                cid = str(h.get("candidate_id") or f"obs{observation_id}")
                under_review = any(
                    row["observation_id"] == observation_id
                    for row in dup_reviews)
                self._proposals[cid] = {
                    "candidate_id": cid, "frame_id": h.get("frame_id"),
                    "pixel_norm": h.get("pixel"), "bbox": h.get("bbox"),
                    "query": initial_text[:300], "step": int(step),
                    "status": ("duplicate_review" if under_review
                               else "uncertain"),
                    "observation_id": observation_id,
                }
                self._log_event(
                    f"observation {observation_id} retained as "
                    f"{'duplicate_review' if under_review else 'uncertain'} "
                    "proposal")
                continue
            changed.append({
                "instance_id": node.iid,
                "observation_id": observation_id,
                "is_new": is_new,
                "association": association,
                "reported": node.reported,
                "frame_id": h.get("frame_id"),
                "confidence": round(float(h.get("point_score", 0.0)), 3),
            })
        if not changed and not dup_reviews:
            self._no_hit_queries += 1
            return None if select else []
        self._no_hit_queries = 0
        if changed:
            ids = ", ".join(
                f"obs{row['observation_id']}->#{row['instance_id']}"
                f" ({'new' if row['is_new'] else row['association']})"
                for row in changed)
            self._log_event(f"pointing updated instances {ids}")
            print(f"[NavAgent] step={step} 3D 实例记忆更新: {ids}")
        if dup_reviews:
            self._log_event(
                "duplicate review pending for observations "
                + str([row["observation_id"] for row in dup_reviews]))
        if select:
            return self._choose_high_level_target(
                observation, "world_state_updated")
        return changed

    def _dup_review_payload(self):
        """读取并清空 _last_dup_reviews，附新观测与邻近实例的证据图。"""
        rows = list(getattr(self, "_last_dup_reviews", []) or [])
        self._last_dup_reviews = []
        if not rows:
            return [], []
        images = []
        for entry in rows:
            cid = entry.get("candidate_id")
            if cid:
                try:
                    meta, payload = self.client.get_candidate_evidence(cid)
                except Exception:
                    meta, payload = {"found": False}, b""
                if meta.get("found") and payload:
                    images.append(
                        (f"dup_new_obs{entry['observation_id']}", payload))
            for neighbor in entry.get("neighbors", [])[:2]:
                payload = self._tool_view_instance(neighbor["instance_id"])
                if payload:
                    images.append(
                        (f"dup_existing_{neighbor['instance_id']}", payload))
        return rows, images

    def _tool_resolve_duplicate(self, observation_id, decision,
                                duplicate_of=None, text=""):
        """实例化去重裁决：NEW 新建实例；DUPLICATE 并入既有实例。"""
        observation = self.memory.get_observation(observation_id)
        if observation is None:
            return {"error": f"observation {observation_id!r} not found"}
        if self.memory.instance_for_observation(observation.oid) is not None:
            return {"error": f"observation {observation.oid} already resolved"}
        decision = str(decision or "").strip().upper()
        if decision == "NEW":
            node = self.memory.create_instance(
                observation, text=str(text or observation.text))
            resolved = "new"
        elif decision == "DUPLICATE":
            node = self.memory.get(duplicate_of)
            if node is None:
                return {"error": f"duplicate_of {duplicate_of!r} is not an "
                                 "existing instance"}
            self.memory.attach_observation(
                node, observation, text=str(text or ""))
            resolved = "duplicate"
        else:
            return {"error": "decision must be NEW or DUPLICATE"}
        for proposal in self._proposals.values():
            if proposal.get("observation_id") == observation.oid:
                proposal["status"] = f"resolved_{resolved}"
        self._log_event(
            f"observation {observation.oid} resolved as {resolved} "
            f"-> instance {node.iid}")
        return {"instance_id": node.iid, "observation_id": observation.oid,
                "resolved": resolved, "reported": bool(node.reported)}

    # 规划
    def _refresh_anchor(self, poses, frame_ids):
        """锚定到最新关键帧位姿，并重放锚点之后的动作做航位推算。"""
        order = np.argsort(frame_ids)
        fid = int(np.asarray(frame_ids)[order][-1])
        pose = np.asarray(poses, dtype=np.float64)[order][-1]
        if self.align_R is None:
            self.align_R = nav.gravity_alignment(
                np.asarray(poses, dtype=np.float64),
                cam_up=nav.mount_compensated_cam_up())
        self._update_pool_slam_anchor(
            np.asarray(poses, dtype=np.float64)[order][0])
        scale = self._metric_scale_value()
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
        scale = self._metric_scale_value()
        if scale is None:
            return False
        self._last_plan_step = observation.step_count
        try:
            # A target path must use the same locked server snapshot for its
            # pose anchor, camera centres and occupancy grid.  Fetching poses
            # and points through separate RPCs can mix pre/post-loop-closure
            # coordinate frames and produce a path through a wall.
            frames = self.client.get_frame_points(stride=6)
            pose_by_frame = {
                int(frame.get("frame_id", -1)): np.asarray(
                    frame["pose"], dtype=np.float64)
                for frame in (frames or []) if frame.get("pose") is not None
            }
            frame_ids = np.asarray(sorted(pose_by_frame), dtype=np.int64)
            if len(frame_ids) < 5:
                return False
            poses = np.stack([pose_by_frame[int(fid)] for fid in frame_ids])
            if self.align_R is None:
                self.align_R = nav.gravity_alignment(
                    poses, cam_up=nav.mount_compensated_cam_up())
            grid = nav.OccupancyGrid.from_frame_points(frames, self.align_R)
            if grid is None:
                return False
            raw_scale = (1.0 / grid.unit_per_m
                         if grid.unit_per_m > 0 else None)
            scale = self._update_metric_snapshot(
                raw_scale or scale,
                source="grid" if raw_scale else "calibrator")
            self.grid = grid
            self._refresh_anchor(poses, frame_ids)
        except Exception as e:
            print(f"[NavAgent] 原子地图快照/规划锚点失败: {e}")
            return False

        cam_centers = np.asarray(poses, dtype=np.float64)[:, :3, 3] \
            @ self.align_R.T

        self._apply_nav_temporary_blocks(observation)

        goal_xy = (self.align_R @ self.target_point)[:2]
        # 起点只能吸附到几何确认的 free；traversed 层不参与规划。
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
        self._metric_replan_required = False
        self._plan_failures = 0
        dist_m = sum(np.linalg.norm(np.asarray(path[i + 1]) - np.asarray(path[i]))
                     for i in range(len(path) - 1)) * scale
        print(f"[NavAgent] 规划成功: {len(path)} 航点, 路径长约 {dist_m:.1f}m")
        return True

    # NAV：跟随路径
    def _nav_action(self, observation):
        """返回 (action, arrived, stuck)。stuck=True 表示连续碰撞无法到达
        目标，由 act() 登记不可达并触发 nav_failed 决策（VLM 换目标）。"""
        scale = self._metric_scale_value()
        if scale is None or self.follower is None:
            return None, False, False
        try:
            poses, frame_ids = self.client.get_all_poses()
            if poses is not None and len(poses) >= 3:
                self._refresh_anchor(poses, frame_ids)
        except Exception:
            pass

        # 碰撞恢复不能在同一张地图上直接重规划：它通常会得到原路径并再次
        # 撞墙。失败后先转向脱困，再把前方小段临时封路重规划；多次失败才
        # 放弃目标。只有“上一动作是前进且实际有视觉位移”才清空失败计数，
        # 转向本身不应掩盖刚发生的碰撞。
        previous = getattr(observation, "previous_action", None)
        if previous == int(Action.MOVE_FORWARD) and self._last_motion_failed:
            self._nav_collision_streak += 1
            self._log_event(
                f"nav collision streak={self._nav_collision_streak}")
            if self._nav_collision_streak >= self.nav_collision_limit:
                self._nav_stuck_replanned = False
                return None, False, True
            self._block_failed_nav_direction(observation)
            self._nav_recovery_stage = 1
            turn = (int(Action.TURN_LEFT) if self._nav_collision_streak % 2
                    else int(Action.TURN_RIGHT))
            self._nav_recovery_queue = [turn] * self.nav_escape_turns
        elif previous == int(Action.MOVE_FORWARD):
            self._nav_collision_streak = 0
            self._nav_stuck_replanned = False

        if self._nav_recovery_queue:
            return int(self._nav_recovery_queue.pop(0)), False, False
        if self._nav_recovery_stage == 1:
            self._nav_recovery_stage = 2
            self._nav_stuck_replanned = True
            if not self._plan_to_target(observation):
                return None, False, False
            action, arrived = self.follower.next_action()
            if arrived:
                return None, True, False
            return (int(action), False, False) if action is not None else \
                (None, False, False)

        # 到达判定：到原始目标点的水平距离（评测阈值 1.0m，默认 0.8m 留裕量）
        goal_xy = (self.align_R @ self.target_point)[:2]
        dist_m = math.hypot(goal_xy[0] - self.follower.x,
                            goal_xy[1] - self.follower.y) * scale
        if dist_m < self.reach_m:
            return None, True, False

        # 定期重规划（地图增长 / 回环改写位姿 / 卡死兜底）
        if observation.step_count - self._last_plan_step >= self.replan_interval:
            if not self._plan_to_target(observation):
                return None, False, False      # 地图反倒变差，退回探索

        action, arrived = self.follower.next_action()
        if arrived:
            return None, True, False
        if action is None:
            return None, False, False
        return int(action), False, False
