"""多目标导航 agent：语义记忆 -> 实例定位 -> 栅格规划 -> 路径跟随。

流程（多目标状态机）：
1. EXPLORE：持续建图并周期查询 caption 语义记忆；VLM pointing 命中后，
   从 VGGT 点云 patch 恢复 3D 目标点并直接写入实例记忆。
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

import io
import math
import os
import re

import numpy as np
from PIL import Image

from benchmark_api import Action
from agents import navigator as nav
from agents import planner
from agents import skeleton as skel
from agents.mapping_agent import MappingAgent
from agents.memory import InstanceMemory
from decision import DecisionLoop, DecisionTraceLogger, VLMDecisionClient
from runtime_paths import env_debug_path

# 实例入库时的实例级初始文本：pointing overlay + bbox 局部图 + 任务上下文。
# 关键帧 caption 描述整张图像，不足以区分图中哪个物体被 pointing 命中。
INSTANCE_TEXT_PROMPT = """You are labeling one object instance in the memory of
an embodied navigation agent. Task instruction: "{task}".

The first attached image is a pointing overlay: the marked point/patch is the
detected object.{crop_line}
Full-frame caption for context: "{caption}"

Write a concise instance-level description (at most 2 sentences) of the marked
object only: likely category, visual attributes (color, material, shape), its
immediate surroundings, and any uncertainty about the detection. Do not
describe the whole scene. Reply with plain text only."""


class NavAgent(MappingAgent):
    def __init__(self):
        super().__init__()
        self.query_interval = int(os.environ.get("NAV_QUERY_INTERVAL", "20"))
        self.replan_interval = int(os.environ.get("NAV_REPLAN_INTERVAL", "20"))
        self.warmup_steps = int(os.environ.get("NAV_WARMUP_STEPS", "40"))
        self.reach_m = float(os.environ.get("NAV_REACH_M", "0.8"))
        self.finish_patience = int(os.environ.get("NAV_FINISH_PATIENCE", "5"))
        self.finish_frontier_patience = int(os.environ.get(
            "NAV_FINISH_FRONTIER_PATIENCE", "3"))
        self.finish_map_stable_steps = int(os.environ.get(
            "NAV_FINISH_MAP_STABLE_STEPS", "100"))
        self.ground_top_k = int(os.environ.get("NAV_GROUND_TOP_K", "5"))
        self.adjust_max_steps = max(1, int(os.environ.get(
            "NAV_ADJUST_MAX_STEPS", "10")))
        self.adjust_map_radius_m = max(1.0, float(os.environ.get(
            "NAV_ADJUST_MAP_RADIUS_M", "4.0")))
        # 唯一候选生成语义链路：caption 检索 + pointing + 3D instance memory
        self.vlm = VLMDecisionClient.from_env()
        self.vlm.set_trace_path(env_debug_path(
            "NAV_VLM_TRACE", os.path.join(self.output_dir, "vlm_calls.jsonl")))
        # Save the byte-identical images sent to the decision API beside its
        # JSONL trace. This includes the current RGB and annotated top-down map.
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
                self.decision_loop = DecisionLoop(
                    chat_fn=self.vlm.agentic_chat,
                    tools={"search_captions": self._tool_search_captions,
                           "search_instances": self._tool_search_instances,
                           "look_instance": self._tool_look_instance,
                           "inspect_instance": self._tool_inspect_instance,
                           "update_instance": self._tool_update_instance,
                           "merge_instances": self._tool_merge_instances,
                           "undo_merge": self._tool_undo_merge},
                    logger=DecisionTraceLogger(env_debug_path(
                        "NAV_DECIDER_LOG",
                        os.path.join(self.output_dir,
                                     "decision_trace.jsonl"))),
                    max_tool_rounds=int(os.environ.get(
                        "NAV_DECIDER_MAX_TOOL_ROUNDS", "3")))
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
        # 与 InstanceMemory 的 merge 历史一一对应，用于恢复被 merge 改写的目标。
        self._target_merge_history = []
        self.follower = None
        self.grid = None
        self.align_R = None
        self._last_query_step = -10 ** 9
        self._last_plan_step = -10 ** 9
        self._last_anchor_step = -10 ** 9
        self._plan_failures = 0
        self._scanning = False          # 到达后原地 360° 扫描确认中
        self._scan_steps = 0
        self._scan_images = []
        self.memory = InstanceMemory()
        self._reported_count = 0
        self._no_hit_queries = 0
        self._target_mode = "any"
        self._target_count = None
        self._selected_evidence = None
        # VLM 显式 START_ADJUST/END_ADJUST 控制的通用微调状态。
        self._adjusting = False
        self._adjust_steps = 0
        self._adjust_source_event = None
        self._adjust_context_images = []
        # Prevent START_ADJUST -> END_ADJUST -> START_ADJUST recursion within
        # one benchmark observation when the VLM repeats the same request.
        self._adjust_reentry_blocked_step = None
        self._last_decision_output = None
        # 决策层状态：近期事件流 + 最近一次探索规划的 frontier 缓存
        self._events = []
        self._last_frontier_clusters = []
        self._explore_grid = None
        # frontier 候选、原始边界与鸟瞰图必须来自同一栅格快照。
        self._frontier_grid = None
        self._frontier_layers = None
        self._frontier_revision = 0
        self._frontier_trajectory = []
        self._frontier_pose = None
        self._frontier_slam_pose = None
        self._frontier_scale = None
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
        self.decision_map_refresh_interval = max(1, int(os.environ.get(
            "NAV_DECISION_MAP_REFRESH_INTERVAL", "5")))
        self.map_max_instances = max(1, int(os.environ.get(
            "NAV_MAP_MAX_INSTANCES", "12")))
        self.explore_enabled = os.environ.get(
            "NAV_FRONTIER_EXPLORE", "1") == "1"
        self._frontier_empty_streak = 0
        self._last_frontier_count = None
        self._last_reachable_frontier_count = None
        self._last_frontier_step = -10 ** 9
        self._recent_frontiers = []
        self._frontier_failures = {}
        self._active_frontier_key = None
        self._frontier_exhausted_reported = False
        self._last_map_submaps = 0
        self._last_map_growth_step = 0
        self.frontier_cooldown_steps = int(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_STEPS", "100"))
        self.frontier_cooldown_m = float(os.environ.get(
            "NAV_FRONTIER_COOLDOWN_M", "1.0"))
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
        if self._last_motion_failed:
            if self._active_frontier_key is not None:
                key = self._active_frontier_key
                self._frontier_failures[key] = \
                    self._frontier_failures.get(key, 0) + 1
                self._log_event(
                    f"frontier navigation failed {key} "
                    f"count={self._frontier_failures[key]}")
                self._active_frontier_key = None
            self._explore_follower = None      # 碰撞后旧路径不可信
            return super()._explore_action(observation)
        if not self.explore_enabled:
            return super()._explore_action(observation)
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
            # 到达短 frontier 后保留正常重规划间隔，避免每一步重新选择
            # 同一个已经到达的边界。
            self._last_explore_plan = observation.step_count
            return None
        fl.dead_reckon(int(action))
        return int(action)

    def _plan_exploration(self, observation, select=True):
        """重建栅格并刷新可达 frontier；select=False 时只更新候选。"""
        self._last_explore_plan = observation.step_count
        try:
            # 点云、位姿和 frame_id 必须来自同一个服务端锁内快照。过去
            # 先拉 poses 再拉 frame points，回环可能在两次 RPC 中间改写
            # 坐标系，导致轨迹穿墙、当前位置落在 unknown。
            frames = self.client.get_frame_points(stride=6)
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
            grid = nav.OccupancyGrid.from_frame_points(
                frames, self.align_R)
            if grid is None:
                return False
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
        scale = 1.0 / grid.unit_per_m if grid.unit_per_m > 0 else None
        self._frontier_scale = scale or self.calibrator.current_scale() or 1.0
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
        for c in raw_clusters:
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

        self._last_frontier_count = len(valid)
        self._last_reachable_frontier_count = len(reachable)
        self._frontier_stats = {
            "raw_clusters": len(raw_clusters),
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
        if hasattr(self.vlm, "set_trace_context"):
            self.vlm.set_trace_context(
                episode=str(observation.episode_id),
                step=int(observation.step_count),
                goal_text=str(observation.goal_text or ""))
        self._target_mode = str(observation.target_mode or "any").lower()
        self._target_count = observation.target_count
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
                action, arrived = self._nav_action(observation)
                if arrived:
                    result, decided_action = self._arrival_vlm_decision(
                        observation)
                    if result is None:
                        self._log_event(
                            "arrival decision unavailable; leaving candidate "
                            "without report")
                        self._clear_current_target()
                        self.mode = "explore"
                        action = self._explore_action(observation)
                    elif result.action == "REPORT_FOUND":
                        # 报告由决策 VLM 直接授权；不再做 ground_frame 或 servo。
                        action = self._report_found()
                    elif result.action == "SCAN":
                        # SCAN 只有在决策 VLM 明确选择后才启动。
                        self._scanning = True
                        self._scan_steps = 0
                        self._scan_images = []
                        action = int(Action.TURN_LEFT)
                    elif result.action == "EXPLORE":
                        # _decider_next 已把 EXPLORE 映射为确定性最优前沿；
                        # 不要再次 clear，否则会删掉刚建立的 frontier follower。
                        action = (decided_action if decided_action is not None
                                  else self._autonomous_explore_action(
                                      observation))
                    else:
                        # GOTO_INSTANCE/GOTO_FRONTIER/FINISH 已由
                        # _decider_next 映射到底层动作。
                        action = (decided_action if decided_action is not None
                                  else self._explore_action(observation))
                elif action is None:    # 路径走丢，退回探索
                    self.mode = "explore"
                    action = self._explore_action(observation)
        else:
            self._periodic_anchor(observation)
            action = self._maybe_query_target(observation)
            if action is None:
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
        self._plan_failures = 0
        self._scanning = False
        self._scan_steps = 0
        self._scan_images = []
        self._selected_evidence = None

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

    def _refresh_memory_candidates(self, instance_ids=None):
        """回环后刷新候选坐标；可只处理即将进入 VLM 状态的实例。"""
        if not self._ensure_alignment():
            return
        selected = None if instance_ids is None else {
            int(iid) for iid in instance_ids}
        nodes = [node for node in self.memory.nodes
                 if node.candidate_id and
                 (selected is None or node.iid in selected)]
        if not nodes:
            return
        candidate_to_node = {str(node.candidate_id): node for node in nodes}
        batch = getattr(self.client, "resolve_candidates", None)
        if callable(batch):
            try:
                resolved_rows = batch(list(candidate_to_node))
                for candidate_id, resolved in resolved_rows.items():
                    node = candidate_to_node.get(str(candidate_id))
                    if node is not None and resolved.get("found"):
                        self.memory.refresh_point(
                            node, self._aligned_point(resolved["point"]))
                return
            except Exception:
                # 兼容尚未更新的 mapping server，退回逐个协议。
                pass
        for node in nodes:
            try:
                resolved = self.client.resolve_candidate(node.candidate_id)
                if resolved.get("found"):
                    self.memory.refresh_point(
                        node, self._aligned_point(resolved["point"]))
            except Exception:
                pass

    def _report_found(self):
        node = self.memory.get(self.target_instance_id)
        if node is None and self.target_point is not None:
            node, _ = self.memory.remember(
                self._aligned_point(self.target_point),
                text=f"Reported as satisfying task: {self.target_text}",
                step=getattr(self, "_last_report_step", 0),
                candidate_id=self.target_candidate_id)
            self.target_instance_id = node.iid
        if not self.memory.mark_reported(node):
            self._log_event("ignored duplicate/invalid TARGET_FOUND report")
            self._clear_current_target()
            self.mode = "explore"
            self._scanning = False
            return int(Action.TURN_LEFT)
        self._reported_count += 1
        self._no_hit_queries = 0
        self._log_event(f"reported TARGET_FOUND '{self.target_text}' "
                        f"(total {self._reported_count})")
        self.mode = "reported"
        self._scanning = False
        return int(Action.TARGET_FOUND)

    # Phase 4：VLM 决策层（NAV_DECIDER=vlm，事件驱动，不进控制回路）
    def _log_event(self, message):
        self._events.append(str(message))
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def _tool_search_captions(self, text):
        """决策层只读工具：caption 语义记忆检索。"""
        try:
            results = self.client.retrieve_captions(text, top_k=5)
        except Exception as exc:
            return {"error": str(exc)[:200]}
        return [{"frame_id": r.get("frame_id"),
                 "score": round(float(r.get("score", 0.0)), 3),
                 "caption": str(r.get("caption", ""))[:300]}
                for r in results]

    def _tool_search_instances(self, keywords, reported=None, top_k=10):
        """按VLM给出的关键词直接检索实例text；OR匹配，命中数排序。"""
        if isinstance(keywords, str):
            raw = keywords.replace(",", " ").split()
        elif isinstance(keywords, (list, tuple)):
            raw = keywords
        else:
            return {"error": "keywords must be a non-empty string array"}
        terms = []
        for value in raw:
            term = str(value or "").strip().lower()
            if term and term not in terms:
                terms.append(term)
        if not terms:
            return {"error": "keywords must not be empty"}
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
                "matched_keywords": matched,
                "evidence_count": len(node.evidence),
                "frame_ids": [item.get("frame_id")
                              for item in node.evidence
                              if item.get("frame_id") is not None],
            })
        rows.sort(key=lambda row: (-len(row["matched_keywords"]), row["id"]))
        return rows[:limit]

    def _tool_look_instance(self, instance_id):
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
        }

    def _tool_inspect_instance(self, instance_id):
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

    def _tool_merge_instances(self, instance_ids, text=""):
        old_target = self.memory.get(self.target_instance_id)
        merged = self.memory.merge(instance_ids, text=text)
        if merged is None:
            return {"error": "merge requires at least two existing instances"}
        requested = {str(value) for value in (instance_ids or [])}
        self._target_merge_history.append({
            "target_was_merged": (old_target is not None and
                                  str(old_target.iid) in requested),
            "target_id": old_target.iid if old_target is not None else None,
            "merged_id": merged.iid,
        })
        self._target_merge_history = self._target_merge_history[-50:]
        if old_target is not None and str(old_target.iid) in requested:
            self.target_instance_id = merged.iid
            self.target_point = self._raw_point(merged.point)
            self.target_candidate_id = merged.candidate_id
        self._log_event(f"VLM merged instances {sorted(requested)} -> {merged.iid}")
        return self._instance_tool_view(merged)

    def _tool_undo_merge(self):
        """撤销最近一次 merge：恢复被合并实例的原始记录。"""
        outcome = self.memory.undo_merge()
        if outcome is None:
            return {"error": "no merge to undo"}
        target_record = (self._target_merge_history.pop()
                         if self._target_merge_history else None)
        keep = self.memory.get(outcome["keep_id"])
        restore_id = None
        if target_record and target_record["target_was_merged"] and \
                self.target_instance_id == target_record["merged_id"]:
            restore_id = target_record["target_id"]
        target = self.memory.get(restore_id) if restore_id is not None else None
        if target is None and keep is not None and \
                self.target_instance_id == keep.iid:
            target = keep
        if target is not None:
            self.target_instance_id = target.iid
            self.target_point = self._raw_point(target.point)
            self.target_candidate_id = target.candidate_id
        self._log_event(
            f"VLM undid merge -> kept {outcome['keep_id']}, "
            f"restored {outcome['restored_ids']}")
        return {
            "kept": self._instance_tool_view(keep),
            "restored": [self._instance_tool_view(self.memory.get(iid))
                         for iid in outcome["restored_ids"]],
        }

    def _dead_reckon_snapshot_pose(self, pose, frame_id, scale=None):
        """从同一地图快照的末帧位姿重放尚未进入 SLAM 的动作。"""
        try:
            x, y, yaw = nav.pose_to_yaw_2d(pose, self.align_R)
            tracker = nav.PathFollower(
                scale=scale or self.calibrator.current_scale() or 1.0,
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
            scale = self.calibrator.current_scale() or 1.0
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

        scale = scale or self.calibrator.current_scale() or 1.0
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
        """组装决策输入：世界状态 JSON + 俯视标注地图 PNG（编号一致）。"""
        from agents.decision_state import build_world_state
        self._ensure_decision_map_snapshot(observation, force=local_map)
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
            self.calibrator.current_scale() or 1.0
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
            "active_target": active_target,
            "map_revision": int(self._frontier_revision),
            "mapping_snapshot_revision": self._frontier_server_revision,
            "snapshot_age_steps": max(
                0, int(observation.step_count) - int(self._last_frontier_step)),
        }
        state["frontier_status"] = dict(self._frontier_stats)
        map_png = None
        if grid is not None:
            try:
                from agents.map_render import render_topdown
                trajectory = list(self._frontier_trajectory)
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
                map_png = render_topdown(
                    grid, trajectory=trajectory, pose=pose,
                    instances=[{"id": nd.iid, "xy": tuple(nd.point[:2]),
                                "reported": nd.reported}
                               for nd in self.memory.nodes
                               if nd.iid in visible_ids],
                    frontiers=[{"id": f"f{i}", "xy": tuple(c["world"][:2]),
                                "reason": c.get("reason", "geometry")}
                               for i, c in enumerate(frontiers)],
                    frontier_layers=self._frontier_layers,
                    frontier_stats=self._frontier_stats,
                    active_target=render_target,
                    crop_center=(pose[:2] if local_map and pose is not None
                                 else None),
                    crop_radius=crop_radius,
                    step=observation.step_count,
                    map_revision=self._frontier_revision)
            except Exception as exc:
                print(f"[NavAgent] 俯视地图渲染失败: {exc}")
                map_png = None
        return state, map_png

    def _apply_decider_steering(self, observation, result):
        """把决策结果映射到现有状态机动作（GOTO_INSTANCE/GOTO_FRONTIER）。
        底层跟随/避障/重规划仍由确定性模块执行。"""
        if result.action == "GOTO_INSTANCE" and result.target_id is not None:
            nd = self.memory.get(result.target_id)
            if nd is not None and not nd.reported:
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
                return
        if result.action == "GOTO_FRONTIER" and result.target_id is not None:
            cluster = None
            for i, c in enumerate(self._last_frontier_clusters):
                if f"f{i}" == str(result.target_id):
                    cluster = c
                    break
            if cluster is None:
                return
            self._activate_frontier(
                observation, cluster,
                source=f"decider GOTO_FRONTIER {result.target_id}")

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
        state, map_png = self._build_decider_input(observation)
        result = self.decision_loop.decide(
            "finish_check", state, map_png,
            state_fn=lambda: self._build_decider_input(observation))
        if result is None:
            return None                       # 回退规则
        print(f"[NavAgent] 决策层 finish_check: {result}")
        if result.action == "FINISH":
            return True
        self._apply_decider_steering(observation, result)
        return False

    def _decider_next(self, observation, event, images=None, state_fn=None):
        """事件驱动咨询决策层。返回 (DecisionResult|None, action|None)。"""
        try:
            images = list(images or [])
            if event in ("world_state_updated", "arrival", "scan_complete",
                         "adjustment") and not any(
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
        if result.action in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            action_map = {
                "MOVE_FORWARD": int(Action.MOVE_FORWARD),
                "TURN_LEFT": int(Action.TURN_LEFT),
                "TURN_RIGHT": int(Action.TURN_RIGHT),
            }
            return result, action_map[result.action]
        if result.action in ("GOTO_INSTANCE", "GOTO_FRONTIER"):
            self._apply_decider_steering(observation, result)
            if self.mode == "nav":
                action, arrived = self._nav_action(observation)
                if action is not None:
                    return result, action
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
        self._adjusting = True
        self._adjust_steps = 0
        self._adjust_source_event = str(source_event)
        # Current RGB must be regenerated after every action. Preserve only
        # historical/context evidence across adjustment rounds.
        self._adjust_context_images = [
            (label, payload) for label, payload in (context_images or [])
            if label != "current_observation"]
        self._log_event(
            f"adjustment started from {self._adjust_source_event}")
        return self._adjustment_action(observation)

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
        state["adjustment"] = {
            "active": True,
            "source_event": self._adjust_source_event,
            "steps_used": self._adjust_steps,
            "max_steps": self.adjust_max_steps,
            "steps_remaining": max(
                0, self.adjust_max_steps - self._adjust_steps),
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
            "local_topdown_map": {
                "attached": map_png is not None,
                "radius_m": self.adjust_map_radius_m,
                "robot_marker": "blue arrow labeled YOU",
                "active_target_marker": "orange star labeled TARGET <id>",
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
        if self._adjust_steps >= self.adjust_max_steps:
            self._log_event(
                f"adjustment safety limit reached ({self.adjust_max_steps})")
            return self._end_adjustment_and_resume(observation, "safety_limit")
        result, action = self._decider_next(
            observation, "adjustment",
            images=self._adjust_context_images,
            state_fn=lambda: self._adjustment_state(observation))
        if result is None:
            self._log_event("adjustment decision unavailable")
            return self._end_adjustment_and_resume(observation, "unavailable")
        if result.action == "END_ADJUST":
            return self._end_adjustment_and_resume(observation, "vlm")
        if action is None:
            self._log_event(
                f"adjustment produced no executable action: {result.action}")
            return self._end_adjustment_and_resume(observation, "invalid")
        self._adjust_steps += 1
        self._trace_adjustment_execution(observation, result, action)
        self._log_event(
            f"adjustment step {self._adjust_steps}/{self.adjust_max_steps}: "
            f"{result.action}")
        return action

    def _end_adjustment_and_resume(self, observation, reason):
        """Exit adjustment and immediately resume its originating decision."""
        source_event = self._adjust_source_event or "world_state_updated"
        context_images = list(self._adjust_context_images)
        self._adjusting = False
        self._adjust_steps = 0
        self._adjust_source_event = None
        self._adjust_context_images = []
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
            return self._report_found()
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
                action, _arrived = self._nav_action(observation)
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
        if self.calibrator.current_scale() is None:
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
        try:
            results = self.client.ground_object(
                phrase, top_k=self.ground_top_k)
            hits = [item for item in results if item.get("found")]
        except Exception as exc:
            self._log_event(f"post-scan memory refresh failed: {exc}")
            hits = []
        if hits:
            self._ingest_semantic_hits(observation, hits, select=False)
        else:
            self._no_hit_queries += 1
        return self._choose_high_level_target(
            observation, "scan_complete", images=images)

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
        nodes = self._ordered_memory_nodes()
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

    # EXPLORE：pointing 命中直接写入统一 instance memory
    def _ingest_semantic_hits(self, observation, hits, select=True):
        """每个具有有限 3D 点的 pointing 结果都成为可导航实例。"""
        step = observation.step_count
        hits.sort(key=lambda r: r.get("point_score", 0.0), reverse=True)
        changed = []
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
                "bbox": h.get("bbox"),
                "depth_std": h.get("depth_std"),
            }
            node, is_new = self.memory.remember(
                aligned, text=initial_text, evidence=[evidence],
                frame_id=h.get("frame_id"), step=step,
                candidate_id=h.get("candidate_id"))
            if is_new:
                self._generate_instance_text(node, h)
            changed.append((node.iid, is_new))
        if not changed:
            self._no_hit_queries += 1
            return None
        self._no_hit_queries = 0
        ids = ", ".join(f"#{iid}{' new' if fresh else ''}"
                        for iid, fresh in changed)
        self._log_event(f"pointing updated instances {ids}")
        print(f"[NavAgent] step={step} 3D 实例记忆更新: {ids}")
        if select:
            return self._choose_high_level_target(
                observation, "world_state_updated")
        return None

    # 实例级初始文本
    def _generate_instance_text(self, node, hit):
        """新实例入库后生成实例级初始描述。

        输入 pointing overlay（优先）与 bbox 局部裁剪图，结合任务文本与
        关键帧 caption。VLM 不可用、无图像证据或调用失败时，保留入库时
        的 caption 文本，不影响主流程。"""
        vlm = getattr(self, "vlm", None)
        chat_text = getattr(vlm, "chat_text", None)
        if vlm is None or not getattr(vlm, "enabled", False) or \
                chat_text is None:
            return
        images = []
        candidate_id = hit.get("candidate_id") or node.candidate_id
        if candidate_id:
            try:
                meta, payload = self.client.get_candidate_evidence(
                    candidate_id)
                if meta.get("found") and payload:
                    images.append(("pointing_overlay", payload))
            except Exception:
                pass
        crop = self._instance_crop(hit)
        if crop:
            images.append(("instance_crop", crop))
        if not images:
            return
        prompt = INSTANCE_TEXT_PROMPT.format(
            task=self.target_text or "",
            crop_line=(" The second image is a cropped close-up around the "
                       "detection box." if crop else ""),
            caption=str(hit.get("text") or hit.get("caption") or "")[:500])
        try:
            text = chat_text(prompt, images)
        except Exception:
            text = None
        text = str(text or "").strip()
        if not text:
            return
        self.memory.update_text(node.iid, text)
        self._log_event(f"instance {node.iid} described: {text[:120]}")

    def _instance_crop(self, hit, margin=0.35):
        """从源关键帧裁出 pointing bbox 局部图（JPEG 字节）；失败 None。"""
        bbox = hit.get("bbox")
        frame_id = hit.get("frame_id")
        if not bbox or frame_id is None:
            return None
        try:
            meta, payload = self.client.get_frame_image(frame_id)
            if not meta.get("found") or not payload:
                return None
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            w, h = image.size
            x0, y0, x1, y1 = (float(v) for v in list(bbox)[:4])
            if max(x0, y0, x1, y1) <= 1.5:      # 归一化坐标
                x0, x1 = x0 * w, x1 * w
                y0, y1 = y0 * h, y1 * h
            bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            box = (max(0, int(x0 - bw * margin)),
                   max(0, int(y0 - bh * margin)),
                   min(w, int(x1 + bw * margin)),
                   min(h, int(y1 + bh * margin)))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            crop = image.crop(box)
            crop.thumbnail((512, 512))
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue()
        except Exception:
            return None

    def _maybe_query_target(self, observation):
        step = observation.step_count
        if step < self.warmup_steps or step - self._last_query_step < \
                self.query_interval:
            return None
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
            return None

        phrase = self._target_phrase(observation)
        self.target_text = phrase
        self._ensure_alignment()
        self._refresh_memory_candidates()
        try:
            results = self.client.ground_object(
                phrase, top_k=self.ground_top_k)
        except Exception as e:          # server 忙/异常不应杀死 episode
            print(f"[NavAgent] ground_object 失败: {e}")
            return None
        hits = [r for r in results if r.get("found")]
        if not hits:
            self._no_hit_queries += 1
            print(f"[NavAgent] step={step} 未定位到 '{phrase}'")
            return self._choose_high_level_target(
                observation, "world_state_updated")
        return self._ingest_semantic_hits(observation, hits)

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

    # NAV：跟随路径
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
