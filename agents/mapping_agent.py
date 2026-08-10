"""RGB-only 建图验证 agent：随机探索并把每步 RGB 喂给 VGGT-SLAM。

该 agent 不读取深度、GPS、compass 或仿真器真值。episode 结束时保存
SLAM 轨迹和在线尺度诊断；ATE 等真值指标应由 benchmark evaluator 在
agent 边界之外计算。

运行（需先启动 mapping 服务端，见 scripts/run_mapping_server.sh）：

    python run_eval.py \
      --config hm3d_config.yaml \
      --agent agents_to_test.mapping_agent:MappingAgent \
      --goal-type description \
      --dataset-dir dataset_semantic \
      --scene-root /home/wenbofu/datasets/hm3d/versioned_data/hm3d-0.2/hm3d/val \
      --limit 1 --episode-limit 1 --max-steps 300
"""

import atexit
import os

import numpy as np

from benchmark_api import Action
from mapping.client import MappingClient
from mapping.scale_calibration import ScaleCalibrator

class MappingAgent:
    def __init__(self):
        self.client = MappingClient(
            host=os.environ.get("VGGT_SLAM_HOST", "127.0.0.1"),
            port=int(os.environ.get("VGGT_SLAM_PORT", "5555")),
        )
        self.output_dir = os.environ.get(
            "MAPPING_DEBUG_DIR",
            os.path.expanduser("~/vggt_nav_agent/mapping_debug"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_id = None
        self.rng = np.random.default_rng(0)
        self._last_visual = None
        self._last_motion_failed = False
        self.stuck_steps = 0
        self._server_busy = False
        self.calibrator = ScaleCalibrator(
            window=int(os.environ.get("MAPPING_SCALE_WINDOW", "20")))
        atexit.register(self._finalize)

    def reset(self):
        self._finalize()
        self.client.reset_map()
        self.episode_id = None
        self._last_visual = None
        self._last_motion_failed = False
        self.stuck_steps = 0
        self._server_busy = False
        self.calibrator.reset()

    def act(self, observation):
        self._feed_frame(observation)
        action = self._explore_action(observation)
        self._record_and_update(observation, action)
        return action

    # ------------------------------------------------------------------
    # 供子类复用的拆块
    # ------------------------------------------------------------------
    def _feed_frame(self, observation):
        """pacing + 喂帧，并仅用 RGB 变化估计上一前进动作是否失败。"""
        if self.episode_id is None:
            self.episode_id = observation.episode_id
            try:
                self.client.set_episode(self.episode_id)
            except Exception:
                pass
            self._frame_save_dir = None
            save_root = os.environ.get("NAV_SAVE_FRAMES_DIR")
            if save_root:
                self._frame_save_dir = os.path.join(
                    save_root, str(self.episode_id))
                os.makedirs(self._frame_save_dir, exist_ok=True)
                print(f"[MappingAgent] 逐帧 RGB 保存到 {self._frame_save_dir}")

        # pacing：上一轮喂帧时服务端繁忙则先等它空闲，避免关键帧缓冲
        # 被裁剪丢帧。阻塞不消耗 episode 步数预算（步数按 act 次数计）。
        if self._server_busy:
            self.client.wait_idle(timeout=30.0)
        rgb = observation.rgb
        visual = np.asarray(rgb)[::16, ::16, :3].astype(np.int16)
        self._last_motion_failed = False
        if self._last_visual is not None and \
                observation.previous_action == int(Action.MOVE_FORWARD) and \
                visual.shape == self._last_visual.shape:
            delta = float(np.mean(np.abs(visual - self._last_visual)))
            threshold = float(os.environ.get("MAPPING_STUCK_RGB_DELTA", "1.0"))
            self._last_motion_failed = delta < threshold
            if self._last_motion_failed and self.calibrator.actions and \
                    self.calibrator.actions[-1] == int(Action.MOVE_FORWARD):
                # 上一步命令未产生视觉运动，不把它当作尺度样本或航位
                # 推算中的真实前进。-1 是内部 no-op，不会发给 benchmark。
                self.calibrator.actions[-1] = -1
        self._last_visual = visual
        # 实验：喂给 SLAM 前中心裁剪，把 hfov 从 90° 收窄到 VGGT
        # 训练分布内的范围（0.5 -> 约 53°）。只影响 SLAM 输入，
        # agent 自身感知与 benchmark 相机配置不变。
        crop_frac = float(os.environ.get("MAPPING_CROP_FRAC", "1.0"))
        if crop_frac < 1.0:
            h, w = rgb.shape[:2]
            ch, cw = int(h * crop_frac), int(w * crop_frac)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            rgb = rgb[y0:y0 + ch, x0:x0 + cw]
        # 诊断：逐帧保存喂给 SLAM/CLIP 的 RGB（默认与原观测一致）。
        # server 端 feed_frame 也会全量保存；此处为 agent 侧冗余。
        if getattr(self, "_frame_save_dir", None):
            try:
                arr = np.asarray(rgb)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    img = arr
                cv2.imwrite(
                    os.path.join(
                        self._frame_save_dir,
                        f"rgb_{observation.step_count:05d}.jpg"),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            except Exception as exc:
                if getattr(self, "_frame_save_err", False) is False:
                    self._frame_save_err = True
                    print(f"[MappingAgent] 逐帧 RGB 保存失败: {exc}")
        info = self.client.feed_frame(rgb)
        self._server_busy = bool(info.get("busy"))
        if observation.step_count % 20 == 0:
            print(f"[MappingAgent] step={observation.step_count} "
                  f"keyframe={info.get('is_keyframe')} "
                  f"queued={info.get('queued_keyframes')} "
                  f"busy={info.get('busy')}")
        return info

    def _explore_action(self, observation):
        """简单随机探索：默认前进，偶发转向；检测卡住（位置没动）则转向。"""
        if self._last_motion_failed:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps >= 2:
            return int(self.rng.choice([Action.TURN_LEFT, Action.TURN_RIGHT]))
        # 转向概率可调：MAPPING_TURN_PROB 默认 0.3（左右各半）。
        # 平滑探索实验设为 0.05 左右，验证大旋转对 SLAM 的影响。
        turn_prob = float(os.environ.get("MAPPING_TURN_PROB", "0.3"))
        r = self.rng.random()
        if r >= turn_prob:
            return int(Action.MOVE_FORWARD)
        if r < turn_prob / 2:
            return int(Action.TURN_LEFT)
        return int(Action.TURN_RIGHT)

    def _record_and_update(self, observation, action):
        """记录动作 + 定期更新在线尺度估计（滑动窗口，带 pacing 时位姿基本最新）。"""
        self.calibrator.record_action(action)
        if observation.step_count % 10 == 0 and observation.step_count > 0:
            poses, frame_ids = self.client.get_all_poses()
            scale = self.calibrator.update(poses, frame_ids)
            if scale is not None:
                print(f"[MappingAgent] step={observation.step_count} "
                      f"在线尺度估计={scale:.4f} m/unit")

    # ------------------------------------------------------------------
    def _finalize(self):
        if self.episode_id is None:
            return
        # flush_map 同时等待在途子图并提交未满子图的尾帧。
        self.client.flush_map()

        poses, frame_ids = self.client.get_all_poses()
        episode = self.episode_id
        self.episode_id = None
        if poses is None or len(poses) < 1:
            print(f"[MappingAgent] episode {episode}: SLAM 位姿不足，跳过保存")
            return
        slam_xyz = poses[:, 0:3, 3]
        cal_scale = self.calibrator.current_scale()

        out = os.path.join(self.output_dir, f"traj_{episode}.npz")
        np.savez(out, slam_xyz=slam_xyz, frame_ids=frame_ids,
                 actions=np.asarray(self.calibrator.actions),
                 calibrator_scale=cal_scale if cal_scale else np.nan,
                 calibrator_history=np.array(
                     [h[1] for h in self.calibrator.scale_history]))
        print(f"[MappingAgent] episode {episode}: keyframes={len(poses)} -> {out}")
        if cal_scale:
            print(f"[MappingAgent] 在线尺度标定={cal_scale:.4f} m/unit")


if __name__ == "__main__":
    # 冒烟测试：不依赖 habitat，直接用渐变图验证 server/client 通路
    client = MappingClient()
    client.reset_map()
    for i in range(40):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(client.feed_frame(img))
    print(client.get_state())
    client.close()
