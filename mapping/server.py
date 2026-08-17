"""VGGT-SLAM 在线建图服务端。在 vggtslam (python 3.11) conda 环境中运行：

    conda activate vggtslam
    python -m mapping.server --port 5555

或直接使用 scripts/run_mapping_server.sh。

服务逻辑照搬 VGGT-SLAM 仓库 main_realtime.py 的线程模型：
主线程收 RGB 帧、做光流关键帧筛选；攒满一个子图后在后台线程跑
VGGT 前向 + 因子图优化，查询接口随时可取最新位姿与全局点云。

注意：位姿/点云以"子图提交"为粒度更新（默认每 16 个关键帧），
且为单目相对尺度（Sim(3) 意义下一致），米制尺度需客户端自行标定。
"""

import argparse
import json
import os
import shutil
import socket
import threading
import time
import traceback

import cv2
import numpy as np
import torch


def _parse_args():
    parser = argparse.ArgumentParser(description="VGGT-SLAM mapping server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlapping-window-size", type=int, default=1)
    parser.add_argument("--max-loops", type=int, default=1,
                        help="0 关闭回环检测（SALAD ckpt 缺失时也会自动关闭）")
    parser.add_argument("--min-disparity", type=float, default=50)
    parser.add_argument("--conf-threshold", type=float, default=25.0)
    parser.add_argument("--lc-thres", type=float, default=0.95)
    parser.add_argument("--keyframe-dir", type=str, default="mapping_keyframes",
                        help="关键帧临时落盘目录（复用 VGGT 官方预处理）")
    parser.add_argument("--vis", action="store_true",
                        help="开启 viser 可视化（占用 8080 端口）")
    parser.add_argument("--no-semantic", action="store_true",
                        help="关闭 caption/pointing 语义记忆（语义查询不可用）")
    parser.add_argument("--diag-dir", type=str, default="mapping_diag",
                        help="语义查询诊断目录：JSONL 记录 + top-K 帧图像转储")
    return parser.parse_args()


class _NullViewer:
    """headless 环境下替代 viser Viewer 的空实现（不占用 8080 端口）。"""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _has_salad_ckpt():
    ckpt = os.path.join(torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
    return os.path.exists(ckpt)


class MappingServer:
    def __init__(self, args):
        self.args = args

        if not args.vis:
            import vggt_slam.solver as solver_module
            solver_module.Viewer = _NullViewer

        from mapping.online_solver import OnlineSolver
        from vggt_slam.map import GraphMap
        from vggt_slam.graph import PoseGraph
        from vggt_slam.frame_overlap import FrameTracker

        self._GraphMap = GraphMap
        self._PoseGraph = PoseGraph
        self._FrameTracker = FrameTracker

        self.use_loop_closure = args.max_loops > 0 and _has_salad_ckpt()
        if args.max_loops > 0 and not self.use_loop_closure:
            print("[server] WARNING: dino_salad.ckpt 缺失，回环检测已禁用")

        self.solver = OnlineSolver(
            init_conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres,
        )
        if not self.use_loop_closure:
            # 用一个返回空结果的 dummy 替换 SALAD 检索，避免依赖 ckpt。
            self.solver.image_retrieval = _NullRetrieval()

        print("[server] 加载 VGGT-1B 权重（首次运行会从 HuggingFace 下载约 5GB）...")
        from vggt.models.vggt import VGGT
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
        model.eval()
        self.model = model.to(torch.bfloat16).to("cuda")
        print("[server] 模型加载完成")

        self.solver_lock = threading.Lock()  # 保证同时只有一个子图在处理
        self.data_lock = threading.Lock()    # 保护 solver 内部状态
        self.gpu_lock = threading.Lock()     # VGGT/语义模型不并发抢显存

        # 唯一语义链路：查询无关 caption + BGE-M3 检索 + VLM pointing。
        self.retrieve_top_k = int(os.environ.get("NAV_RETRIEVE_TOP_K", "10"))
        self.point_patch = int(os.environ.get("NAV_POINT_PATCH", "11"))
        self.vllm = None
        self.embedder = None
        self.caption_store = None
        self.caption_worker = None
        self.pointer = None
        if not args.no_semantic:
            self._init_semantic_memory()

        self.keyframe_paths = []
        self.target_size = args.submap_size + args.overlapping_window_size
        self.num_frames = 0
        self.num_submaps_launched = 0
        self._candidate_seq = 0
        self._ground_candidates = {}
        self._response_cache = {}

        os.makedirs(args.keyframe_dir, exist_ok=True)

        # 语义查询诊断：JSONL 记录 caption 检索、VQA 与 pointing 结果，
        # 并转储候选帧（keyframe_dir 会在 reset_map 时清空）。
        self.diag_dir = args.diag_dir
        self.diag_lock = threading.Lock()
        self._current_episode = None
        self._diag_fp = None
        self._diag_frame_dir = None
        os.makedirs(self.diag_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # caption / pointing 语义记忆
    # ------------------------------------------------------------------
    def _semantic_enabled(self):
        return self.embedder is not None

    def _init_semantic_memory(self):
        """caption 语义记忆 + VLM pointing 链路。

        模型路径全部走环境变量（远端 /root/autodl-tmp/models/ 下，经
        ModelScope/HF 镜像下载）；任一组件缺失只降级对应能力并打
        warning，不让 server 崩溃。
        """
        from mapping.caption_store import (BGEM3Embedder, CaptionStore,
                                           CaptionWorker)
        from mapping.pointing import PointingGrounder
        from mapping.vllm_client import VLLMGateway

        self.vllm = VLLMGateway(
            url=os.environ.get("NAV_VLLM_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("NAV_VLLM_API_KEY", "EMPTY"))
        store_path = os.environ.get(
            "NAV_CAPTION_STORE_PATH",
            os.path.join(self.args.diag_dir, "caption_store"))
        self.caption_store = CaptionStore(persist_dir=store_path)
        try:
            self.embedder = BGEM3Embedder(
                os.environ.get("NAV_EMBED_MODEL_PATH", ""))
        except RuntimeError as exc:
            print(f"[server] WARNING: {exc}；caption 检索不可用", flush=True)
            self.embedder = None
        caption_model = os.environ.get("NAV_CAPTION_MODEL_PATH", "")
        if self.embedder is not None and caption_model:
            self.caption_worker = CaptionWorker(
                self.vllm, self.embedder, self.caption_store,
                model=caption_model,
                busy_fn=lambda: self.gpu_lock.locked()
                or self.solver_lock.locked())
            print("[server] caption worker 已启动 "
                  f"(model={caption_model})", flush=True)
        else:
            print("[server] WARNING: caption worker 未启动（缺 embedder 或 "
                  "NAV_CAPTION_MODEL_PATH）", flush=True)
        try:
            self.pointer = PointingGrounder(
                self.vllm, model=os.environ.get("NAV_POINTING_MODEL_PATH", ""))
            print("[server] pointing grounder 就绪", flush=True)
        except RuntimeError as exc:
            print(f"[server] WARNING: {exc}；pointing 不可用", flush=True)
            self.pointer = None

    # ------------------------------------------------------------------
    # 帧输入与子图处理
    # ------------------------------------------------------------------
    def feed_frame(self, rgb):
        """喂入一帧 RGB (H, W, 3) uint8。返回处理信息 dict。"""
        self.num_frames += 1
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # 诊断：逐帧保存全部 RGB（JPEG，文件名即 frame_id），供离线
        # "目标是否进入视野"核查。episode 目录由 set_episode 创建。
        if self._current_episode and self._diag_frame_dir:
            try:
                cv2.imwrite(
                    os.path.join(
                        self._diag_frame_dir,
                        f"rgb_{self.num_frames:06d}.jpg"),
                    bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            except Exception:
                pass
        is_keyframe = self.solver.flow_tracker.compute_disparity(
            bgr, self.args.min_disparity)

        if is_keyframe:
            path = os.path.join(
                self.args.keyframe_dir, f"frame_{self.num_frames:06d}.png")
            cv2.imwrite(path, bgr)
            self.keyframe_paths.append(path)

        launched = False
        if len(self.keyframe_paths) >= self.target_size:
            if self.solver_lock.acquire(blocking=False):
                self.num_submaps_launched += 1
                print(f"[server] {time.strftime('%H:%M:%S')} 启动子图 "
                      f"#{self.num_submaps_launched}, 缓冲 {len(self.keyframe_paths)} 帧",
                      flush=True)
                t = threading.Thread(
                    target=self._process_submap,
                    args=(list(self.keyframe_paths),),
                    daemon=True,
                )
                t.start()
                launched = True
                self.keyframe_paths = \
                    self.keyframe_paths[-self.args.overlapping_window_size:]
            else:
                # SLAM 忙，限制积压
                if len(self.keyframe_paths) > self.target_size * 2:
                    self.keyframe_paths = self.keyframe_paths[-self.target_size:]
                if self.num_frames % 20 == 0:
                    print(f"[server] {time.strftime('%H:%M:%S')} 跳过启动(锁被持有), "
                          f"缓冲 {len(self.keyframe_paths)}", flush=True)

        return {
            "is_keyframe": bool(is_keyframe),
            "queued_keyframes": len(self.keyframe_paths),
            "submap_launched": launched,
            "busy": self.solver_lock.locked(),
        }

    def _process_submap(self, image_paths, release_solver_lock=True):
        t_start = time.time()
        try:
            print(f"[server] {time.strftime('%H:%M:%S')} 处理子图 ({len(image_paths)} 帧)...")
            max_loops = self.args.max_loops if self.use_loop_closure else 0
            with self.gpu_lock:
                predictions = self.solver.run_predictions(
                    image_paths, self.model, max_loops)
            with self.data_lock:
                self.solver.add_points(predictions)
                self.solver.graph.optimize()
            self._enqueue_captions()
            del predictions
            torch.cuda.empty_cache()
            print(f"[server] 子图完成, 耗时 {time.time() - t_start:.1f}s, 显存占用 "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")
        except Exception as e:
            print(f"[server] 子图处理失败: {e}")
            traceback.print_exc()
        finally:
            if release_solver_lock and self.solver_lock.locked():
                self.solver_lock.release()
            lock_state = "锁已释放" if release_solver_lock else "同步处理完成"
            print(f"[server] {time.strftime('%H:%M:%S')} 子图线程退出, "
                  f"{lock_state}", flush=True)

    def _enqueue_captions(self):
        """子图处理挂点：为新子图关键帧排队生成 caption（异步、最低优先级）。

        重叠窗口帧已在上一子图入队过，CaptionWorker/Store 按 frame_id
        去重，不会重复生成。
        """
        if self.caption_worker is None:
            return
        try:
            from torchvision.transforms.functional import to_pil_image
            with self.data_lock:
                submap = self.solver.map.get_latest_submap()
                if submap is None:
                    return
                frame_ids = submap.get_frame_ids()
                try:
                    poses = submap.get_all_poses_world(self.solver.graph)
                except Exception:
                    poses = None
                items = []
                for idx in range(len(frame_ids)):
                    pose = (poses[idx] if poses is not None
                            and idx < len(poses) else None)
                    items.append((int(frame_ids[idx]),
                                  to_pil_image(submap.get_frame_at_index(idx)),
                                  pose))
            for fid, pil, pose in items:
                self.caption_worker.enqueue(fid, pil, pose)
        except Exception as exc:
            print(f"[server] caption 入队失败（不影响建图）: {exc}", flush=True)

    def flush_map(self):
        """同步提交 episode 尾部不足一个完整子图的关键帧。"""
        with self.solver_lock:  # 先等待已有后台子图完成
            overlap = self.args.overlapping_window_size \
                if self.num_submaps_launched > 0 else 0
            useful = len(self.keyframe_paths) - overlap
            if useful <= 0 or len(self.keyframe_paths) < 2:
                return {"flushed": False, "queued_keyframes": useful}
            paths = list(self.keyframe_paths)
            self.num_submaps_launched += 1
            self.keyframe_paths = []
            print(f"[server] 同步提交尾部子图 #{self.num_submaps_launched}, "
                  f"{len(paths)} 帧", flush=True)
            self._process_submap(paths, release_solver_lock=False)
        return {"flushed": True, "queued_keyframes": 0}

    def reset_map(self):
        """清空地图，开始新 episode。复用同一个 Solver 以免重载 SALAD。"""
        with self.solver_lock:  # 等待在途子图完成
            with self.data_lock:
                self.solver.map = self._GraphMap()
                self.solver.graph = self._PoseGraph()
                self.solver.flow_tracker = self._FrameTracker()
                self.solver.current_working_submap = None
                self.keyframe_paths = []
                self.num_frames = 0
                self.num_submaps_launched = 0
                self._candidate_seq = 0
                self._ground_candidates = {}
        if self.caption_worker is not None:
            self.caption_worker.clear()
        if self.caption_store is not None:
            self.caption_store.save()
            self.caption_store.clear()
        if os.path.isdir(self.args.keyframe_dir):
            shutil.rmtree(self.args.keyframe_dir)
        os.makedirs(self.args.keyframe_dir, exist_ok=True)
        return {"reset": True}

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def get_state(self):
        with self.data_lock:
            num_submaps = self.solver.map.get_num_submaps()
            num_loops = self.solver.graph.get_num_loops()
        return {
            "num_frames": self.num_frames,
            "queued_keyframes": len(self.keyframe_paths),
            "num_submaps": num_submaps,
            "num_loop_closures": num_loops,
            "loop_closure_enabled": self.use_loop_closure,
            "busy": self.solver_lock.locked(),
            # 语义记忆进度：caption worker 尚未消化完的关键帧数 / 最新完成帧。
            # agent 检索前据此等待，避免漏掉刚入图的关键帧。
            "caption_pending": (self.caption_worker.pending()
                                if self.caption_worker is not None else 0),
            "caption_last_completed": (
                self.caption_worker.last_completed_frame_id
                if self.caption_worker is not None else None),
        }

    def _collect_poses(self):
        """返回 (frame_ids, poses cam2world (N,4,4))，需持有 data_lock。"""
        frame_ids, poses = [], []
        for submap in self.solver.map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses.append(submap.get_all_poses_world(self.solver.graph))
            frame_ids.extend(submap.get_frame_ids())
        if not poses:
            return [], None
        return frame_ids, np.vstack(poses).astype(np.float32)

    def get_all_poses(self):
        with self.data_lock:
            frame_ids, poses = self._collect_poses()
        if poses is None:
            return {"has_pose": False}
        return {
            "has_pose": True,
            "frame_ids": frame_ids,
            "poses": poses.tolist(),
        }

    def get_latest_pose(self):
        with self.data_lock:
            frame_ids, poses = self._collect_poses()
        if poses is None:
            return {"has_pose": False}
        return {
            "has_pose": True,
            "frame_id": frame_ids[-1],
            "pose": poses[-1].tolist(),
        }

    def get_map_points(self, max_points):
        with self.data_lock:
            pts, cols = [], []
            for submap in self.solver.map.ordered_submaps_by_key():
                pts.append(submap.get_points_in_world_frame(self.solver.graph))
                cols.append(submap.get_points_colors())
        if not pts:
            return {"num_points": 0}, b""
        points = np.concatenate(pts, axis=0).astype(np.float32)
        colors = np.concatenate(cols, axis=0).astype(np.uint8)
        if max_points > 0 and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points, colors = points[idx], colors[idx]
        payload = points.tobytes() + colors.tobytes()
        return {"num_points": len(points)}, payload

    def get_frame_points(self, stride):
        """逐帧返回世界系稠密点（NaN=低置信无效点）+ 位姿，用于客户端
        做逐帧局部地板锚定的自由空间投票（全局点云在子图边界有重影，
        直接分层不可靠）。

        返回 ({"frames": [{frame_id, h, w, stride, pose}]}, payload)，
        payload 按 frames 顺序拼接每帧 (h*w, 3) float32 点。
        """
        stride = max(int(stride), 1)
        with self.data_lock:
            metas, chunks = [], []
            for submap in self.solver.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue
                poses = submap.get_all_poses_world(self.solver.graph)
                fids = submap.get_frame_ids()
                for index in range(len(submap.pointclouds)):
                    hom = self.solver.graph.get_homography(
                        index + submap.get_id())
                    pts = submap.pointclouds[index][::stride, ::stride, :]
                    conf = submap.conf_masks[index][::stride, ::stride] \
                        > submap.conf_threshold
                    hh, ww = pts.shape[:2]
                    flat = pts.reshape(-1, 3).astype(np.float64)
                    flat = np.hstack(
                        [flat, np.ones((len(flat), 1))])
                    world = (hom @ flat.T).T
                    world = (world[:, :3] / world[:, 3:]).astype(np.float32)
                    world[~conf.reshape(-1)] = np.nan
                    metas.append({
                        "frame_id": int(fids[index]) if fids else index,
                        "h": hh, "w": ww, "stride": stride,
                        "pose": np.asarray(
                            poses[index], dtype=np.float32).tolist(),
                    })
                    chunks.append(world.tobytes())
        return {"frames": metas}, b"".join(chunks)

    def get_intrinsics(self):
        """返回 VGGT 预测的各子图首帧内参（预处理图像坐标系，518 宽）。"""
        with self.data_lock:
            subs = list(self.solver.map.ordered_submaps_by_key())
            mats = []
            for s in subs:
                if s.proj_mats is not None and len(s.proj_mats) > 0:
                    mats.append(np.asarray(s.proj_mats[0])[:3, :3].tolist())
        return {"intrinsics": mats, "num_submaps": len(subs)}

    # ------------------------------------------------------------------
    # 语义查询
    # ------------------------------------------------------------------
    def _set_episode(self, episode_id):
        """记录当前 episode，并切换诊断 JSONL 输出。"""
        episode_id = str(episode_id or "unknown").strip()
        if episode_id == self._current_episode:
            return
        if self.caption_worker is not None:
            self.caption_worker.clear()
        if self.caption_store is not None:
            self.caption_store.set_episode(episode_id)
        with self.diag_lock:
            if self._diag_fp is not None:
                try:
                    self._diag_fp.close()
                except Exception:
                    pass
                self._diag_fp = None
            self._current_episode = episode_id
            self._diag_frame_dir = os.path.join(
                self.diag_dir, f"{episode_id}_frames")
            os.makedirs(self._diag_frame_dir, exist_ok=True)
            self._diag_fp = open(
                os.path.join(self.diag_dir, f"{episode_id}_queries.jsonl"),
                "a", encoding="utf-8")
            print(f"[server] 诊断 episode={episode_id} -> {self.diag_dir}",
                  flush=True)

    def _diag_write(self, record):
        """线程安全地写一条诊断 JSONL。"""
        record = dict(record)
        record.setdefault("episode", self._current_episode)
        record.setdefault("t", time.strftime("%H:%M:%S"))
        with self.diag_lock:
            if self._diag_fp is None:
                return
            try:
                self._diag_fp.write(json.dumps(record) + "\n")
                self._diag_fp.flush()
            except Exception:
                pass

    def _diag_dump_frame(self, frame, frame_id, tag):
        """把一帧图像转储到诊断目录，返回相对文件名。"""
        if self._diag_frame_dir is None:
            return None
        try:
            from torchvision.transforms.functional import to_pil_image
            name = f"{tag}_frame_{int(frame_id):06d}.png"
            to_pil_image(frame).save(
                os.path.join(self._diag_frame_dir, name))
            return name
        except Exception:
            return None

    def retrieve_captions(self, text, top_k):
        """文文检索：goal_text -> top-K 关键帧 caption。

        返回 [{frame_id, caption, score, pose}]；位姿按当前图优化结果
        实时刷新（存储位姿是入库时刻的，回环后可能过时）。
        """
        if self.embedder is None or self.caption_store is None:
            return {"results": [], "error": "semantic memory disabled"}
        with self.gpu_lock:
            query_emb = self.embedder.encode([text])[0]
        results = self.caption_store.retrieve(query_emb, k=top_k)
        with self.data_lock:
            frame_ids, poses = self._collect_poses()
        pose_by_fid = {}
        if poses is not None:
            for fid, pose in zip(frame_ids, poses):
                pose_by_fid[int(fid)] = np.asarray(pose).tolist()
        for r in results:
            r["pose"] = pose_by_fid.get(int(r["frame_id"]), r.get("pose"))
        self._diag_write({
            "cmd": "retrieve_captions", "text": text, "top_k": top_k,
            "topk": [{"frame_id": r["frame_id"],
                      "score": round(r["score"], 4),
                      "caption": r["caption"][:120]} for r in results],
        })
        return {"results": results}

    def _locate_frame(self, frame_id):
        """frame_id -> (submap_id, frame_index)；找不到返回 None。"""
        with self.data_lock:
            for submap in self.solver.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue
                for idx, fid in enumerate(submap.get_frame_ids()):
                    if int(fid) == int(frame_id):
                        return submap.get_id(), idx
        return None

    def query_text(self, text, top_k):
        """兼容客户端命令名；统一执行 caption 文文检索。"""
        return self.retrieve_captions(text, top_k)

    def ground_object(self, text, top_k):
        """文本 -> caption 检索 + VLM pointing -> 3D 目标点。"""
        return self._ground_object_semantic(text, top_k)

    def _ground_object_semantic(self, text, top_k):
        """caption 检索 -> pointing -> patch 深度采样。

        探索阶段不做类别确认：每个 pointing 像素只要能恢复有效 3D 点就
        成为 instance。语义判断留给决策 VLM 在导航到达后的真实观测完成。
        """
        if self.pointer is None or self.embedder is None:
            return {"results": [], "error": "semantic memory disabled"}
        from torchvision.transforms.functional import to_pil_image

        top_k = max(int(top_k), self.retrieve_top_k)
        results = []
        diag_frames = []
        for item in self.retrieve_captions(text, top_k)["results"]:
            fid = int(item["frame_id"])
            located = self._locate_frame(fid)
            if located is None:
                continue
            sid, idx = located
            with self.data_lock:
                submap = self.solver.map.get_submap(sid)
                frame = submap.get_frame_at_index(idx)
            pil = to_pil_image(frame)
            dump_name = self._diag_dump_frame(frame, fid, "topk")
            entry = {"frame_id": fid, "score": item["score"],
                     "caption": item["caption"], "pose": item["pose"],
                     "text": item["caption"]}
            frame_diag = {"frame_id": fid, "retrieve_score": item["score"],
                          "dump": dump_name, "points": []}
            pts = self.pointer.point(pil, text, frame_key=f"frame_{fid}")
            if not pts:
                entry["found"] = False
                results.append(entry)
                diag_frames.append(frame_diag)
                continue
            for pt in pts:
                resolved = self._resolve_point(sid, idx, fid, pt)
                instance = dict(entry)
                instance.update(resolved)
                results.append(instance)
                frame_diag["points"].append({
                    "pixel": list(pt["pixel"]),
                    "point_score": pt["confidence"],
                    "found": resolved.get("found", False),
                    "depth_std": resolved.get("depth_std"),
                })
            diag_frames.append(frame_diag)
        self._diag_write({
            "cmd": "ground_object", "backend": "semantic_memory",
            "text": text, "top_k": top_k, "frames": diag_frames,
        })
        return {"results": results}

    def _resolve_point(self, sid, idx, frame_id, point_info, register=True):
        """pointing 像素 -> patch 深度采样 -> 当前图优化坐标系 3D 点。

        保留"图像定位 -> 点云采深度 -> 3D 点"这一跳：VLM 可能隔几米
        认出目标，导航端不能只用拍照位姿。
        """
        from mapping.pointing import sample_point_depth

        pixel = point_info["pixel"]
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            pts_local = np.asarray(submap.pointclouds[idx], dtype=np.float64)
            conf = np.asarray(submap.conf_masks[idx]) \
                > submap.get_conf_threshold()
            hom = self.solver.graph.get_homography(idx + submap.get_id())
        h, w = pts_local.shape[:2]
        flat = np.hstack([pts_local.reshape(-1, 3),
                          np.ones((h * w, 1))])
        world = (hom @ flat.T).T
        pts_world = (world[:, :3] / world[:, 3:]).reshape(h, w, 3)
        cam_origin = (hom @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        sampled = sample_point_depth(
            pts_world, conf, pixel, patch=self.point_patch,
            cam_origin=cam_origin)
        if not sampled["found"]:
            return {"found": False, "pixel": [float(pixel[0]), float(pixel[1])],
                    "point_score": float(point_info["confidence"]),
                    "num_points": int(sampled["num_points"])}
        out = {
            "found": True,
            "point": [float(v) for v in sampled["point"]],
            "num_points": int(sampled["num_points"]),
            "depth_std": sampled["depth_std"],
            "spread": sampled["spread"],
            "pixel": [float(pixel[0]), float(pixel[1])],
            "point_score": float(point_info["confidence"]),
        }
        if register:
            out.update(self._register_point_candidate(
                sid, idx, frame_id, pixel, point_info.get("bbox"),
                point_info["confidence"]))
        return out

    def _register_point_candidate(self, sid, idx, frame_id, pixel, bbox,
                                  score):
        """point 候选注册：存像素 + patch，供 resolve_candidate 在最新
        图优化坐标系下重采样；同时合成小圆盘 mask 供 evidence 复用。"""
        self._candidate_seq += 1
        candidate_id = f"c{self._candidate_seq}"
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            h, w = submap.pointclouds[idx].shape[:2]
        if bbox is not None:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        else:
            r = max(self.point_patch, 8)
            x0, y0 = pixel[0] - r, pixel[1] - r
            x1, y1 = pixel[0] + r, pixel[1] + r
        yy, xx = np.mgrid[0:h, 0:w]
        disk = (xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2 \
            <= (max(self.point_patch, 8) / 2.0) ** 2
        self._ground_candidates[candidate_id] = {
            "submap_id": sid,
            "frame_index": int(idx),
            "frame_id": int(frame_id),
            "pixel": (float(pixel[0]), float(pixel[1])),
            "mask": disk,
            "bbox": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            "point_score": float(score),
        }
        while len(self._ground_candidates) > 128:
            self._ground_candidates.pop(next(iter(self._ground_candidates)))
        return {"candidate_id": candidate_id,
                "point_score": float(score),
                "bbox": [x0, y0, x1, y1]}

    def resolve_candidate(self, candidate_id):
        """在当前图优化结果下重新采样 pointing 像素的 3D 点。"""
        cand = self._ground_candidates.get(str(candidate_id))
        if cand is None:
            return {"found": False, "error": "unknown candidate"}
        resolved = self._resolve_point(
            cand["submap_id"], cand["frame_index"], cand["frame_id"],
            {"pixel": cand["pixel"], "confidence": cand["point_score"],
             "bbox": None}, register=False)
        if not resolved.get("found"):
            return {"found": False,
                    "num_points": resolved.get("num_points", 0)}
        return {"found": True, "point": resolved["point"],
                "num_points": resolved["num_points"],
                "depth_std": resolved.get("depth_std")}

    def candidate_evidence(self, candidate_id):
        """生成紧凑的候选 crop + mask overlay JPEG。"""
        cand = self._ground_candidates.get(str(candidate_id))
        if cand is None:
            return {"found": False, "error": "unknown candidate"}, b""
        from torchvision.transforms.functional import to_pil_image
        with self.data_lock:
            submap = self.solver.map.get_submap(cand["submap_id"])
            rgb = np.asarray(to_pil_image(
                submap.get_frame_at_index(cand["frame_index"])))
        mask = cand["mask"]
        overlay = rgb.copy()
        overlay[mask] = (0.45 * overlay[mask] +
                         0.55 * np.array([255, 32, 32])).astype(np.uint8)
        x0, y0, x1, y1 = np.asarray(cand["bbox"], dtype=int)
        h, w = overlay.shape[:2]
        margin = max(8, int(0.1 * max(x1 - x0, y1 - y0)))
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
        crop = overlay[y0:y1, x0:x1]
        if crop.size == 0:
            crop = overlay
        panel = self._evidence_panel(overlay, crop)
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return {"found": False, "error": "jpeg encode failed"}, b""
        return {"found": True, "mime_type": "image/jpeg"}, encoded.tobytes()

    @staticmethod
    def _evidence_panel(context, crop, height=320):
        """左侧全局上下文、右侧 mask 近景，合成一个 token-efficient JPEG。"""
        def resize_to_height(image):
            h, w = image.shape[:2]
            new_w = max(1, int(round(w * height / max(h, 1))))
            return cv2.resize(image, (new_w, height),
                              interpolation=cv2.INTER_AREA)

        context = resize_to_height(np.asarray(context, dtype=np.uint8))
        crop = resize_to_height(np.asarray(crop, dtype=np.uint8))
        separator = np.full((height, 6, 3), 255, dtype=np.uint8)
        return np.concatenate([context, separator, crop], axis=1)

    def ground_frame(self, rgb, text):
        """对单张实时 RGB 做 VQA + pointing 确认，不查询历史记忆。"""
        return self._ground_frame_semantic(rgb, text)

    def _ground_frame_semantic(self, rgb, text):
        """当前帧 pointing + 逐条属性 VQA 复核。"""
        if self.pointer is None:
            return {"found": False, "error": "semantic memory disabled"}
        from PIL import Image
        pil = Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8))
        ver = self.pointer.verify_frame(pil, text)
        pts = self.pointer.point(pil, text) if ver["match"] else []
        best = max(pts, key=lambda p: p["confidence"]) if pts else None
        target_pixels = None
        if best is not None and best.get("bbox") is not None:
            x0, y0, x1, y1 = best["bbox"]
            target_pixels = float(max(x1 - x0, y1 - y0))
        return {
            "found": bool(ver["match"] and best is not None),
            "score": float(best["confidence"]) if best else 0.0,
            "verify": ver,
            "points": [{"pixel": [float(p["pixel"][0]), float(p["pixel"][1])],
                        "confidence": float(p["confidence"]),
                        "bbox": p.get("bbox")} for p in pts],
            "bbox": best.get("bbox") if best else None,
            "target_pixels": target_pixels,
        }

    def get_frame_image(self, frame_id):
        """返回指定关键帧的 JPEG（决策层 look_at 工具用）。"""
        located = self._locate_frame(int(frame_id))
        if located is None:
            return {"found": False, "error": "unknown frame"}, b""
        sid, idx = located
        from torchvision.transforms.functional import to_pil_image
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            rgb = np.asarray(to_pil_image(submap.get_frame_at_index(idx)))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return {"found": False, "error": "jpeg encode failed"}, b""
        return {"found": True, "mime_type": "image/jpeg",
                "frame_id": int(frame_id)}, encoded.tobytes()

    # ------------------------------------------------------------------
    # socket 主循环
    # ------------------------------------------------------------------
    def handle_message(self, header, payload):
        cmd = header.get("cmd")
        if cmd == "ping":
            return {"ok": True}, b""
        if cmd == "feed":
            shape = header["shape"]
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
            return {"ok": True, **self.feed_frame(rgb)}, b""
        if cmd == "reset_map":
            return {"ok": True, **self.reset_map()}, b""
        if cmd == "flush_map":
            return {"ok": True, **self.flush_map()}, b""
        if cmd == "get_state":
            return {"ok": True, **self.get_state()}, b""
        if cmd == "set_episode":
            self._set_episode(header.get("episode_id"))
            return {"ok": True}, b""
        if cmd == "get_latest_pose":
            return {"ok": True, **self.get_latest_pose()}, b""
        if cmd == "get_all_poses":
            return {"ok": True, **self.get_all_poses()}, b""
        if cmd == "get_map":
            resp, points_payload = self.get_map_points(
                int(header.get("max_points", 200000)))
            return {"ok": True, **resp}, points_payload
        if cmd == "get_intrinsics":
            return {"ok": True, **self.get_intrinsics()}, b""
        if cmd == "query_text":
            return {"ok": True, **self.query_text(
                header["text"], int(header.get("top_k", 5)))}, b""
        if cmd == "retrieve_captions":
            return {"ok": True, **self.retrieve_captions(
                header["text"], int(header.get("top_k", 10)))}, b""
        if cmd == "get_frame_image":
            resp, image = self.get_frame_image(header["frame_id"])
            return {"ok": True, **resp}, image
        if cmd == "ground_object":
            return {"ok": True, **self.ground_object(
                header["text"], int(header.get("top_k", 3)))}, b""
        if cmd == "resolve_candidate":
            return {"ok": True, **self.resolve_candidate(
                header["candidate_id"])}, b""
        if cmd == "candidate_evidence":
            resp, evidence = self.candidate_evidence(header["candidate_id"])
            return {"ok": True, **resp}, evidence
        if cmd == "ground_frame":
            shape = header["shape"]
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
            return {"ok": True, **self.ground_frame(
                rgb, header["text"])}, b""
        if cmd == "get_frame_points":
            resp, payload_out = self.get_frame_points(
                int(header.get("stride", 6)))
            return {"ok": True, **resp}, payload_out
        if cmd == "shutdown":
            return {"ok": True, "shutdown": True}, b""
        return {"ok": False, "error": f"unknown cmd: {cmd}"}, b""

    def serve(self):
        from mapping.protocol import recv_msg, send_msg

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.args.host, self.args.port))
        sock.listen(1)
        print(f"[server] 监听 {self.args.host}:{self.args.port}")
        while True:
            conn, addr = sock.accept()
            print(f"[server] 客户端接入: {addr}")
            try:
                while True:
                    header, payload = recv_msg(conn)
                    request_id = header.get("request_id")
                    cacheable = header.get("cmd") in {
                        "feed", "reset_map", "flush_map", "ground_object",
                        "ground_frame", "shutdown",
                    }
                    if cacheable and request_id in self._response_cache:
                        resp, resp_payload = self._response_cache[request_id]
                    else:
                        resp, resp_payload = self.handle_message(header, payload)
                        if cacheable and request_id:
                            self._response_cache[request_id] = (
                                resp, resp_payload)
                            while len(self._response_cache) > 16:
                                self._response_cache.pop(
                                    next(iter(self._response_cache)))
                    send_msg(conn, resp, resp_payload)
                    if resp.get("shutdown"):
                        print("[server] 收到 shutdown，退出")
                        conn.close()
                        return
            except (ConnectionError, OSError):
                print("[server] 客户端断开")
            finally:
                conn.close()


class _NullRetrieval:
    """SALAD 缺失时的空检索器：不产生任何回环。"""

    def get_all_submap_embeddings(self, submap):
        return []

    def find_loop_closures(self, *args, **kwargs):
        return []


def main():
    args = _parse_args()
    server = MappingServer(args)
    server.serve()


if __name__ == "__main__":
    main()
