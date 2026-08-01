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
                        help="关闭 CLIP 语义记忆（query_text/ground_object 不可用）")
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

        # 语义层：CLIP 关键帧记忆（随子图处理顺带算向量）+
        # SAM3 实例定位（查询时懒加载，不常驻显存）
        self.clip = None
        if not args.no_semantic:
            from mapping.semantic import HFClipAdapter, Sam3Grounder
            print("[server] 加载 CLIP (openai/clip-vit-base-patch32)...")
            self.clip = HFClipAdapter()
            self.grounder = Sam3Grounder()
            print("[server] CLIP 加载完成")

        self.solver_lock = threading.Lock()  # 保证同时只有一个子图在处理
        self.data_lock = threading.Lock()    # 保护 solver 内部状态
        self.gpu_lock = threading.Lock()     # VGGT/CLIP/SAM3 不并发抢显存

        self.keyframe_paths = []
        self.target_size = args.submap_size + args.overlapping_window_size
        self.num_frames = 0
        self.num_submaps_launched = 0
        self._candidate_seq = 0
        self._ground_candidates = {}
        self._response_cache = {}

        os.makedirs(args.keyframe_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 帧输入与子图处理
    # ------------------------------------------------------------------
    def feed_frame(self, rgb):
        """喂入一帧 RGB (H, W, 3) uint8。返回处理信息 dict。"""
        self.num_frames += 1
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
                if self.clip is not None:
                    predictions = self.solver.run_predictions(
                        image_paths, self.model, max_loops,
                        self.clip, self.clip.image_preprocess)
                else:
                    predictions = self.solver.run_predictions(
                        image_paths, self.model, max_loops, None, None)
            with self.data_lock:
                self.solver.add_points(predictions)
                self.solver.graph.optimize()
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
    def _semantic_topk(self, text, top_k):
        """CLIP 检索 top-K 关键帧，返回 [(score, submap_id, frame_index)]。

        按 frame_id 去重：子图重叠帧会出现在相邻两个子图中，
        同一物理帧只保留最高分的一项。
        """
        from vggt_slam.slam_utils import compute_text_embeddings

        prompts = self.grounder.expand_prompts(text)
        with self.gpu_lock:
            text_emb = np.vstack([
                compute_text_embeddings(
                    self.clip, self.clip.text_tokenizer, prompt)[0]
                for prompt in prompts
            ])
        best_by_frame = {}  # frame_id -> (score, submap_id, frame_index)
        with self.data_lock:
            for submap in self.solver.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue
                vecs = submap.get_all_semantic_vectors()
                if vecs is None or len(vecs) == 0:
                    continue
                vecs = np.asarray(vecs)
                sims = np.max(vecs @ text_emb.T, axis=1)
                frame_ids = submap.get_frame_ids()
                sid = submap.get_id()
                for idx in np.argsort(-sims)[:top_k]:
                    fid = int(frame_ids[idx]) if idx < len(frame_ids) else -1
                    key = fid if fid >= 0 else (sid, int(idx))
                    score = float(sims[idx])
                    if key not in best_by_frame or score > best_by_frame[key][0]:
                        best_by_frame[key] = (score, sid, int(idx))
        cands = sorted(best_by_frame.values(), key=lambda c: -c[0])
        return cands[:top_k]

    def query_text(self, text, top_k):
        """文本 -> top-K 关键帧 (frame_id, score, 位姿)。"""
        if self.clip is None:
            return {"results": [], "error": "semantic disabled"}
        topk = self._semantic_topk(text, top_k)
        results = []
        with self.data_lock:
            for score, sid, idx in topk:
                submap = self.solver.map.get_submap(sid)
                frame_ids = submap.get_frame_ids()
                pose = submap.get_all_poses_world(self.solver.graph)[idx]
                results.append({
                    "score": score,
                    "submap_id": sid,
                    "frame_index": idx,
                    "frame_id": int(frame_ids[idx]) if idx < len(frame_ids) else -1,
                    "pose": np.asarray(pose).tolist(),
                })
        return {"results": results}

    def ground_object(self, text, top_k):
        """文本 -> top-K 关键帧 SAM3 分割 -> 3D 目标点（查询时按需调用）。"""
        if self.clip is None:
            return {"results": [], "error": "semantic disabled"}
        from torchvision.transforms.functional import to_pil_image

        results = []
        prompts = self.grounder.expand_prompts(text)
        for item in self.query_text(text, top_k)["results"]:
            sid, idx = item["submap_id"], item["frame_index"]
            with self.data_lock:
                submap = self.solver.map.get_submap(sid)
                frame = submap.get_frame_at_index(idx)
            with self.gpu_lock:
                masks, boxes, scores, best_prompt = self.grounder.ground(
                    to_pil_image(frame), prompts)
            entry = dict(item)
            if len(masks) == 0:
                entry["found"] = False
                results.append(entry)
            else:
                for mask_index in np.argsort(-scores):
                    candidate = self._register_candidate(
                        sid, idx, item["frame_id"], masks[mask_index],
                        boxes[mask_index], scores[mask_index], best_prompt)
                    resolved = self.resolve_candidate(candidate["candidate_id"])
                    instance = dict(entry)
                    instance.update(candidate)
                    instance.update(resolved)
                    results.append(instance)
        return {"results": results}

    def _register_candidate(self, sid, idx, frame_id, mask, bbox, score, prompt):
        self._candidate_seq += 1
        candidate_id = f"c{self._candidate_seq}"
        self._ground_candidates[candidate_id] = {
            "submap_id": sid,
            "frame_index": int(idx),
            "frame_id": int(frame_id),
            "mask": np.asarray(mask, dtype=bool),
            "bbox": np.asarray(bbox, dtype=np.float32),
            "sam_score": float(score),
            "prompt": prompt,
        }
        while len(self._ground_candidates) > 128:
            self._ground_candidates.pop(next(iter(self._ground_candidates)))
        return {
            "candidate_id": candidate_id,
            "sam_score": float(score),
            "bbox": np.asarray(bbox).tolist(),
            "matched_prompt": prompt,
        }

    def resolve_candidate(self, candidate_id):
        """在当前图优化结果下重投影缓存的像素 mask。"""
        cand = self._ground_candidates.get(str(candidate_id))
        if cand is None:
            return {"found": False, "error": "unknown candidate"}
        with self.data_lock:
            submap = self.solver.map.get_submap(cand["submap_id"])
            mask = cand["mask"].copy()
            conf = np.asarray(submap.get_conf_masks_frame(
                cand["frame_index"])) > submap.get_conf_threshold()
            if conf.shape == mask.shape:
                mask &= conf
            pts = submap.get_points_in_mask(
                cand["frame_index"], mask, self.solver.graph)
        pts = np.asarray(pts)
        if len(pts) < 10:
            return {"found": False, "num_points": int(len(pts))}
        obb = self._robust_obb(pts)
        return {
            "found": True,
            "point": np.median(pts, axis=0).tolist(),
            "num_points": int(len(pts)),
            "obb": obb,
        }

    @staticmethod
    def _robust_obb(points):
        """用 PCA + 2/98 分位边界生成抗离群点的 3D OBB。"""
        points = np.asarray(points, dtype=np.float64)
        origin = np.median(points, axis=0)
        centered = points - origin
        _, axes = np.linalg.eigh(np.cov(centered, rowvar=False))
        axes = axes[:, ::-1]
        local = centered @ axes
        lo, hi = np.percentile(local, [2.0, 98.0], axis=0)
        center = origin + (0.5 * (lo + hi)) @ axes.T
        return {
            "center": center.tolist(),
            "extent": (hi - lo).tolist(),
            "rotation": axes.tolist(),
        }

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
        """对单张实时 RGB 做 SAM3 分割（到达前的视觉确认，不查记忆）。

        返回 {found, score, bbox, mask_ratio}。grounder 只在 serve 主循环
        中使用，无需加锁。"""
        if self.clip is None:
            return {"found": False, "error": "semantic disabled"}
        from PIL import Image
        with self.gpu_lock:
            masks, boxes, scores, _ = self.grounder.ground(
                Image.fromarray(rgb), self.grounder.expand_prompts(text))
        if len(masks) == 0:
            return {"found": False, "score": 0.0}
        best = int(np.argmax(scores))
        h, w = rgb.shape[:2]
        return {
            "found": True,
            "score": float(scores[best]),
            "bbox": np.asarray(boxes[best]).tolist(),
            "mask_ratio": float(masks[best].sum() / (h * w)),
        }

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
