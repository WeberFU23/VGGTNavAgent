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

from mapping.keyframes import (
    SUPPORTED_SUBMAP_OVERLAP,
    AdaptiveKeyframeSelector,
    pop_submap_window,
    validate_supported_overlap,
)
from runtime_paths import run_debug_path


def _parse_args():
    parser = argparse.ArgumentParser(description="VGGT-SLAM mapping server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument(
        "--overlapping-window-size", type=int,
        default=SUPPORTED_SUBMAP_OVERLAP,
        help="相邻子图共享帧数；当前上游 add_edge 仅正确支持 1")
    parser.add_argument("--max-loops", type=int, default=1,
                        help="0 关闭回环检测（SALAD ckpt 缺失时也会自动关闭）")
    parser.add_argument(
        "--min-disparity", type=float, default=40,
        help="相对上一关键帧的平均光流像素阈值")
    parser.add_argument(
        "--max-keyframe-interval", type=int, default=3,
        help="最多允许连续多少个观测不刷新关键帧，防止弱纹理直行断链")
    parser.add_argument("--conf-threshold", type=float, default=25.0)
    parser.add_argument("--lc-thres", type=float, default=0.95)
    parser.add_argument("--keyframe-dir", type=str,
                        default=run_debug_path("mapping", "keyframes"),
                        help="关键帧临时落盘目录（复用 VGGT 官方预处理）")
    parser.add_argument("--vis", action="store_true",
                        help="开启 viser 可视化（占用 8080 端口）")
    parser.add_argument("--no-semantic", action="store_true",
                        help="关闭 caption/pointing 语义记忆（语义查询不可用）")
    parser.add_argument("--diag-dir", type=str,
                        default=run_debug_path("mapping", "diagnostics"),
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
        self._closed = False
        self._server_socket = None
        self._frame_save_warned = False
        # 已处理完 submap 的最大帧号：只有这些帧能被 _locate_frame 检索。
        # 最新 feed 帧可能仍在 keyframe 缓冲/子图处理中，direct 查询会
        # "unknown frame_id"，故决策层 current_frame_id 必须用它而不是
        # feed 帧号。int 读写原子，与 num_frames 同风格不加锁。
        self._last_available_frame_id = 0

        if not torch.cuda.is_available():
            raise RuntimeError("mapping server 需要可用的 CUDA GPU")
        if args.submap_size < 2:
            raise ValueError("submap-size 必须至少为 2")
        validate_supported_overlap(args.overlapping_window_size)

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
            expected_salad = os.path.join(
                torch.hub.get_dir(), "checkpoints", "dino_salad.ckpt")
            print(f"[server] WARNING: {expected_salad} 缺失，回环检测已禁用")

        self.solver = OnlineSolver(
            init_conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres,
        )
        self.keyframe_selector = AdaptiveKeyframeSelector(
            self.solver.flow_tracker,
            min_disparity=args.min_disparity,
            max_interval=args.max_keyframe_interval,
        )
        if not self.use_loop_closure:
            # 用一个返回空结果的 dummy 替换 SALAD 检索，避免依赖 ckpt。
            self.solver.image_retrieval = _NullRetrieval()

        default_ckpt = os.path.join(
            torch.hub.get_dir(), "checkpoints", "model.pt")
        checkpoint = os.environ.get("VGGT_MODEL_CKPT", default_ckpt)
        allow_download = os.environ.get("VGGT_ALLOW_DOWNLOAD", "0") == "1"
        if not os.path.isfile(checkpoint) and not allow_download:
            raise RuntimeError(
                f"VGGT-1B 权重未找到: {checkpoint}。为避免意外下载大模型，"
                "请设置 VGGT_MODEL_CKPT 指向已有 model.pt；确认需要下载时才"
                "设置 VGGT_ALLOW_DOWNLOAD=1")
        print(f"[server] 加载 VGGT-1B 权重: {checkpoint}")
        from vggt.models.vggt import VGGT
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        if os.path.isfile(checkpoint) and \
                os.path.realpath(checkpoint) != os.path.realpath(default_ckpt):
            state_dict = torch.load(checkpoint, map_location="cpu")
        else:
            # torch hub 命中 checkpoints/model.pt 时不会重新下载。
            state_dict = torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(state_dict)
        model.eval()
        major, _minor = torch.cuda.get_device_capability()
        model_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.model = model.to(dtype=model_dtype, device="cuda")
        print("[server] 模型加载完成")

        self.solver_lock = threading.Lock()  # 保证同时只有一个子图在处理
        self.data_lock = threading.Lock()    # 保护 solver 内部状态
        self.gpu_lock = threading.Lock()     # VGGT/语义模型不并发抢显存

        # 诊断设施必须先于语义 worker 初始化，因为 vLLM trace 回调依赖它。
        self.diag_dir = args.diag_dir
        self.diag_lock = threading.Lock()
        self._current_episode = None
        self._diag_fp = None
        self._vlm_trace_fps = {}
        self._frame_manifest_fp = None
        self._diag_frame_dir = None
        self._diag_write_warned = False
        self._frame_manifest_warned = False
        os.makedirs(self.diag_dir, exist_ok=True)

        # 唯一语义链路：查询无关 caption + BGE-M3 检索 + VLM pointing。
        self.retrieve_top_k = int(os.environ.get("NAV_RETRIEVE_TOP_K", "2"))
        self.point_patch = int(os.environ.get("NAV_POINT_PATCH", "11"))
        self.vllm = None
        self.embedder = None
        self.caption_store = None
        self.caption_worker = None
        self.pointer = None
        if not args.no_semantic:
            self._init_semantic_memory()

        # SAM mask 精化 + SoM 全分割；未安装/无权重时自动禁用，退回旧行为。
        from mapping.sam_backend import SAMRefiner
        self.sam = SAMRefiner.from_env()
        if self.sam.available:
            print(f"[server] SAM 就绪: {self.sam.ckpt} ({self.sam.model_type})",
                  flush=True)
        else:
            print(f"[server] SAM 不可用（{self.sam.disabled_reason}），"
                  "mask 精化/SoM 禁用", flush=True)
        self._som_cache = {}
        self._som_cache_order = []

        self.keyframe_paths = []
        self.target_size = args.submap_size + args.overlapping_window_size
        self.num_frames = 0
        self.num_submaps_launched = 0
        self._candidate_seq = 0
        self._ground_candidates = {}
        self._response_cache = {}

        os.makedirs(args.keyframe_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # caption / pointing 语义记忆
    # ------------------------------------------------------------------
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
            api_key=os.environ.get("NAV_VLLM_API_KEY", "EMPTY"),
            trace_fn=lambda record: self._diag_write(
                {"cmd": "vlm_call", **record}))

        # caption 可走独立的高级 VLM API（pointing 仍留本地 Qwen2.5-VL）。
        # NAV_CAPTION_API_MODEL 必填；URL/Key 缺省回落到决策 VLM 的配置。
        caption_api_model = os.environ.get("NAV_CAPTION_API_MODEL", "").strip()
        caption_api_url = (os.environ.get("NAV_CAPTION_API_URL")
                           or os.environ.get("NAV_VLM_API_URL", "")).strip()
        # caption 走远端 API 不占 GPU，可并发消化积压（DashScope 按 QPS
        # 限流，超限返回 429，网关会退避重试）；本地 vLLM 保持串行。
        caption_workers = int(os.environ.get("NAV_CAPTION_WORKERS", "4"))
        caption_thinking = os.environ.get(
            "NAV_CAPTION_ENABLE_THINKING", "0").strip().lower() in {
            "1", "true", "yes", "on"}
        if caption_api_model and caption_api_url:
            self.vllm_caption = VLLMGateway(
                url=caption_api_url,
                api_key=(os.environ.get("NAV_CAPTION_API_KEY")
                         or os.environ.get("NAV_VLM_API_KEY", "")),
                trace_fn=lambda record: self._diag_write(
                    {"cmd": "vlm_call", **record}),
                workers=caption_workers,
                enable_thinking=caption_thinking)
            print(f"[server] caption 走独立 API: {caption_api_model} @ "
                  f"{caption_api_url} (workers={caption_workers}, "
                  f"thinking={caption_thinking})", flush=True)
            require_caption_probe = os.environ.get(
                "NAV_REQUIRE_CAPTION_PREFLIGHT", "1").strip().lower() in {
                    "1", "true", "yes", "on"}
            if require_caption_probe:
                try:
                    probe = self.vllm_caption.probe_chat(
                        caption_api_model, timeout=float(os.environ.get(
                            "NAV_CAPTION_HEALTH_TIMEOUT", "15")))
                except Exception as exc:
                    raise RuntimeError(
                        f"CAPTION_BACKEND_UNAVAILABLE: {exc}") from exc
                print("[server] caption live generation 就绪: "
                      f"{probe['model']} @ {probe['url']}", flush=True)
        else:
            self.vllm_caption = None
        store_path = os.environ.get(
            "NAV_CAPTION_STORE_PATH",
            os.path.join(self.args.diag_dir, "caption_store"))
        self.caption_store = CaptionStore(persist_dir=store_path)
        try:
            self.embedder = BGEM3Embedder(
                os.environ.get("NAV_EMBED_MODEL_PATH", ""),
                device=os.environ.get("NAV_EMBED_DEVICE", "cuda"),
            )
        except RuntimeError as exc:
            print(f"[server] WARNING: {exc}；caption 检索不可用", flush=True)
            self.embedder = None
        if self.vllm_caption is not None:
            caption_gateway = self.vllm_caption
            caption_model = caption_api_model
        else:
            caption_gateway = self.vllm
            caption_model = os.environ.get("NAV_CAPTION_MODEL_PATH", "")
        if self.embedder is not None and caption_model:
            self.caption_worker = CaptionWorker(
                caption_gateway, self.embedder, self.caption_store,
                model=caption_model,
                busy_fn=lambda: self.gpu_lock.locked()
                or self.solver_lock.locked(),
                result_fn=self._caption_diag_result,
                workers=(caption_workers
                         if self.vllm_caption is not None else 1))
            print("[server] caption worker 已启动 "
                  f"(model={caption_model})", flush=True)
        else:
            print("[server] WARNING: caption worker 未启动（缺 embedder 或 "
                  "NAV_CAPTION_MODEL_PATH）", flush=True)
        self.pointer = PointingGrounder(
            self.vllm, model=os.environ.get("NAV_POINTING_MODEL_PATH", ""))
        try:
            health = self.pointer.check_health(timeout=float(os.environ.get(
                "NAV_POINTING_HEALTH_TIMEOUT", "10")))
            print("[server] pointing grounder 就绪: "
                  f"{health['url']} model={self.pointer.model}", flush=True)
        except Exception as exc:  # pointing 后端缺失不应阻止启动（som 主链路不依赖它）
            print(f"[server] WARNING: pointing grounder 未就绪（{exc}），"
                  "point_pixels 将返回 POINTING_BACKEND_UNAVAILABLE",
                  flush=True)

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
                saved = cv2.imwrite(
                    os.path.join(
                        self._diag_frame_dir,
                        f"rgb_{self.num_frames:06d}.jpg"),
                    bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not saved:
                    raise OSError("cv2.imwrite returned false")
            except Exception as exc:  # OpenCV/文件系统失败不应中断建图
                if not self._frame_save_warned:
                    print(f"[server] WARNING: RGB 诊断图保存失败: {exc}",
                          flush=True)
                    self._frame_save_warned = True
        is_keyframe, keyframe_reason = self.keyframe_selector.select(bgr)
        self._frame_manifest_write({
            "event": "frame_saved",
            "frame_id": int(self.num_frames),
            "image": f"rgb_{self.num_frames:06d}.jpg",
            "is_keyframe": bool(is_keyframe),
            "keyframe_reason": keyframe_reason,
            "frames_since_keyframe": (
                self.keyframe_selector.frames_since_keyframe),
            "caption_expected": bool(is_keyframe and self.caption_worker),
        })

        if is_keyframe:
            path = os.path.join(
                self.args.keyframe_dir, f"frame_{self.num_frames:06d}.png")
            if not cv2.imwrite(path, bgr):
                raise OSError(f"关键帧保存失败: {path}")
            self.keyframe_paths.append(path)

        launched = False
        if len(self.keyframe_paths) >= self.target_size:
            acquired = self.solver_lock.acquire(blocking=False)
            if not acquired and len(self.keyframe_paths) >= self.target_size * 2:
                # 后端持续落后时施加背压，避免旧实现截断队列后丢失桥接帧。
                print(f"[server] {time.strftime('%H:%M:%S')} 关键帧积压 "
                      f"{len(self.keyframe_paths)}，等待上一子图完成", flush=True)
                self.solver_lock.acquire()
                acquired = True
            if acquired:
                self._launch_buffered_submap()
                launched = True
            elif self.num_frames % 20 == 0:
                print(f"[server] {time.strftime('%H:%M:%S')} 后端忙，"
                      f"缓冲 {len(self.keyframe_paths)} 帧", flush=True)

        return {
            "frame_id": int(self.num_frames),
            "last_available_frame_id": int(self._last_available_frame_id or 0),
            "is_keyframe": bool(is_keyframe),
            "keyframe_reason": keyframe_reason,
            "frames_since_keyframe": (
                self.keyframe_selector.frames_since_keyframe),
            "queued_keyframes": len(self.keyframe_paths),
            "submap_launched": launched,
            "busy": self.solver_lock.locked(),
        }

    def _launch_buffered_submap(self):
        """启动队首固定窗口；调用方必须已经取得 solver_lock。"""
        image_paths, self.keyframe_paths = pop_submap_window(
            self.keyframe_paths, self.target_size,
            self.args.overlapping_window_size)
        self.num_submaps_launched += 1
        print(f"[server] {time.strftime('%H:%M:%S')} 启动子图 "
              f"#{self.num_submaps_launched}, {len(image_paths)} 帧, "
              f"待处理 {len(self.keyframe_paths)} 帧", flush=True)
        thread = threading.Thread(
            target=self._process_submap,
            args=(image_paths,),
            daemon=True,
        )
        thread.start()

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
                submap = self.solver.map.get_latest_submap()
                if submap is not None:
                    fids = submap.get_frame_ids()
                    if fids:
                        self._last_available_frame_id = max(
                            int(f) for f in fids)
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

            # 先按固定窗口清空完整子图，避免 episode 结束时把积压帧一次性
            # 送入 VGGT 导致显存峰值和配准退化。
            while len(self.keyframe_paths) >= self.target_size:
                paths, self.keyframe_paths = pop_submap_window(
                    self.keyframe_paths, self.target_size,
                    self.args.overlapping_window_size)
                self.num_submaps_launched += 1
                print(f"[server] 同步提交完整子图 "
                      f"#{self.num_submaps_launched}, {len(paths)} 帧",
                      flush=True)
                self._process_submap(paths, release_solver_lock=False)

            overlap = self.args.overlapping_window_size \
                if self.num_submaps_launched > 0 else 0
            useful = len(self.keyframe_paths) - overlap
            if useful > 0 and len(self.keyframe_paths) >= 2:
                paths = list(self.keyframe_paths)
                self.num_submaps_launched += 1
                print(f"[server] 同步提交尾部子图 "
                      f"#{self.num_submaps_launched}, {len(paths)} 帧",
                      flush=True)
                self._process_submap(paths, release_solver_lock=False)
            self.keyframe_paths = []
        return {"flushed": True, "queued_keyframes": 0}

    def reset_map(self):
        """清空地图，开始新 episode。复用同一个 Solver 以免重载 SALAD。"""
        with self.solver_lock:  # 等待在途子图完成
            with self.data_lock:
                self.solver.map = self._GraphMap()
                self.solver.graph = self._PoseGraph()
                self.solver.flow_tracker = self._FrameTracker()
                self.keyframe_selector = AdaptiveKeyframeSelector(
                    self.solver.flow_tracker,
                    min_disparity=self.args.min_disparity,
                    max_interval=self.args.max_keyframe_interval,
                )
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
        if self.vllm is not None:
            self.vllm.clear_cache()
        self._recreate_keyframe_dir()
        return {"reset": True}

    def _recreate_keyframe_dir(self):
        """只重建明确配置的关键帧临时目录，拒绝宽泛危险路径。"""
        target = os.path.realpath(os.path.abspath(self.args.keyframe_dir))
        forbidden = {
            os.path.realpath(os.path.abspath(os.sep)),
            os.path.realpath(os.path.expanduser("~")),
            os.path.realpath(os.getcwd()),
        }
        if target in forbidden or not os.path.basename(target):
            raise RuntimeError(f"拒绝清空不安全的 keyframe-dir: {target}")
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.makedirs(target, exist_ok=True)

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
            "keyframe_policy": {
                "min_disparity": self.keyframe_selector.min_disparity,
                "max_interval": self.keyframe_selector.max_interval,
                "frames_since_keyframe": (
                    self.keyframe_selector.frames_since_keyframe),
                "forced_keyframes": self.keyframe_selector.num_forced,
                "submap_overlap": self.args.overlapping_window_size,
            },
            "busy": self.solver_lock.locked(),
            "semantic": {
                "caption_enabled": self.caption_worker is not None,
                "embedding_enabled": self.embedder is not None,
                "pointing_enabled": self.pointer is not None,
                "caption_errors": (self.caption_worker.errors
                                   if self.caption_worker is not None else 0),
            },
            # 语义记忆进度：caption worker 尚未消化完的关键帧数 / 最新完成帧。
            # agent 检索前据此等待，避免漏掉刚入图的关键帧。
            "caption_pending": (self.caption_worker.pending()
                                if self.caption_worker is not None else 0),
            "caption_last_completed": (
                self.caption_worker.last_completed_frame_id
                if self.caption_worker is not None else None),
        }

    def get_captioned_frame_ids(self):
        """返回当前 episode 已完成 caption 并写入语义库的关键帧。"""
        if self.caption_store is None:
            return {"enabled": False, "frame_ids": []}
        return {
            "enabled": self.caption_worker is not None,
            "frame_ids": [int(fid) for fid in self.caption_store.frame_ids],
        }

    def get_captions(self, frame_ids):
        """按 frame_id 批量取回 caption（决策层新关键帧编号通知用）。
        未入库的帧跳过。"""
        if self.caption_store is None:
            return {"captions": []}
        return {"captions": self.caption_store.get_captions(frame_ids)}

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

    def get_frame_pose(self, frame_id):
        """按 frame_id 返回优化后世界位姿；找不到返回 found=False。

        供 agent 把"geometry 解析失败"的候选重看目标导航到该帧拍摄位置。
        """
        with self.data_lock:
            frame_ids, poses = self._collect_poses()
        if poses is None:
            return {"found": False}
        pose_by_fid = {int(fid): pose for fid, pose in zip(frame_ids, poses)}
        pose = pose_by_fid.get(int(frame_id))
        if pose is None:
            return {"found": False, "frame_id": int(frame_id)}
        return {"found": True, "frame_id": int(frame_id),
                "pose": np.asarray(pose).tolist()}

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
        """原子返回逐帧世界点、RGB 和位姿。

        occupancy、frontier 和决策 VLM 鸟瞰图必须建立在同一次图优化
        快照上。每帧 payload 依次存放 ``N*3 float32`` 点和 ``N*3
        uint8`` RGB；低置信点保留为 NaN，方便客户端用同一 mask 同时
        过滤点和颜色。
        """
        stride = max(int(stride), 1)
        with self.data_lock:
            metas, chunks = [], []
            snapshot_revision = {
                "num_frames": int(self.num_frames),
                "num_submaps": int(self.solver.map.get_num_submaps()),
                "num_loop_closures": int(
                    self.solver.graph.get_num_loops()),
            }
            for submap in self.solver.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue
                poses = submap.get_all_poses_world(self.solver.graph)
                fids = submap.get_frame_ids()
                for index in range(len(submap.pointclouds)):
                    hom = self.solver.graph.get_homography(
                        index + submap.get_id())
                    pts = submap.pointclouds[index][::stride, ::stride, :]
                    colors = np.asarray(
                        submap.colors[index][::stride, ::stride, :],
                        dtype=np.uint8)
                    if colors.shape != pts.shape:
                        raise RuntimeError(
                            "VGGT point/color grid shape mismatch: "
                            f"{pts.shape} != {colors.shape}")
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
                        "has_colors": True,
                        "pose": np.asarray(
                            poses[index], dtype=np.float32).tolist(),
                    })
                    chunks.extend((world.tobytes(),
                                   colors.reshape(-1, 3).tobytes()))
        return {"frames": metas,
                "snapshot_revision": snapshot_revision}, b"".join(chunks)

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
        raw_episode_id = str(episode_id or "unknown").strip()
        episode_id = "".join(
            ch if ch.isalnum() or ch in "-_." else "_"
            for ch in raw_episode_id)[:120] or "unknown"
        if episode_id == self._current_episode:
            return
        if self.caption_worker is not None:
            self.caption_worker.clear()
        if self.vllm is not None:
            self.vllm.clear_cache()
        if self.caption_store is not None:
            self.caption_store.set_episode(episode_id)
        with self.diag_lock:
            if self._diag_fp is not None:
                try:
                    self._diag_fp.close()
                except Exception:
                    pass
                self._diag_fp = None
            if self._frame_manifest_fp is not None:
                try:
                    self._frame_manifest_fp.close()
                except Exception:
                    pass
                self._frame_manifest_fp = None
            self._current_episode = episode_id
            self._diag_frame_dir = os.path.join(
                self.diag_dir, f"{episode_id}_frames")
            os.makedirs(self._diag_frame_dir, exist_ok=True)
            self._diag_fp = open(
                os.path.join(self.diag_dir, f"{episode_id}_queries.jsonl"),
                "a", encoding="utf-8")
            self._frame_manifest_fp = open(
                os.path.join(
                    self.diag_dir, f"{episode_id}_frame_captions.jsonl"),
                "a", encoding="utf-8")
            print(f"[server] 诊断 episode={episode_id} -> {self.diag_dir}",
                  flush=True)

    def _diag_write(self, record):
        """线程安全地写一条诊断 JSONL。"""
        record = dict(record)
        record.setdefault("episode", self._current_episode)
        record.setdefault("t", time.strftime("%H:%M:%S"))
        with self.diag_lock:
            # VLM 调用（caption/pointing）按角色拆到独立文件，保留完整
            # prompt + 内联图像 + 输出；主 queries 文件只写无图像摘要。
            if record.get("cmd") == "vlm_call":
                self._vlm_trace_write(record)
            if self._diag_fp is None:
                return
            try:
                slim = {k: v for k, v in record.items() if k != "images"}
                self._diag_fp.write(json.dumps(slim) + "\n")
                self._diag_fp.flush()
            except OSError as exc:
                if not self._diag_write_warned:
                    print(f"[server] WARNING: 诊断查询日志写入失败: {exc}",
                          flush=True)
                    self._diag_write_warned = True

    def _vlm_trace_write(self, record):
        """把 vlm_call 记录按角色写入 vlm_{caption,pointing,other}.jsonl。"""
        kind = str(record.get("kind") or "other")
        role = {"caption": "caption", "point": "pointing",
                "verify": "pointing"}.get(kind, "other")
        fp = self._vlm_trace_fps.get(role)
        if fp is None:
            path = os.path.join(self.diag_dir, f"vlm_{role}.jsonl")
            fp = open(path, "a", encoding="utf-8")
            self._vlm_trace_fps[role] = fp
        try:
            fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fp.flush()
        except OSError as exc:
            if not self._diag_write_warned:
                print(f"[server] WARNING: VLM trace 写入失败: {exc}",
                      flush=True)
                self._diag_write_warned = True

    def _frame_manifest_write(self, record):
        """Append frame/caption events joined by frame_id."""
        item = dict(record)
        item.setdefault("episode", self._current_episode)
        item.setdefault("t", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self.diag_lock:
            if self._frame_manifest_fp is None:
                return
            try:
                self._frame_manifest_fp.write(json.dumps(
                    item, ensure_ascii=False, default=str) + "\n")
                self._frame_manifest_fp.flush()
            except OSError as exc:
                if not self._frame_manifest_warned:
                    print(f"[server] WARNING: frame/caption 日志写入失败: {exc}",
                          flush=True)
                    self._frame_manifest_warned = True

    def _caption_diag_result(self, record):
        item = dict(record)
        frame_id = int(item.get("frame_id"))
        item.update({
            "event": "caption_result",
            "image": f"rgb_{frame_id:06d}.jpg",
        })
        self._frame_manifest_write(item)
        self._diag_write({"cmd": "caption_result", **item})

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

    def ground_object_pixels(self, text, top_k):
        """文本 -> caption 检索 + pointing，仅返回待审核像素。"""
        if self.pointer is None or self.embedder is None:
            return {"results": [], "error": "semantic memory disabled"}
        from torchvision.transforms.functional import to_pil_image
        results = []
        for item in self.retrieve_captions(text, max(int(top_k),
                                                     self.retrieve_top_k))["results"]:
            fid = int(item["frame_id"])
            located = self._locate_frame(fid)
            if located is None:
                continue
            sid, idx = located
            with self.data_lock:
                submap = self.solver.map.get_submap(sid)
                frame = submap.get_frame_at_index(idx)
            pil = to_pil_image(frame)
            frame_key = f"{self._current_episode or 'unknown'}_frame_{fid}"
            try:
                pts = self.pointer.point(pil, text, frame_key=frame_key)
            except Exception as exc:
                from mapping.pointing import PointingBackendUnavailable
                if isinstance(exc, PointingBackendUnavailable):
                    return {"results": [],
                            "error_code": "POINTING_BACKEND_UNAVAILABLE",
                            "error": str(exc)[:300]}
                raise
            w, h = pil.size
            for pt in pts:
                x, y = pt["pixel"]
                results.append({
                    "found": True, "frame_id": fid,
                    "pixel": [float(x) / w * 1000.0,
                               float(y) / h * 1000.0],
                    "point_score": float(pt["confidence"]),
                    "bbox": pt.get("bbox"), "text": item.get("caption", text),
                })
        return {"results": results}

    def point_frame(self, frame_id, text):
        """对指定关键帧直接 pointing + 3D 采样（跳过 caption 检索）。

        决策层定帧 ground_target 的服务端实现：VLM 已通过 view_frame/
        search_frames 选定帧，这里只做定位与实例化。
        """
        if self.pointer is None:
            return {"results": [], "error": "semantic memory disabled"}
        from torchvision.transforms.functional import to_pil_image

        located = self._locate_frame(frame_id)
        if located is None:
            return {"results": [], "error": f"unknown frame_id {frame_id}"}
        sid, idx = located
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            frame = submap.get_frame_at_index(idx)
        pil = to_pil_image(frame)
        fid = int(frame_id)
        frame_key = f"{self._current_episode or 'unknown'}_frame_{fid}_direct"
        try:
            pts = self.pointer.point(pil, text, frame_key=frame_key)
        except Exception as exc:
            from mapping.pointing import PointingBackendUnavailable
            if isinstance(exc, PointingBackendUnavailable):
                return {"results": [],
                        "error_code": "POINTING_BACKEND_UNAVAILABLE",
                        "error": str(exc)[:300]}
            raise
        results = []
        for pt in pts:
            x, y = pt["pixel"]
            ref = self._refine_point_with_sam(sid, idx, fid, x, y)
            pt = {"pixel": (ref["x"], ref["y"]),
                  "confidence": pt["confidence"],
                  "bbox": ref["bbox"] if ref["refined"] else pt.get("bbox"),
                  "mask": ref["mask"]}
            resolved = self._resolve_point(sid, idx, fid, pt)
            resolved["frame_id"] = fid
            results.append(resolved)
        self._diag_write({
            "cmd": "point_frame", "text": text, "frame_id": fid,
            "num_points": len(results),
            "points": [{"pixel": list(pt["pixel"]),
                        "point_score": pt["confidence"],
                        "found": r.get("found", False)}
                       for pt, r in zip(pts, results)],
        })
        return {"results": results}


    def point_pixels(self, frame_id, text):
        """仅指向：对指定关键帧 pointing，返回像素坐标，不做 3D 采样。

        供决策层 point 工具使用；结果中的像素为原图坐标系。
        """
        if self.pointer is None:
            return {"points": [], "error": "semantic memory disabled"}
        from torchvision.transforms.functional import to_pil_image

        located = self._locate_frame(frame_id)
        if located is None:
            return {"points": [], "error": f"unknown frame_id {frame_id}"}
        sid, idx = located
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            frame = submap.get_frame_at_index(idx)
        pil = to_pil_image(frame)
        fid = int(frame_id)
        frame_key = f"{self._current_episode or 'unknown'}_frame_{fid}_direct"
        try:
            pts = self.pointer.point(pil, text, frame_key=frame_key)
        except Exception as exc:
            from mapping.pointing import PointingBackendUnavailable
            if isinstance(exc, PointingBackendUnavailable):
                return {"points": [],
                        "error_code": "POINTING_BACKEND_UNAVAILABLE",
                        "error": str(exc)[:300]}
            raise
        # SAM 点提示精化：粗落点替换为物体 mask 质心，附 mask bbox 与面积占比
        refined = []
        for pt in pts:
            x, y = pt["pixel"]
            ref = self._refine_point_with_sam(sid, idx, fid, x, y)
            row = {"pixel": (ref["x"], ref["y"]),
                   "confidence": pt["confidence"],
                   "bbox": ref["bbox"] if ref["refined"] else pt.get("bbox"),
                   "refined": bool(ref["refined"])}
            if ref["area_frac"] is not None:
                row["area_frac"] = ref["area_frac"]
            refined.append(row)
        pts = refined
        self._diag_write({
            "cmd": "point_pixels", "text": text, "frame_id": fid,
            "num_points": len(pts),
            "points": [{"pixel": list(pt["pixel"]),
                        "point_score": pt["confidence"],
                        "refined": pt.get("refined", False)} for pt in pts],
        })
        return {"width": int(pil.size[0]), "height": int(pil.size[1]),
                "points": [
            {"pixel": [float(pt["pixel"][0]), float(pt["pixel"][1])],
             "confidence": float(pt["confidence"]),
             "bbox": pt.get("bbox"),
             "refined": bool(pt.get("refined", False)),
             "area_frac": pt.get("area_frac")} for pt in pts]}

    def instantiate_pixels(self, frame_id, pixels, normalized=True):
        """按给定像素实例化：patch 深度采样 + 候选注册。

        pixels: [[x, y], ...]；normalized=True 时为 0-1000 归一化坐标
        （决策 VLM 与 harness 之间的统一坐标约定），按点云网格宽高换算。
        """
        located = self._locate_frame(frame_id)
        if located is None:
            return {"results": [], "error": f"unknown frame_id {frame_id}"}
        sid, idx = located
        fid = int(frame_id)
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            h, w = submap.pointclouds[idx].shape[:2]
        results = []
        for px in list(pixels or [])[:16]:
            try:
                x, y = float(px[0]), float(px[1])
            except (TypeError, ValueError, IndexError):
                continue
            if normalized:
                x, y = x / 1000.0 * w, y / 1000.0 * h
            x = min(max(x, 0.0), w - 1)
            y = min(max(y, 0.0), h - 1)
            ref = self._refine_point_with_sam(sid, idx, fid, x, y)
            pt = {"pixel": (ref["x"], ref["y"]), "confidence": 1.0,
                  "bbox": ref["bbox"] if ref["refined"] else None,
                  "mask": ref["mask"]}
            resolved = self._resolve_point(sid, idx, fid, pt)
            resolved["frame_id"] = fid
            results.append(resolved)
        self._diag_write({
            "cmd": "instantiate_pixels", "frame_id": fid,
            "num_points": len(results),
            "points": [{"pixel": r.get("pixel"),
                        "found": r.get("found", False)} for r in results],
        })
        return {"results": results}

    def prepare_pixels(self, frame_id, pixels, normalized=True):
        """仅准备视觉审核候选；审核通过后才允许解析 3D 深度。"""
        located = self._locate_frame(frame_id)
        if located is None:
            return {"candidates": [], "error": f"unknown frame_id {frame_id}"}
        sid, idx = located
        fid = int(frame_id)
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            h, w = submap.pointclouds[idx].shape[:2]
        candidates = []
        for px in list(pixels or [])[:16]:
            try:
                x, y = float(px[0]), float(px[1])
            except (TypeError, ValueError, IndexError):
                continue
            if normalized:
                x, y = x / 1000.0 * w, y / 1000.0 * h
            x = min(max(x, 0.0), w - 1)
            y = min(max(y, 0.0), h - 1)
            ref = self._refine_point_with_sam(sid, idx, fid, x, y)
            meta = self._register_point_candidate(
                sid, idx, fid, (ref["x"], ref["y"]),
                ref["bbox"] if ref["refined"] else None, 1.0,
                mask=ref["mask"])
            candidates.append({
                "candidate_id": meta["candidate_id"],
                "frame_id": fid,
                "pixel": [ref["x"], ref["y"]],
                "pixel_norm": meta["pixel_norm"],
                "point_score": 1.0,
                "bbox": meta.get("bbox"),
                "refined": bool(ref["refined"]),
            })
        return {"candidates": candidates}

    # ------------------------------------------------------------------
    # SoM 全分割：整帧 SAM 分割 -> 编号 overlay -> VLM 选 mask -> 注册候选
    # ------------------------------------------------------------------
    def som_segment(self, frame_id, max_masks=None):
        """对指定关键帧做全景分割，返回 mask 元数据 + 编号 overlay JPEG。

        mask 缓存在服务端（_som_cache），供后续 som_pick 按 mask_id 引用；
        overlay 中每个 mask 半透明着色并在质心标注 mask_id。
        """
        if self.sam is None or not self.sam.available:
            return {"found": False, "error_code": "SAM_UNAVAILABLE",
                    "error": f"SAM disabled: {getattr(self.sam, 'disabled_reason', None)}"}, b""
        located = self._locate_frame(int(frame_id))
        if located is None:
            return {"found": False, "error": f"unknown frame_id {frame_id}"}, b""
        sid, idx = located
        fid = int(frame_id)
        from torchvision.transforms.functional import to_pil_image
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            frame = submap.get_frame_at_index(idx)
        rgb = np.asarray(to_pil_image(frame))
        h, w = rgb.shape[:2]
        masks = self.sam.segment_all(rgb, max_masks=max_masks)
        # LRU 缓存：同帧重复分割直接换新结果，最多保留 8 帧
        self._som_cache[fid] = {"submap_id": sid, "frame_index": idx,
                                "masks": masks, "shape": (h, w)}
        if fid in self._som_cache_order:
            self._som_cache_order.remove(fid)
        self._som_cache_order.append(fid)
        while len(self._som_cache_order) > 8:
            oldest = self._som_cache_order.pop(0)
            self._som_cache.pop(oldest, None)
        from mapping.sam_backend import render_som_overlay
        jpeg = render_som_overlay(rgb, masks)
        rows = []
        for row in masks:
            cx, cy = row["centroid"]
            x0, y0, x1, y1 = row["bbox"]
            rows.append({
                "mask_id": int(row["mask_id"]),
                "centroid": [round(cx / w * 1000, 1), round(cy / h * 1000, 1)],
                "bbox": [round(x0 / w * 1000, 1), round(y0 / h * 1000, 1),
                         round(x1 / w * 1000, 1), round(y1 / h * 1000, 1)],
                "area_frac": round(float(row["area_frac"]), 4),
            })
        self._diag_write({"cmd": "som_segment", "frame_id": fid,
                          "num_masks": len(masks)})
        meta = {"found": True, "frame_id": fid, "width": w, "height": h,
                "masks": rows}
        if jpeg:
            meta["mime_type"] = "image/jpeg"
        return meta, (jpeg or b"")

    def som_pick(self, frame_id, mask_ids):
        """按 mask_id 注册候选（质心 + 实例 mask），供 commit 流程复用。"""
        fid = int(frame_id)
        cached = self._som_cache.get(fid)
        if cached is None:
            return {"candidates": [], "error_code": "SOM_CACHE_MISS",
                    "error": "no segmentation cached for this frame; "
                             "call som_segment first"}
        by_id = {int(row["mask_id"]): row for row in cached["masks"]}
        sid, idx = cached["submap_id"], cached["frame_index"]
        candidates = []
        for raw_id in list(mask_ids or [])[:16]:
            row = by_id.get(int(raw_id))
            if row is None or row.get("centroid") is None:
                continue
            cx, cy = row["centroid"]
            meta = self._register_point_candidate(
                sid, idx, fid, (cx, cy), row["bbox"], 1.0, mask=row["mask"])
            candidates.append({
                "candidate_id": meta["candidate_id"],
                "frame_id": fid,
                "mask_id": int(raw_id),
                "pixel": [cx, cy],
                "pixel_norm": meta["pixel_norm"],
                "point_score": 1.0,
                "bbox": meta.get("bbox"),
                "refined": True,
            })
        self._diag_write({"cmd": "som_pick", "frame_id": fid,
                          "mask_ids": [int(v) for v in (mask_ids or [])][:16],
                          "num_candidates": len(candidates)})
        return {"candidates": candidates}

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
            frame_key = f"{self._current_episode or 'unknown'}_frame_{fid}"
            try:
                pts = self.pointer.point(pil, text, frame_key=frame_key)
            except Exception as exc:
                from mapping.pointing import PointingBackendUnavailable
                if isinstance(exc, PointingBackendUnavailable):
                    return {"results": [],
                            "error_code": "POINTING_BACKEND_UNAVAILABLE",
                            "error": str(exc)[:300]}
                raise
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
        认出目标，导航端不能只用拍照位姿。point_info 带 mask 时优先在
        SAM mask 区域内采样（见 sample_point_depth）。
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
            cam_origin=cam_origin, bbox=point_info.get("bbox"),
            mask_hw=point_info.get("mask"))
        pixel_norm = [float(pixel[0]) / w * 1000.0,
                      float(pixel[1]) / h * 1000.0]
        if not sampled["found"]:
            return {"found": False, "pixel": [float(pixel[0]), float(pixel[1])],
                    "pixel_norm": pixel_norm,
                    "point_score": float(point_info["confidence"]),
                    "num_points": int(sampled["num_points"])}
        out = {
            "found": True,
            "point": [float(v) for v in sampled["point"]],
            "num_points": int(sampled["num_points"]),
            "depth_std": sampled["depth_std"],
            "spread": sampled["spread"],
            "pixel": [float(pixel[0]), float(pixel[1])],
            "pixel_norm": pixel_norm,
            "point_score": float(point_info["confidence"]),
        }
        if register:
            out.update(self._register_point_candidate(
                sid, idx, frame_id, pixel, point_info.get("bbox"),
                point_info["confidence"], mask=point_info.get("mask")))
        return out

    def _refine_point_with_sam(self, sid, idx, fid, x, y):
        """SAM 点提示精化：粗落点 -> 物体 mask -> 质心 + mask bbox。

        返回 {x, y, mask, bbox, area_frac, refined}；SAM 不可用或选中
        背景时 refined=False 且坐标原样返回（调用方退回旧行为）。
        """
        result = {"x": float(x), "y": float(y), "mask": None, "bbox": None,
                  "area_frac": None, "refined": False}
        if self.sam is None or not self.sam.available:
            return result
        from torchvision.transforms.functional import to_pil_image
        with self.data_lock:
            submap = self.solver.map.get_submap(sid)
            frame = submap.get_frame_at_index(idx)
        rgb = np.asarray(to_pil_image(frame))
        try:
            hit = self.sam.segment_at_point(
                rgb, (float(x), float(y)),
                cache_key=f"{self._current_episode or 'unknown'}_sam_{fid}")
        except Exception as exc:  # noqa: BLE001 - 单次失败退回旧行为
            print(f"[server] SAM 精化失败 frame={fid}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return result
        if hit is None or hit.get("centroid") is None:
            return result
        result.update({
            "x": float(hit["centroid"][0]), "y": float(hit["centroid"][1]),
            "mask": hit["mask"], "bbox": hit["bbox"],
            "area_frac": float(hit["area_frac"]), "refined": True,
        })
        return result

    def _register_point_candidate(self, sid, idx, frame_id, pixel, bbox,
                                  score, mask=None):
        """point 候选注册：存像素 + mask，供 resolve_candidate 在最新
        图优化坐标系下重采样；mask 为 SAM 实例 mask，缺省时合成小圆盘。"""
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
        if mask is not None and np.asarray(mask).shape == (h, w):
            disk = np.asarray(mask, dtype=bool)
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            disk = (xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2 \
                <= (max(self.point_patch, 8) / 2.0) ** 2
        # pixel_norm 与决策 VLM 的 pixels_1000 同约定（0-1000 归一化），
        # 供语义审核拒绝记录原样返回，避免 VLM 把原始帧坐标当归一化坐标。
        pixel_norm = [float(pixel[0]) / w * 1000.0,
                      float(pixel[1]) / h * 1000.0]
        self._ground_candidates[candidate_id] = {
            "submap_id": sid,
            "frame_index": int(idx),
            "frame_id": int(frame_id),
            "pixel": (float(pixel[0]), float(pixel[1])),
            "pixel_norm": pixel_norm,
            "mask": disk,
            "bbox": np.asarray([x0, y0, x1, y1], dtype=np.float32),
            "point_score": float(score),
        }
        while len(self._ground_candidates) > 128:
            self._ground_candidates.pop(next(iter(self._ground_candidates)))
        return {"candidate_id": candidate_id,
                "point_score": float(score),
                "bbox": [x0, y0, x1, y1],
                "pixel_norm": pixel_norm}

    def resolve_candidate(self, candidate_id):
        """在当前图优化结果下重新采样 pointing 像素的 3D 点。"""
        cand = self._ground_candidates.get(str(candidate_id))
        if cand is None:
            return {"found": False, "error": "unknown candidate"}
        resolved = self._resolve_point(
            cand["submap_id"], cand["frame_index"], cand["frame_id"],
            {"pixel": cand["pixel"], "confidence": cand["point_score"],
             "bbox": None, "mask": cand.get("mask")}, register=False)
        if not resolved.get("found"):
            return {"found": False,
                    "num_points": resolved.get("num_points", 0)}
        return {"found": True, "point": resolved["point"],
                "num_points": resolved["num_points"],
                "depth_std": resolved.get("depth_std")}

    def resolve_candidates(self, candidate_ids):
        """批量刷新候选坐标，避免 agent 为一张地图逐个建立 RPC。"""
        ids = list(dict.fromkeys(str(cid) for cid in candidate_ids or []))[:64]
        rows = {}
        for candidate_id in ids:
            try:
                rows[candidate_id] = self.resolve_candidate(candidate_id)
            except Exception as exc:
                rows[candidate_id] = {
                    "found": False,
                    "error": f"resolve failed: {type(exc).__name__}",
                }
        return {"candidates": rows}

    def candidate_evidence(self, candidate_id, wide_only=False):
        """生成候选证据 JPEG：默认左全帧+右放大裁剪；wide_only 只给全帧。

        wide_only 用于实例去重裁决：放大裁剪会主导 VLM 判断（同一物体
        不同角度/光照的局部差异被放大），而全帧里"标记在同一位置"的
        线索更可靠。
        """
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
        # Keep the exact sampling location visible without covering a small
        # object: a high-contrast crosshair is clearer to the verifier than a
        # translucent patch alone.
        px, py = (int(round(value)) for value in cand["pixel"])
        self._draw_crosshair(overlay, px, py)
        if wide_only:
            h, w = overlay.shape[:2]
            height = 320
            panel = cv2.resize(
                overlay, (max(1, int(round(w * height / max(h, 1)))), height),
                interpolation=cv2.INTER_AREA)
        else:
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

    def _draw_crosshair(self, overlay, px, py, radius=None):
        """高对比十字：黑 outline + 黄芯，标出验证/确认的精确像素。"""
        if radius is None:
            radius = max(10, self.point_patch)
        for thickness, color in ((5, (0, 0, 0)), (2, (255, 255, 0))):
            cv2.line(overlay, (max(0, px - radius), py),
                     (min(overlay.shape[1] - 1, px + radius), py),
                     color, thickness, cv2.LINE_AA)
            cv2.line(overlay, (px, max(0, py - radius)),
                     (px, min(overlay.shape[0] - 1, py + radius)),
                     color, thickness, cv2.LINE_AA)

    def evidence_for_point(self, frame_id, pixel, bbox=None):
        """按像素渲染十字证据面板（不注册候选）。

        供 use_molmo_point 把 pointing 模型标点渲染成 VLM 可确认的证据图：
        左半全帧 overlay + 十字，右半 bbox 放大裁剪 + 十字。与候选证据
        面板同形态，但完全无状态，不进入 _ground_candidates。
        """
        located = self._locate_frame(int(frame_id))
        if located is None:
            return {"found": False, "error": "unknown frame"}, b""
        from torchvision.transforms.functional import to_pil_image
        with self.data_lock:
            submap = self.solver.map.get_submap(located[0])
            rgb = np.asarray(to_pil_image(
                submap.get_frame_at_index(located[1])))
        overlay = rgb.copy()
        try:
            px, py = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        except (TypeError, ValueError, IndexError):
            return {"found": False, "error": "invalid pixel"}, b""
        self._draw_crosshair(overlay, px, py)
        if bbox is not None:
            try:
                x0, y0, x1, y1 = (int(round(v)) for v in bbox)
            except (TypeError, ValueError):
                x0, y0, x1, y1 = None, None, None, None
            if x0 is not None:
                h, w = overlay.shape[:2]
                margin = max(8, int(0.1 * max(x1 - x0, y1 - y0)))
                x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
                x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
                crop = overlay[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = overlay
            else:
                crop = overlay
        else:
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
            rgb = self._decode_rgb(header.get("shape"), payload)
            return {"ok": True, **self.feed_frame(rgb)}, b""
        if cmd == "reset_map":
            return {"ok": True, **self.reset_map()}, b""
        if cmd == "flush_map":
            return {"ok": True, **self.flush_map()}, b""
        if cmd == "get_state":
            return {"ok": True, **self.get_state()}, b""
        if cmd == "get_captioned_frame_ids":
            return {"ok": True, **self.get_captioned_frame_ids()}, b""
        if cmd == "get_captions":
            return {"ok": True, **self.get_captions(
                header.get("frame_ids", []))}, b""
        if cmd == "set_episode":
            self._set_episode(header.get("episode_id"))
            return {"ok": True}, b""
        if cmd == "get_latest_pose":
            return {"ok": True, **self.get_latest_pose()}, b""
        if cmd == "get_all_poses":
            return {"ok": True, **self.get_all_poses()}, b""
        if cmd == "get_frame_pose":
            return {"ok": True, **self.get_frame_pose(
                header.get("frame_id"))}, b""
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
                header["text"], int(header.get("top_k", 2)))}, b""
        if cmd == "ground_object_pixels":
            return {"ok": True, **self.ground_object_pixels(
                header["text"], int(header.get("top_k", 2)))}, b""
        if cmd == "point_frame":
            return {"ok": True, **self.point_frame(
                int(header["frame_id"]), header["text"])}, b""
        if cmd == "point_pixels":
            return {"ok": True, **self.point_pixels(
                int(header["frame_id"]), header["text"])}, b""
        if cmd == "instantiate_pixels":
            return {"ok": True, **self.instantiate_pixels(
                int(header["frame_id"]), header.get("pixels", []),
                bool(header.get("normalized", True)))}, b""
        if cmd == "prepare_pixels":
            return {"ok": True, **self.prepare_pixels(
                int(header["frame_id"]), header.get("pixels", []),
                bool(header.get("normalized", True)))}, b""
        if cmd == "som_segment":
            resp, overlay = self.som_segment(
                int(header["frame_id"]), header.get("max_masks"))
            return {"ok": True, **resp}, overlay
        if cmd == "som_pick":
            return {"ok": True, **self.som_pick(
                int(header["frame_id"]), header.get("mask_ids", []))}, b""
        if cmd == "resolve_candidate":
            return {"ok": True, **self.resolve_candidate(
                header["candidate_id"])}, b""
        if cmd == "resolve_candidates":
            return {"ok": True, **self.resolve_candidates(
                header.get("candidate_ids", []))}, b""
        if cmd == "candidate_evidence":
            resp, evidence = self.candidate_evidence(
                header["candidate_id"], bool(header.get("wide_only")))
            return {"ok": True, **resp}, evidence
        if cmd == "evidence_for_point":
            resp, evidence = self.evidence_for_point(
                int(header["frame_id"]), header.get("pixel", []),
                header.get("bbox"))
            return {"ok": True, **resp}, evidence
        if cmd == "ground_frame":
            rgb = self._decode_rgb(header.get("shape"), payload)
            return {"ok": True, **self.ground_frame(
                rgb, header["text"])}, b""
        if cmd == "get_frame_points":
            resp, payload_out = self.get_frame_points(
                int(header.get("stride", 6)))
            return {"ok": True, **resp}, payload_out
        if cmd == "shutdown":
            return {"ok": True, "shutdown": True}, b""
        return {"ok": False, "error": f"unknown cmd: {cmd}"}, b""

    @staticmethod
    def _decode_rgb(shape, payload):
        """校验并解码协议中的 HxWx3 uint8 RGB，避免畸形请求终止服务。"""
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("RGB shape 必须是 [H, W, 3]")
        try:
            h, w, channels = (int(v) for v in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("RGB shape 必须包含整数") from exc
        if h <= 0 or w <= 0 or channels != 3:
            raise ValueError("RGB shape 必须是正尺寸 [H, W, 3]")
        expected = h * w * channels
        if len(payload) != expected:
            raise ValueError(
                f"RGB payload 长度不匹配: {len(payload)} != {expected}")
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, channels)

    def serve(self):
        from mapping.protocol import recv_msg, send_msg

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.args.host, self.args.port))
        sock.listen(1)
        print(f"[server] 监听 {self.args.host}:{self.args.port}")
        try:
            while not self._closed:
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
                            try:
                                resp, resp_payload = self.handle_message(
                                    header, payload)
                            except Exception as exc:  # 单个请求失败不拖垮服务
                                print(f"[server] 请求 {header.get('cmd')!r} 失败: "
                                      f"{exc}", flush=True)
                                traceback.print_exc()
                                resp = {"ok": False, "error": str(exc)}
                                resp_payload = b""
                            if cacheable and request_id:
                                self._response_cache[request_id] = (
                                    resp, resp_payload)
                                while len(self._response_cache) > 16:
                                    self._response_cache.pop(
                                        next(iter(self._response_cache)))
                        send_msg(conn, resp, resp_payload)
                        if resp.get("shutdown"):
                            print("[server] 收到 shutdown，退出")
                            return
                except (ConnectionError, OSError):
                    print("[server] 客户端断开")
                except Exception as exc:
                    # 协议帧本身损坏时无法可靠回复，只关闭当前连接。
                    print(f"[server] 客户端协议错误: {exc}", flush=True)
                    traceback.print_exc()
                finally:
                    conn.close()
        finally:
            sock.close()
            self._server_socket = None

    def close(self):
        """幂等释放后台 worker、诊断文件与监听 socket。"""
        if self._closed:
            return
        self._closed = True
        if self.caption_worker is not None:
            self.caption_worker.close()
        if self.caption_store is not None:
            self.caption_store.save()
        if self.vllm is not None:
            self.vllm.close()
        with self.diag_lock:
            for fp_name in ("_diag_fp", "_frame_manifest_fp"):
                fp = getattr(self, fp_name, None)
                if fp is not None:
                    try:
                        fp.close()
                    except OSError:
                        pass
                    setattr(self, fp_name, None)
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass


class _NullRetrieval:
    """SALAD 缺失时的空检索器：不产生任何回环。"""

    def get_all_submap_embeddings(self, submap):
        return []

    def find_loop_closures(self, *args, **kwargs):
        return []


def main():
    args = _parse_args()
    server = MappingServer(args)
    try:
        server.serve()
    finally:
        server.close()


if __name__ == "__main__":
    main()
