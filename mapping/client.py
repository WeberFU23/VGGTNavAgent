"""VGGT-SLAM 建图客户端。在 habitat (python 3.9) 环境中运行，只依赖 numpy。

用法::

    client = MappingClient(port=5555)
    client.reset_map()                    # 每个 episode 开始
    info = client.feed_frame(rgb)         # 每个动作步喂当前 RGB
    pose = client.get_latest_pose()       # 最新关键帧位姿 (cam2world, 4x4) 或 None
    poses, frame_ids = client.get_all_poses()
    points, colors = client.get_map_points()

注意：返回位姿是单目相对尺度，世界系锚定第一个子图；
米制尺度需要通过已知动作步长（前进 0.25m）另行标定。
"""

import socket
import time
import uuid

import numpy as np

from mapping.protocol import recv_msg, send_msg


class MappingClient:
    def __init__(self, host="127.0.0.1", port=5555, timeout=120.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None
        self._session_id = uuid.uuid4().hex
        self._request_seq = 0
        self.last_frame_snapshot_revision = None

    def _connect(self):
        if self._sock is None:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout)

    def _request(self, header, payload=b"", retries=1):
        header = dict(header)
        self._request_seq += 1
        header["request_id"] = f"{self._session_id}:{self._request_seq}"
        for attempt in range(retries + 1):
            try:
                self._connect()
                send_msg(self._sock, header, payload)
                resp, resp_payload = recv_msg(self._sock)
                if not resp.get("ok"):
                    raise RuntimeError(f"mapping server error: {resp}")
                return resp, resp_payload
            except (ConnectionError, OSError):
                self.close()
                if attempt >= retries:
                    raise
        raise RuntimeError("unreachable")

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # ------------------------------------------------------------------
    def ping(self):
        self._request({"cmd": "ping"})
        return True

    def reset_map(self):
        resp, _ = self._request({"cmd": "reset_map"})
        self.last_frame_snapshot_revision = None
        return resp

    def set_episode(self, episode_id):
        """告知服务端当前 episode，用于语义查询诊断日志归档。"""
        resp, _ = self._request(
            {"cmd": "set_episode", "episode_id": str(episode_id)})
        return resp

    def flush_map(self):
        """提交不足一个完整子图的尾部关键帧，并等待处理完成。"""
        resp, _ = self._request({"cmd": "flush_map"})
        return resp

    def feed_frame(self, rgb):
        """喂入一帧 RGB (H, W, 3) uint8，返回关键帧筛选等信息。"""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._validate_rgb(rgb)
        resp, _ = self._request(
            {"cmd": "feed", "shape": list(rgb.shape)}, rgb.tobytes())
        return resp

    def get_state(self):
        resp, _ = self._request({"cmd": "get_state"})
        return resp

    def wait_idle(self, timeout=60.0, poll=0.2):
        """等待服务端完成在途子图处理。返回 True 表示已空闲。

        用于 agent pacing：Habitat 离散步执行速度远快于 SLAM 吞吐，
        忙时短暂等待可避免关键帧缓冲被裁剪丢弃。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.get_state().get("busy"):
                return True
            time.sleep(poll)
        return False

    def wait_captions(self, timeout=30.0, poll=0.5):
        """等待 caption worker 消化完已入队关键帧（语义记忆追上 SLAM）。
        返回 True 表示队列清空；超时返回 False（调用方继续，检索可能漏新帧）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if int(self.get_state().get("caption_pending") or 0) <= 0:
                return True
            time.sleep(poll)
        return False

    def get_latest_pose(self):
        """返回最新关键帧的 cam2world 4x4 位姿，尚无可位姿时返回 None。"""
        resp, _ = self._request({"cmd": "get_latest_pose"})
        if not resp.get("has_pose"):
            return None
        return np.asarray(resp["pose"], dtype=np.float32).reshape(4, 4)

    def get_all_poses(self):
        """返回 (poses (N,4,4) cam2world, frame_ids list)；无数据返回 (None, [])。"""
        resp, _ = self._request({"cmd": "get_all_poses"})
        if not resp.get("has_pose"):
            return None, []
        poses = np.asarray(resp["poses"], dtype=np.float32).reshape(-1, 4, 4)
        return poses, list(resp["frame_ids"])

    def get_map_points(self, max_points=200000):
        """返回 (points (N,3) float32, colors (N,3) uint8)。"""
        resp, payload = self._request({"cmd": "get_map", "max_points": max_points})
        n = int(resp.get("num_points", 0))
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        expected = n * 15
        if len(payload) != expected:
            raise RuntimeError(
                f"mapping payload 长度不匹配: {len(payload)} != {expected}")
        points = np.frombuffer(payload[: n * 12], dtype=np.float32).reshape(n, 3)
        colors = np.frombuffer(payload[n * 12:], dtype=np.uint8).reshape(n, 3)
        return points.copy(), colors.copy()

    def get_frame_points(self, stride=6):
        """返回同一服务端快照中的逐帧世界点、RGB 与位姿。

        ``colors`` 对旧服务端为 ``None``，使客户端可以在滚动部署期间
        继续建图；新版服务端则保证颜色与 points 逐点对应。
        """
        resp, payload = self._request(
            {"cmd": "get_frame_points", "stride": stride})
        revision = resp.get("snapshot_revision")
        self.last_frame_snapshot_revision = (
            dict(revision) if isinstance(revision, dict) else None)
        frames = []
        offset = 0
        for meta in resp.get("frames", []):
            h, w = int(meta["h"]), int(meta["w"])
            if h <= 0 or w <= 0:
                raise RuntimeError(f"mapping frame 尺寸无效: {h}x{w}")
            n = h * w
            end = offset + n * 12
            if end > len(payload):
                raise RuntimeError("mapping frame payload 截断")
            pts = np.frombuffer(payload[offset:end],
                                dtype=np.float32).reshape(n, 3)
            offset = end
            colors = None
            if meta.get("has_colors"):
                color_end = offset + n * 3
                if color_end > len(payload):
                    raise RuntimeError("mapping frame RGB payload 截断")
                colors = np.frombuffer(
                    payload[offset:color_end], dtype=np.uint8).reshape(n, 3)
                offset = color_end
            rows = np.repeat(
                np.arange(h, dtype=np.int32) * meta["stride"], w)
            frames.append({
                "frame_id": meta["frame_id"],
                "pose": np.asarray(meta["pose"], dtype=np.float32)
                .reshape(4, 4),
                "points": pts.copy(),
                "colors": None if colors is None else colors.copy(),
                "rows": rows,
            })
        if offset != len(payload):
            raise RuntimeError(
                f"mapping frame payload 有多余字节: {len(payload) - offset}")
        return frames

    def get_captioned_frame_ids(self):
        """返回 (semantic_enabled, completed_frame_ids)。"""
        resp, _ = self._request({"cmd": "get_captioned_frame_ids"})
        return bool(resp.get("enabled")), [
            int(fid) for fid in resp.get("frame_ids", [])]

    def get_intrinsics(self):
        """返回 VGGT 预测的各子图首帧内参列表（预处理图像坐标系）。"""
        resp, _ = self._request({"cmd": "get_intrinsics"})
        return resp.get("intrinsics", [])

    def query_text(self, text, top_k=5):
        """按 caption 检索 top-K 关键帧。
        返回 [{frame_id, caption, score, pose, ...}]。"""
        resp, _ = self._request({"cmd": "query_text", "text": text,
                                 "top_k": top_k})
        return resp.get("results", [])

    def retrieve_captions(self, text, top_k=10):
        """caption 语义记忆检索。返回 [{frame_id, caption, score, pose}]。"""
        resp, _ = self._request({"cmd": "retrieve_captions", "text": text,
                                 "top_k": top_k})
        return resp.get("results", [])

    def get_frame_image(self, frame_id):
        """返回指定关键帧的 JPEG（决策层 look_instance 工具）。
        返回 (meta, payload)；失败时 meta["found"]=False。"""
        resp, payload = self._request(
            {"cmd": "get_frame_image", "frame_id": int(frame_id)})
        return resp, payload

    def ground_object(self, text, top_k=3):
        """caption 召回和 pointing 定位（不在探索阶段确认类别）。
        返回 [{found, point, point_score, frame_id, ...}]。"""
        resp, _ = self._request({"cmd": "ground_object", "text": text,
                                 "top_k": top_k})
        return resp.get("results", [])

    def resolve_candidate(self, candidate_id):
        """在最新图优化坐标系中重新计算候选的 3D 点。"""
        resp, _ = self._request(
            {"cmd": "resolve_candidate", "candidate_id": candidate_id})
        return resp

    def resolve_candidates(self, candidate_ids):
        """批量返回候选在最新图优化坐标系中的 3D 点。"""
        resp, _ = self._request({
            "cmd": "resolve_candidates",
            "candidate_ids": list(candidate_ids or []),
        })
        return resp.get("candidates", {})

    def get_candidate_evidence(self, candidate_id):
        """返回候选的紧凑 mask-overlay JPEG，供 VLM 复核。"""
        resp, payload = self._request(
            {"cmd": "candidate_evidence", "candidate_id": candidate_id})
        return resp, payload

    def ground_frame(self, rgb, text):
        """对当前实时帧做 VQA + pointing；仅保留给诊断/兼容调用。"""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._validate_rgb(rgb)
        resp, _ = self._request(
            {"cmd": "ground_frame", "text": text,
             "shape": list(rgb.shape)}, rgb.tobytes())
        return resp

    @staticmethod
    def _validate_rgb(rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] <= 0 \
                or rgb.shape[1] <= 0:
            raise ValueError(
                f"RGB 必须是非空 HxWx3 数组，实际为 {rgb.shape}")

    def shutdown_server(self):
        resp, _ = self._request({"cmd": "shutdown"})
        self.close()
        return resp
