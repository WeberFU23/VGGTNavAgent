"""语义记忆：关键帧 caption 生成与文文检索（server 端，vggtslam 环境）。

架构（替代 CLIP 图文检索）：
1. CaptionWorker：挂在子图处理完成后的挂点上，异步为每个关键帧生成
   查询无关的详细 caption（场景类型/房间 + 可见物体清单含颜色材质 +
   物体间空间关系）。经 VLLMGateway 调 Qwen2.5-VL-3B，优先级最低，
   GPU 忙（子图处理/pointing 在途）时让路。
2. CaptionStore：{frame_id, 位姿, caption, embedding(BGE-M3)} 记忆库，
   落盘持久化，支持按 episode 清空。检索 = 余弦相似度 top-K（文文匹配，
   不用 CLIP text encoder——77 token 截断对长 caption 检索质量差）。

CaptionStore 只用 numpy，可脱离 GPU/网络单测；Embedder/Gateway 均可
mock 替换（权重缺失时 BGEM3Embedder 构造抛清晰错误，由上层降级）。
"""

import json
import os
import queue
import threading
import time

import numpy as np

from mapping.vllm_client import Priority, VLLMError

CAPTION_PROMPT = """Describe this indoor RGB image in detail for later text-based
retrieval. This description is query-independent, so prefer completeness over
brevity. Cover:
1. Scene/room type and overall layout.
2. A list of visible objects, each with fine-grained attributes (color,
   material, shape, size relative to surroundings, texture).
3. Spatial relations between objects (on/under/next to/behind/inside, left
   and right from the camera's perspective).
Output plain prose, 3-6 sentences, no markdown, no bullet markers."""


class CaptionStore:
    """{frame_id, pose, caption, embedding} 记忆库，numpy 实现。"""

    def __init__(self, persist_dir=None):
        self.persist_dir = str(persist_dir) if persist_dir else None
        self.episode_id = None
        self.records = []            # [{frame_id, pose, caption}]
        self._embeddings = []        # list of (D,) float32
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.records)

    @property
    def frame_ids(self):
        return [r["frame_id"] for r in self.records]

    def add(self, frame_id, pose, caption, embedding):
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        pose_arr = None
        if pose is not None:
            pose_arr = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        self.records.append({
            "frame_id": int(frame_id),
            "pose": pose_arr,
            "caption": str(caption),
        })
        self._embeddings.append(emb)

    def has(self, frame_id):
        return int(frame_id) in set(self.frame_ids)

    # ------------------------------------------------------------------
    def retrieve(self, query_embedding, k=10):
        """余弦 top-K，返回 [{frame_id, caption, score, pose}]（score 降序）。"""
        if not self.records:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        mat = np.stack(self._embeddings)          # (N, D)，已归一
        sims = mat @ q
        k = max(1, min(int(k), len(self.records)))
        out = []
        for idx in np.argsort(-sims)[:k]:
            rec = self.records[int(idx)]
            out.append({
                "frame_id": rec["frame_id"],
                "caption": rec["caption"],
                "score": float(sims[int(idx)]),
                "pose": (rec["pose"].tolist()
                         if rec["pose"] is not None else None),
            })
        return out

    # ------------------------------------------------------------------
    # 持久化：每 episode 一个目录（captions.jsonl + embeddings.npy）
    # ------------------------------------------------------------------
    def set_episode(self, episode_id):
        """切换 episode：先落盘当前记忆，再清空（跨 episode 不共享）。"""
        episode_id = str(episode_id or "unknown")
        if episode_id == self.episode_id:
            return
        self.save()
        self.clear()
        self.episode_id = episode_id

    def clear(self):
        self.records = []
        self._embeddings = []

    def _episode_dir(self, episode_id=None):
        ep = episode_id or self.episode_id or "unknown"
        return os.path.join(self.persist_dir, ep)

    def save(self, episode_id=None):
        if not self.persist_dir or not self.records:
            return
        out_dir = self._episode_dir(episode_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "captions.jsonl"),
                  "w", encoding="utf-8") as fp:
            for rec in self.records:
                fp.write(json.dumps({
                    "frame_id": rec["frame_id"],
                    "caption": rec["caption"],
                    "pose": (rec["pose"].tolist()
                             if rec["pose"] is not None else None),
                }, ensure_ascii=False) + "\n")
        np.save(os.path.join(out_dir, "embeddings.npy"),
                np.stack(self._embeddings))

    def load(self, episode_id):
        """从磁盘恢复一个 episode 的记忆（离线检索验证用）。"""
        self.clear()
        if not self.persist_dir:
            return 0
        in_dir = self._episode_dir(episode_id)
        jsonl = os.path.join(in_dir, "captions.jsonl")
        npy = os.path.join(in_dir, "embeddings.npy")
        if not (os.path.exists(jsonl) and os.path.exists(npy)):
            return 0
        embs = np.load(npy)
        with open(jsonl, encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                item = json.loads(line)
                self.add(item["frame_id"], item.get("pose"),
                         item["caption"], embs[i])
        self.episode_id = str(episode_id)
        return len(self.records)


class BGEM3Embedder:
    """BGE-M3 文文检索向量（懒加载；权重缺失时构造即报清晰错误）。

    模型路径走 NAV_EMBED_MODEL_PATH 环境变量（如
    /root/autodl-tmp/models/bge-m3），不写死绝对路径。
    """

    def __init__(self, model_path=None, device="cuda"):
        self.model_path = str(model_path or "").strip()
        self.device = device
        self._model = None
        if not self.model_path:
            raise RuntimeError(
                "BGE-M3 模型路径未配置：请设置 NAV_EMBED_MODEL_PATH"
                "（远端建议 /root/autodl-tmp/models/bge-m3）")
        if not os.path.isdir(self.model_path):
            raise RuntimeError(
                f"BGE-M3 权重目录不存在: {self.model_path}"
                "（在 AutoDL 上先 modelscope download 或 HF_ENDPOINT 镜像下载）")

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_path, device=self.device)

    def encode(self, texts):
        """texts: str 列表 -> (N, D) float32，行已 L2 归一化。"""
        self._load()
        vecs = self._model.encode(
            [str(t) for t in texts], normalize_embeddings=True,
            convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class CaptionWorker:
    """异步 caption 生成线程（最低优先级，GPU 忙时让路）。

    busy_fn() -> bool 由 server 注入（子图处理中 / 语义查询在途），
    为 True 时暂停消费，避免与 VGGT/pointing 抢显存。
    """

    def __init__(self, gateway, embedder, store, model,
                 busy_fn=None, max_tokens=512, prompt=CAPTION_PROMPT):
        self.gateway = gateway
        self.embedder = embedder
        self.store = store
        self.model = str(model or "").strip()
        self.busy_fn = busy_fn
        self.max_tokens = int(max_tokens)
        self.prompt = prompt
        self._queue = queue.Queue()
        self._closed = False
        self.errors = 0
        self._thread = threading.Thread(
            target=self._consume, name="caption-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def enqueue(self, frame_id, pil_img, pose=None):
        if self._closed or not self.model:
            return
        if self.store.has(frame_id):
            return
        self._queue.put((int(frame_id), pil_img, pose))

    def clear(self):
        """episode 切换时丢弃未处理的 caption 任务。"""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def pending(self):
        return self._queue.qsize()

    def close(self):
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _consume(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame_id, pil_img, pose = item
            # GPU 忙时让路：稍后再试（不丢任务）
            if self.busy_fn is not None:
                while not self._closed:
                    try:
                        if not self.busy_fn():
                            break
                    except Exception:
                        break
                    time.sleep(0.5)
            if self._closed:
                return
            try:
                caption = self.gateway.chat(
                    self.model, self.prompt, [pil_img], kind="caption",
                    cache_key=f"frame_{frame_id}",
                    priority=Priority.CAPTION, max_tokens=self.max_tokens)
                caption = str(caption).strip()
                if not caption:
                    raise VLLMError("caption 为空")
                emb = self.embedder.encode([caption])[0]
                self.store.add(frame_id, pose, caption, emb)
            except Exception as exc:  # noqa: BLE001 - 单帧失败不拖垮线程
                self.errors += 1
                print(f"[CaptionWorker] frame {frame_id} caption 失败: {exc}",
                      flush=True)
