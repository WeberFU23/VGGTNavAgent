"""语义记忆：关键帧 caption 生成与文文检索（server 端，vggtslam 环境）。

架构：
1. CaptionWorker：挂在子图处理完成后的挂点上，异步为每个关键帧生成
   查询无关的详细 caption（可见物体清单及颜色/材质等属性）。请求经
   VLLMGateway 发给本地多模态 VLM，优先级最低；GPU 忙（子图处理/
   pointing 在途）时让路。
2. CaptionStore：{frame_id, 位姿, caption, embedding(BGE-M3)} 记忆库，
   落盘持久化，支持按 episode 清空。检索使用 BGE-M3 文本向量和余弦
   相似度 top-K，以支持较长 caption 和完整任务描述。caption 模型由
   NAV_CAPTION_API_MODEL（独立 API）或回落时 NAV_CAPTION_MODEL_PATH
   （本地 vLLM）配置，见 server._init_semantic_memory。

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

CAPTION_PROMPT = """Describe this indoor RGB image for later text-based retrieval.
First line: "Scene context:" followed by the probable room/area type and salient
fixed fixtures (for example bathroom with shower area, kitchen with sink), or
"unknown" when the image does not support a room label. Mark uncertain labels as
probable rather than asserting them.
Skip generic room surfaces (walls, floor, ceiling) from the object list.
Second line: "Objects:" followed by the object category names only, comma-separated
(e.g. "chair, table, clock"); list each category once even if several are visible.
Then write one natural-language sentence per object instance, numbered per
category (e.g. "chair 1: ...", "chair 2: ...").

Each sentence must describe ONLY the object's intrinsic appearance and
distinguishing details or contents it visibly holds.
Do NOT mention:
- where the object appears in the image (left/right/center/foreground/...),
- where it is relative to other objects or the room (next to/above/near/mounted
  on/in front of/beside/...),
- or name any other object when describing this one.
To tell same-category instances apart, use intrinsic differences only;
if two instances are truly indistinguishable, still write one sentence each.
At most 20 sentences. Query-independent; prefer completeness over brevity. No
extra commentary beyond the required Scene context and Objects sections."""


def _safe_episode_component(value):
    text = str(value or "unknown").strip()
    return "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in text
    )[:120] or "unknown"


class CaptionStore:
    """{frame_id, pose, caption, embedding} 记忆库，numpy 实现。"""

    def __init__(self, persist_dir=None):
        self.persist_dir = str(persist_dir) if persist_dir else None
        self.episode_id = None
        self.records = []            # [{frame_id, pose, caption}]
        self._embeddings = []        # list of (D,) float32
        self._embedding_dim = None
        self._lock = threading.RLock()
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    def __len__(self):
        with self._lock:
            return len(self.records)

    @property
    def frame_ids(self):
        with self._lock:
            return [r["frame_id"] for r in self.records]

    def add(self, frame_id, pose, caption, embedding):
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            raise ValueError("caption embedding 不能为空")
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        pose_arr = None
        if pose is not None:
            pose_arr = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        with self._lock:
            if self._embedding_dim is not None and emb.size != self._embedding_dim:
                raise ValueError(
                    f"caption embedding 维度不一致: {emb.size} != "
                    f"{self._embedding_dim}")
            self._embedding_dim = int(emb.size)
            self.records.append({
                "frame_id": int(frame_id),
                "pose": pose_arr,
                "caption": str(caption),
            })
            self._embeddings.append(emb)

    def has(self, frame_id):
        frame_id = int(frame_id)
        with self._lock:
            return any(r["frame_id"] == frame_id for r in self.records)

    def get_captions(self, frame_ids):
        """按 frame_id 批量取回 caption；未入库的帧跳过。
        返回 [{frame_id, caption}]，顺序与请求一致。"""
        wanted = [int(fid) for fid in (frame_ids or [])]
        with self._lock:
            by_id = {r["frame_id"]: r["caption"] for r in self.records}
        return [{"frame_id": fid, "caption": by_id[fid]}
                for fid in wanted if fid in by_id]

    def retrieve(self, query_embedding, k=10):
        """余弦 top-K，返回 [{frame_id, caption, score, pose}]（score 降序）。"""
        with self._lock:
            if not self.records:
                return []
            records = list(self.records)
            embeddings = list(self._embeddings)
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        expected_dim = embeddings[0].size
        if q.size != expected_dim:
            raise ValueError(
                f"query embedding 维度不一致: {q.size} != {expected_dim}")
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        mat = np.stack(embeddings)                # (N, D)，已归一
        sims = mat @ q
        k = max(1, min(int(k), len(records)))
        out = []
        for idx in np.argsort(-sims)[:k]:
            rec = records[int(idx)]
            out.append({
                "frame_id": rec["frame_id"],
                "caption": rec["caption"],
                "score": float(sims[int(idx)]),
                "pose": (rec["pose"].tolist()
                         if rec["pose"] is not None else None),
            })
        return out

    # 持久化：每 episode 一个目录（captions.jsonl + embeddings.npy）
    # ------------------------------------------------------------------
    def set_episode(self, episode_id):
        """切换 episode：先落盘当前记忆，再清空（跨 episode 不共享）。"""
        episode_id = str(episode_id or "unknown")
        with self._lock:
            if episode_id == self.episode_id:
                return
            self.save()
            self.clear()
            self.episode_id = episode_id

    def clear(self):
        with self._lock:
            self.records = []
            self._embeddings = []
            self._embedding_dim = None

    def _episode_dir(self, episode_id=None):
        ep = _safe_episode_component(episode_id or self.episode_id)
        return os.path.join(self.persist_dir, ep)

    def save(self, episode_id=None):
        with self._lock:
            if not self.persist_dir or not self.records:
                return
            records = list(self.records)
            embeddings = list(self._embeddings)
        out_dir = self._episode_dir(episode_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "captions.jsonl"),
                  "w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps({
                    "frame_id": rec["frame_id"],
                    "caption": rec["caption"],
                    "pose": (rec["pose"].tolist()
                             if rec["pose"] is not None else None),
                }, ensure_ascii=False) + "\n")
        np.save(os.path.join(out_dir, "embeddings.npy"),
                np.stack(embeddings))

    def load(self, episode_id):
        """从磁盘恢复一个 episode 的记忆（离线检索验证用）。"""
        with self._lock:
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
                items = [json.loads(line) for line in fp if line.strip()]
            if len(items) != len(embs):
                raise ValueError(
                    "caption 持久化文件不一致: "
                    f"{len(items)} 条记录但有 {len(embs)} 个 embedding")
            for item, emb in zip(items, embs):
                self.add(item["frame_id"], item.get("pose"),
                         item["caption"], emb)
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
                 busy_fn=None, max_tokens=512, prompt=CAPTION_PROMPT,
                 result_fn=None, workers=1):
        self.gateway = gateway
        self.embedder = embedder
        self.store = store
        self.model = str(model or "").strip()
        self.busy_fn = busy_fn
        self.max_tokens = int(max_tokens)
        self.prompt = prompt
        self.result_fn = result_fn
        self._queue = queue.Queue()
        self._closed = False
        self._state_lock = threading.RLock()
        self._embed_lock = threading.Lock()  # embedder 非线程安全，串行编码
        self._generation = 0
        self.errors = 0
        # 语义记忆进度：已入队但尚未生成 caption 的关键帧（含在途处理）。
        # agent 端据此判断检索是否会漏掉最新关键帧。
        self._pending = set()       # (generation, frame_id)
        self.last_completed_frame_id = None
        # caption 走远端 API 时不占 GPU，workers>1 并发消化积压；本地
        # vLLM 时保持 1（网关层串行保护显存）。
        self._threads = []
        for i in range(max(1, int(workers))):
            t = threading.Thread(
                target=self._consume, name=f"caption-worker-{i}",
                daemon=True)
            t.start()
            self._threads.append(t)

    def enqueue(self, frame_id, pil_img, pose=None):
        if self._closed or not self.model:
            return
        frame_id = int(frame_id)
        with self._state_lock:
            key = (self._generation, frame_id)
            if self.store.has(frame_id) or key in self._pending:
                return
            self._pending.add(key)
            self._queue.put((self._generation, frame_id, pil_img, pose))

    def clear(self):
        """切换 generation，使排队及在途的旧 episode 任务全部失效。"""
        with self._state_lock:
            self._generation += 1
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._pending.clear()
            self.last_completed_frame_id = None

    def pending(self):
        """已入队但尚未完成 caption 的关键帧数。"""
        with self._state_lock:
            return sum(1 for gen, _ in self._pending
                       if gen == self._generation)

    def pending_frame_ids(self):
        with self._state_lock:
            return sorted(frame_id for gen, frame_id in self._pending
                          if gen == self._generation)

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            # 使可能仍在 HTTP 推理中的任务失效，关闭后不得再写 store。
            self._generation += 1
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._pending.clear()
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=2.0)

    def _consume(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, frame_id, pil_img, pose = item
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
                    cache_key=f"generation_{generation}_frame_{frame_id}",
                    priority=Priority.CAPTION, max_tokens=self.max_tokens)
                caption = str(caption).strip()
                if not caption:
                    raise VLLMError("caption 为空")
                with self._embed_lock:
                    emb = self.embedder.encode([caption])[0]                # clear() 与此临界区互斥；旧 generation 不能写入新 episode。
                with self._state_lock:
                    if generation != self._generation:
                        continue
                    self.store.add(frame_id, pose, caption, emb)
                    if self.last_completed_frame_id is None or \
                            frame_id > self.last_completed_frame_id:
                        self.last_completed_frame_id = frame_id
                self._emit_result({
                    "frame_id": frame_id, "caption": caption,
                    "model": self.model, "status": "completed",
                })
            except Exception as exc:  # noqa: BLE001 - 单帧失败不拖垮线程
                self.errors += 1
                print(f"[CaptionWorker] frame {frame_id} caption 失败: {exc}",
                      flush=True)
                self._emit_result({
                    "frame_id": frame_id, "caption": None,
                    "model": self.model, "status": "failed",
                    "error": str(exc),
                })
            finally:
                with self._state_lock:
                    self._pending.discard((generation, frame_id))

    def _emit_result(self, record):
        if self.result_fn is None:
            return
        try:
            self.result_fn(dict(record))
        except Exception:
            pass
