"""vLLM OpenAI 兼容服务客户端（server 端，vggtslam 环境）。

caption 和 pointing 统一走本地 vLLM 服务（OpenAI chat completions API），
本模块提供：

1. 优先级队列：高优先级请求 > pointing > caption。所有请求经单 worker
   线程串行发出（vLLM 服务端自身做连续批处理，这里只保证提交顺序），
   caption 等低优先级请求在 GPU 繁忙期自然排后。
2. 同帧结果缓存：cache_key（通常是 frame_id + 请求类型 + prompt 摘要）
   命中时直接返回，不重复推理；切换 episode 时由 server 清空缓存。
3. 明确报错与重试：连接失败/5xx/超时按指数退避重试 max_retries 次，
   之后抛 VLLMError；权重未加载（404 model not found）立即报错不重试。

只依赖 requests + 标准库（不 import torch/cv2），py3.9 与 py3.11 均可
import，单测通过 post_fn 注入 mock，不需要真实 vLLM 服务。
"""

import base64
import hashlib
import io
import json
import queue
import re
import threading
import time
from enum import IntEnum
from itertools import count

_DEFAULT_URL = "http://127.0.0.1:8000/v1"


class Priority(IntEnum):
    DECISION = 0
    POINTING = 1
    CAPTION = 2


class VLLMError(RuntimeError):
    """vLLM 调用最终失败（重试耗尽或不可重试错误）。"""


class _Pending:
    __slots__ = ("event", "value", "error")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None


class AsyncResult:
    """chat_async 的返回句柄。"""

    def __init__(self, pending):
        self._pending = pending

    def result(self, timeout=None):
        if not self._pending.event.wait(timeout):
            raise VLLMError("async result wait timeout")
        if self._pending.error is not None:
            raise self._pending.error
        return self._pending.value

    def done(self):
        return self._pending.event.is_set()


class VLLMGateway:
    """vLLM 服务网关：优先级队列 + 缓存 + 重试。

    post_fn 可注入（签名同 requests.post），单测不碰网络。
    start_worker=False 时不起消费线程，便于单测手动检查队列顺序。
    """

    def __init__(self, url=_DEFAULT_URL, api_key="EMPTY", timeout=120.0,
                 max_retries=3, backoff_base=0.5, post_fn=None,
                 start_worker=True, trace_fn=None):
        self.url = str(url or _DEFAULT_URL).rstrip("/")
        self.api_key = str(api_key or "EMPTY")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self._post_fn = post_fn
        self._trace_fn = trace_fn
        self._queue = queue.PriorityQueue()
        self._seq = count()
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._closed = False
        self._worker = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._consume, name="vllm-gateway", daemon=True)
            self._worker.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        # 唤醒 worker 退出
        self._queue.put((Priority.CAPTION, next(self._seq), None))
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def queue_size(self):
        return self._queue.qsize()

    def clear_cache(self):
        """清空图像推理缓存；切换 episode 时必须调用。"""
        with self._cache_lock:
            self._cache.clear()

    def chat(self, model, prompt, images=None, *, kind="generic",
             cache_key=None, priority=Priority.POINTING, max_tokens=1024):
        """同步调用，返回模型文本输出。"""
        return self.chat_async(
            model, prompt, images, kind=kind, cache_key=cache_key,
            priority=priority, max_tokens=max_tokens).result(
            timeout=self.timeout * (self.max_retries + 1) + 10)

    def chat_json(self, model, prompt, images=None, **kwargs):
        """同 chat，但把输出解析为 JSON 对象（去 markdown 围栏）。"""
        text = self.chat(model, prompt, images, **kwargs)
        return self.parse_json(text)

    def chat_async(self, model, prompt, images=None, *, kind="generic",
                   cache_key=None, priority=Priority.POINTING,
                   max_tokens=1024):
        """异步入队，返回 AsyncResult。cache_key 命中时不入队。"""
        if self._closed:
            raise VLLMError("vLLM gateway 已关闭")
        if not model:
            raise VLLMError("model 名为空（检查 NAV_*_MODEL_PATH 配置）")
        key = self._make_cache_key(cache_key, model, kind, prompt)
        if key is not None:
            with self._cache_lock:
                if key in self._cache:
                    pending = _Pending()
                    pending.value = self._cache[key]
                    pending.event.set()
                    self._trace({
                        "kind": str(kind), "model": str(model),
                        "cache_key": cache_key, "cache_hit": True,
                        "prompt": str(prompt),
                        "image_count": len(images or []),
                        "raw_output": pending.value, "ok": True,
                    })
                    return AsyncResult(pending)
        pending = _Pending()
        payload = self._build_payload(model, prompt, images, max_tokens)
        meta = {
            "kind": str(kind), "model": str(model),
            "cache_key": cache_key, "cache_hit": False,
            "prompt": str(prompt), "image_count": len(images or []),
        }
        self._queue.put((int(priority), next(self._seq),
                         (payload, key, pending, meta)))
        return AsyncResult(pending)

    def submit_nowait_for_test(self, priority, item):
        """测试用：直接入队原始项。"""
        self._queue.put((int(priority), next(self._seq), item))

    def _consume(self):
        while True:
            _prio, _seq, item = self._queue.get()
            if item is None:
                return
            if len(item) == 4:
                payload, key, pending, meta = item
            else:  # backward-compatible raw test item
                payload, key, pending = item
                meta = {"kind": "unknown", "model": payload.get("model")}
            try:
                value = self._post_with_retry(payload)
                if key is not None:
                    with self._cache_lock:
                        # 缓存上限，防止长 episode 无界增长
                        while len(self._cache) >= 512:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[key] = value
                pending.value = value
                self._trace({**meta, "raw_output": value, "ok": True})
            except Exception as exc:  # noqa: BLE001 - 透传给等待方
                pending.error = exc
                self._trace({**meta, "error": str(exc), "ok": False})
            finally:
                pending.event.set()

    def _trace(self, record):
        if self._trace_fn is None:
            return
        try:
            item = dict(record)
            item.setdefault("t", time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._trace_fn(item)
        except Exception:
            pass

    def _post_with_retry(self, payload):
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                post_fn = self._post_fn
                if post_fn is None:
                    import requests
                    post_fn = requests.post
                headers = {"Content-Type": "application/json",
                           "Authorization": f"Bearer {self.api_key}"}
                resp = post_fn(self._chat_url(), headers=headers,
                               json=payload, timeout=self.timeout)
                status = getattr(resp, "status_code", 200)
                if status == 404:
                    raise VLLMError(
                        f"vLLM 返回 404：模型 '{payload.get('model')}' 未加载"
                        "（检查权重路径/vLLM 启动参数），不重试")
                if 400 <= status < 500 and status != 429:
                    raise VLLMError(
                        f"vLLM 请求错误 HTTP {status}，不重试")
                if status >= 500:
                    raise VLLMError(f"vLLM 服务端错误 HTTP {status}，将重试")
                resp.raise_for_status()
                return self._extract_text(resp.json())
            except VLLMError as exc:
                if "不重试" in str(exc):
                    raise
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 - 网络/解析错误均可重试
                last_exc = exc
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        raise VLLMError(
            f"vLLM 调用失败（重试 {self.max_retries} 次后仍不可用）: "
            f"{last_exc}")

    def _chat_url(self):
        return (self.url if self.url.endswith("/chat/completions")
                else f"{self.url}/chat/completions")

    @staticmethod
    def _build_payload(model, prompt, images, max_tokens):
        content = [{"type": "text", "text": str(prompt)}]
        for raw in images or []:
            if raw is None:
                continue
            if isinstance(raw, (bytes, bytearray)):
                jpeg = bytes(raw)
            else:  # PIL.Image
                buf = io.BytesIO()
                raw.convert("RGB").save(buf, format="JPEG", quality=88)
                jpeg = buf.getvalue()
            content.append({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,"
                              + base64.b64encode(jpeg).decode("ascii")},
            })
        return {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }

    @staticmethod
    def _make_cache_key(cache_key, model, kind, prompt):
        if cache_key is None:
            return None
        digest = hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:12]
        return f"{model}|{kind}|{cache_key}|{digest}"

    @staticmethod
    def _extract_text(response):
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMError(f"vLLM 响应格式异常: {response!r}") from exc
        if isinstance(content, list):
            content = "".join(str(item.get("text", ""))
                              for item in content if isinstance(item, dict))
        return str(content)

    @staticmethod
    def parse_json(text):
        """从模型输出解析 JSON 对象；失败抛 VLLMError（调用方可校验重试）。"""
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                         str(text).strip(), flags=re.IGNORECASE)
        # 模型偶尔在 JSON 前后多说话，截取第一个 { 到最后一个 }
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise VLLMError(f"模型输出不是合法 JSON: {text[:200]!r}") from exc
        if not isinstance(data, dict):
            raise VLLMError(f"模型输出不是 JSON 对象: {text[:200]!r}")
        return data
