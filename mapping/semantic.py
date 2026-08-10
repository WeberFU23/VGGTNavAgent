"""语义层：CLIP 关键帧记忆 + SAM3 按需实例定位（server 端，vggtslam 环境）。

设计（参考 FOUND-IT 的两级检索）：

1. CLIP 关键帧记忆：每个子图处理时顺带为关键帧算 CLIP 图像向量，
   存入 Submap.semantic_vectors（复用 VGGT-SLAM 内置字段）。文本查询
   时计算余弦相似度，返回 top-K 关键帧。
2. SAM3 实例定位：仅在 ground_object 查询时按需加载/调用，对 top-K
   关键帧做文本提示分割，mask 内点云质心即为 3D 目标点。

CLIP 使用 openai/clip-vit-base-patch32（HuggingFace transformers），
通过下面的 adapter 复用 vggt_slam.slam_utils 里的批量编码函数。
"""

import numpy as np
import torch


class HFClipAdapter:
    """把 transformers CLIP 包装成 slam_utils 期望的接口。

    slam_utils.compute_image_embeddings 需要:
        preprocess(PIL.Image) -> Tensor (3,H,W)
        model.encode_image(batch_tensor) -> (N, D)
    slam_utils.compute_text_embeddings 需要:
        tokenizer([text]) -> 可 .to(device) 的 BatchFeature
        model.encode_text(tokens) -> (1, D)
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.context_length = self.model.config.text_config.max_position_embeddings

    def image_preprocess(self, pil_img):
        return self.processor(
            images=pil_img, return_tensors="pt").pixel_values[0]

    def text_tokenizer(self, texts):
        return self.processor(
            text=list(texts), return_tensors="pt", padding=True,
            truncation=True, max_length=self.context_length)

    def encode_image(self, batch):
        return self.model.get_image_features(pixel_values=batch)

    def encode_text(self, tokens):
        return self.model.get_text_features(**tokens)


class Sam3Grounder:
    """SAM3 文本提示分割，懒加载（首次使用时才占显存）。"""

    # SAM3 对提示词措辞敏感（实测 "tv monitor" 得分 0.12 而
    # "flat screen tv" 0.64）。此处为已知难处理的类别名配置同义变体，
    # 查询时多个变体各跑一次，取最高分变体的结果。
    # basket 类别实测 CLIP 检索 0.21-0.24 无区分度，同义词可小幅
    # 提升 0.22 -> 0.25，且 SAM3 措辞敏感，值得展开。
    PROMPT_SYNONYMS = {
        "tv monitor": ["flat screen tv", "television", "tv"],
        "couch": ["sofa"],
        "basket": ["laundry basket", "storage basket", "hamper"],
        "baskets": ["laundry baskets", "storage baskets", "baskets with handles"],
    }

    def __init__(self, confidence_threshold=0.25, device="cuda"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._processor = None

    def _load(self):
        if self._processor is not None:
            return
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        self._processor = Sam3Processor(
            model, confidence_threshold=self.confidence_threshold)

    def ground(self, pil_img, text):
        """text 可为 str 或 str 列表（多个提示变体，取最高分变体的结果）。
        返回 (masks, boxes, scores, best_prompt)；无命中时 M=0。"""
        self._load()
        variants = [text] if isinstance(text, str) else list(text)
        # SAM3 内部 addmm_act 强制以 bf16 计算（其设计运行精度），
        # 整个推理需包在 bf16 autocast 下，否则 bf16 中间激活撞上 fp32 权重。
        with torch.no_grad(), \
                torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # backbone 前向昂贵，只做一次；多个文本变体只重跑轻量解码。
            # 注意：SAM3 的 set_text_prompt 复用并原地覆盖同一个 output
            # 对象，循环结束后保存的引用会指向最后一个变体（可能为空），
            # 因此必须在循环内立即把结果拷贝到 CPU。
            state = self._processor.set_image(pil_img)
            best = None  # (top_score, masks_np, boxes_np, scores_np, prompt)
            for prompt in variants:
                output = self._processor.set_text_prompt(
                    state=state, prompt=prompt)
                scores = output["scores"].float()
                top = float(scores.max()) if scores.numel() else 0.0
                if best is None or top > best[0]:
                    best = (top,
                            output["masks"].float().cpu().numpy(),
                            output["boxes"].float().cpu().numpy(),
                            scores.float().cpu().numpy(),
                            prompt)
        if best is None:
            return (np.zeros((0, pil_img.height, pil_img.width), dtype=bool),
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32), None)
        _top, masks, boxes, scores, best_prompt = best
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        return masks.astype(bool), boxes, scores, best_prompt

    def expand_prompts(self, text):
        """原始提示 + 同义变体（去重，保持顺序）。

        复数与单数都尝试查同义词表（"baskets" -> "basket" 条目），
        避免 "Find exactly two baskets" 得到的 "baskets" 匹配不上。
        """
        text = str(text).strip()
        variants = [text]
        base = text[:-1] if text.endswith("s") and len(text) > 1 else text
        seen_keys = set()
        for key in (text.lower(), base.lower()):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            for syn in self.PROMPT_SYNONYMS.get(key, []):
                if syn not in variants:
                    variants.append(syn)
        return variants
