"""把 vlm_calls/vlm_caption/vlm_pointing 的 JSONL 日志转成可读目录。

用法：
    python scripts/pretty_vlm_log.py <input.jsonl> [-o output_dir]

每条记录生成一个子目录：
    0001/prompt.txt        prompt 原文（真实换行）
    0001/images/00_<label>.jpg   内联 base64 图像解码落盘
    0001/output.json       raw_response/parsed_output 美化后的 JSON
    0001/meta.json         t/kind/model/ok/context 等元信息
索引文件 index.md 汇总每条记录的时间、类型、输出摘要，便于快速翻阅。
"""
import argparse
import base64
import io
import json
import os
import re


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:80] or "img"


def _extract_images(record, img_dir):
    """从 images 字段提取图像。返回 [(label, filename)]。"""
    saved = []
    for i, img in enumerate(record.get("images") or []):
        if not isinstance(img, dict):
            continue
        label = _safe_name(img.get("label") or f"image_{i}")
        data = img.get("data_b64") or ""
        data_url = img.get("data_url") or ""
        if not data and "base64," in str(data_url):
            data = str(data_url).split("base64,", 1)[1]
        if not data:
            continue
        ext = ".png" if "png" in str(img.get("mime_type", "")) else ".jpg"
        fname = f"{i:02d}_{label}{ext}"
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, fname), "wb") as f:
            f.write(base64.b64decode(data))
        saved.append((label, fname))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    out_dir = args.output or os.path.splitext(os.path.basename(
        args.input))[0] + "_pretty"
    os.makedirs(out_dir, exist_ok=True)

    index_lines = ["# VLM log readable dump", "",
                   f"source: `{os.path.abspath(args.input)}`", ""]
    with open(args.input, encoding="utf-8") as fp:
        for n, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rec_dir = os.path.join(out_dir, f"{n:04d}")
            os.makedirs(rec_dir, exist_ok=True)

            # prompt 原文（真实换行）
            prompt = record.get("prompt")
            if prompt:
                with open(os.path.join(rec_dir, "prompt.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(str(prompt))

            # 图像
            images = _extract_images(record, os.path.join(rec_dir, "images"))

            # 输出（美化 JSON）
            output = {}
            for key in ("raw_response", "parsed_output", "raw_output",
                        "response"):
                if key in record:
                    output[key] = record[key]
            if output:
                with open(os.path.join(rec_dir, "output.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2,
                              default=str)

            # 元信息（去掉大字段）
            meta = {k: v for k, v in record.items()
                    if k not in ("prompt", "images", "raw_response",
                                 "parsed_output", "raw_output", "response")}
            with open(os.path.join(rec_dir, "meta.json"), "w",
                      encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

            # 索引行
            out = record.get("parsed_output") or record.get("raw_output") or ""
            if isinstance(out, dict):
                out = json.dumps(out, ensure_ascii=False)
            out = " ".join(str(out).split())[:150]
            img_md = " ".join(
                f"[{label}]({n:04d}/images/{fname})"
                for label, fname in images)
            index_lines.append(
                f"## {n:04d} | {record.get('t', '')} | "
                f"{record.get('kind', '')} | ok={record.get('ok')}")
            index_lines.append("")
            if out:
                index_lines.append(f"output: {out}")
                index_lines.append("")
            if img_md:
                index_lines.append(f"images: {img_md}")
                index_lines.append("")
            index_lines.append(f"[prompt]({n:04d}/prompt.txt) | "
                               f"[output.json]({n:04d}/output.json) | "
                               f"[meta]({n:04d}/meta.json)")
            index_lines.append("")

    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
