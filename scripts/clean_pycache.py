"""清理项目内 Python 缓存。

用法：
    python scripts/clean_pycache.py
    python scripts/clean_pycache.py --dry-run

只处理项目根目录以内的 ``__pycache__`` 目录、``.pyc`` 和 ``.pyo`` 文件，
不会跟随符号链接访问项目外部路径。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_cache_paths(root: Path) -> tuple[list[Path], list[Path]]:
    """返回待删除的缓存目录和缓存文件。"""
    cache_dirs: list[Path] = []
    cache_files: list[Path] = []

    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir() and path.name == "__pycache__":
            cache_dirs.append(path)
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            cache_files.append(path)

    # __pycache__ 目录会整体删除，因此其中的 .pyc 不需要再次列入目标。
    cache_dirs = sorted(cache_dirs, key=lambda path: len(path.parts), reverse=True)
    cache_files = [
        path for path in cache_files
        if not any(cache_dir in path.parents for cache_dir in cache_dirs)
    ]
    return cache_dirs, cache_files


def main() -> int:
    parser = argparse.ArgumentParser(description="清理项目内 Python 缓存")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要删除的路径，不实际删除",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cache_dirs, cache_files = find_cache_paths(root)
    targets = sorted(cache_dirs + cache_files)

    if not targets:
        print(f"未发现 Python 缓存：{root}")
        return 0

    for target in targets:
        print(("[dry-run] " if args.dry_run else "删除 ") + str(target))
        if args.dry_run:
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        except FileNotFoundError:
            # 允许多个清理进程同时运行，或目录已被外部工具删除。
            continue

    if not args.dry_run:
        print(f"已清理 {len(cache_dirs)} 个缓存目录、{len(cache_files)} 个缓存文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
