#!/bin/bash
# 启动 VGGT-SLAM 建图服务端（在 vggtslam conda 环境中）。
# 用法: bash scripts/run_mapping_server.sh [额外参数传给 mapping.server]
set -euo pipefail
VGGT_SLAM_ENV=${VGGT_SLAM_ENV:-vggtslam}
# 兼容不同机器的 conda 安装位置
_conda_loaded=0
for _conda_sh in ~/anaconda3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$_conda_sh" ]; then
    source "$_conda_sh"
    _conda_loaded=1
    break
  fi
done
if [ "$_conda_loaded" -ne 1 ]; then
  echo "Conda initialization script not found" >&2
  exit 1
fi
conda activate "$VGGT_SLAM_ENV"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
# 集群无外网：强制 transformers/huggingface_hub 走本地缓存，
# 避免每次加载模型先做数轮网络重试
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec python -u -m mapping.server "$@"
