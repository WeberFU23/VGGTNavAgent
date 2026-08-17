#!/usr/bin/env bash
# 四任务模式 smoke 矩阵（远端 AutoDL 用，本地不要跑）。
#
# run_eval.py 没有模式筛选参数——single/any/many/all 由 episodes.json
# 决定，聚合输出按模式分组（sr_mos/sr_moc/sr_any/...）。本脚本对每个
# 统一语义记忆链路跑一次混合 episode，再单独跑红线 episode
#（ep0000004 any basket，F1=1.0 不许回退）。
#
# 用法（在 habitat 环境、benchmark 仓库根目录）：
#   bash /path/to/vggt_nav_agent/scripts/run_smoke_matrix.sh
#
# 前置：mapping server 已在 vggtslam 环境启动（python -m mapping.server），
# 语义记忆需要 vLLM 服务与三个模型权重（见
# AGENT_ARCHITECTURE.md 的 flag 表：NAV_VLLM_URL / NAV_CAPTION_MODEL_PATH /
# NAV_POINTING_MODEL_PATH / NAV_EMBED_MODEL_PATH）。

set -u

BENCH_DIR=${BENCH_DIR:-/root/habitat_benchmark}
SCENE_ROOT=${SCENE_ROOT:-/root/autodl-tmp/hm3d/val}
CONFIG=${CONFIG:-evaluation/main/hm3d_config.yaml}
AGENT=${AGENT:-agents.nav_agent:NavAgent}
MAX_STEPS=${MAX_STEPS:-500}
EPISODE_LIMIT=${EPISODE_LIMIT:-8}          # 混合模式 smoke
EPISODE_REDLINE=${EPISODE_REDLINE:-ep0000004}
OUT_ROOT=${OUT_ROOT:-smoke_matrix_$(date +%m%d_%H%M)}

mkdir -p "$OUT_ROOT"
cd "$BENCH_DIR" || exit 1

run_eval() {
  local tag=$1; shift
  echo "=== $tag ==="
  python evaluation/main/run_eval.py \
    --config "$CONFIG" \
    --agent "$AGENT" \
    --goal-type description \
    --dataset-dir dataset_semantic \
    --scene-root "$SCENE_ROOT" \
    --max-steps "$MAX_STEPS" \
    --log-episodes "$@" 2>&1 | tee "$OUT_ROOT/${tag}.log"
}

# 混合模式：聚合输出里看 sr_mos / sr_moc / f1 / spl_multi
run_eval "semantic_mixed" --limit 1 --episode-limit "$EPISODE_LIMIT"
# 红线 episode：any basket，F1=1.0 不许回退
run_eval "semantic_redline" --limit 1 --episode-id "$EPISODE_REDLINE"

echo
echo "smoke 矩阵完成，日志在 $OUT_ROOT/。"
echo "检查 sr/f1/spl_multi（many/all 主榜单）；红线：${EPISODE_REDLINE} F1=1.0。"
