#!/usr/bin/env bash
# 3-episode harness test run: TEEsavR23oF ep0000001/2/3
# 每个 episode 重启 mapping server，保证干净状态；日志按 episode 分目录。
set -u

PROJECT=/root/autodl-tmp/vggt_nav_agent
BENCH=/root/autodl-tmp/habitat_benchmark_eval_20260827
RUN_ROOT=/root/autodl-tmp/runs_ep18_molmo_20260827
DATASET=/root/autodl-tmp/habitat_benchmark_eval_20260827/benchmark_v3/semantic
SCENE_ROOT=/root/autodl-tmp/datasets/hm3d/val
EPS="TEEsavR23oF_ep0000018"

mkdir -p "$RUN_ROOT"

for EP in $EPS; do
  ER="$RUN_ROOT/$EP"
  mkdir -p "$ER"
  echo "[run] $(date) === $EP: restart mapping server ===" | tee -a "$RUN_ROOT/run.log"

  pkill -f 'mapping.server --port 5555' 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f 'mapping.server --port 5555' >/dev/null || break
    sleep 2
  done

  screen -dmS nav-mapping-3ep bash -lc \
    "source /root/miniconda3/etc/profile.d/conda.sh; conda activate vggtslam; \
     set -a; source '$PROJECT/.env'; set +a; \
     export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=5555 \
       HF_HUB_OFFLINE=1 \
       NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3 NAV_EMBED_DEVICE=cpu \
       NAV_VLLM_TRACE_IMAGES=1 \
       NAV_CAPTION_STORE_PATH='$ER/caption_store' \
       NAV_DEBUG_ROOT='$ER/debug_output' NAV_RUN_ID='$EP' \
       PYTHONPATH='$PROJECT:$BENCH:$BENCH/evaluation/main'; \
     cd '$PROJECT'; exec bash scripts/run_mapping_server.sh --port 5555 \
       >> '$ER/mapping.log' 2>&1"

  ready=0
  for _ in $(seq 1 90); do
    if grep -q '模型加载完成' "$ER/mapping.log" 2>/dev/null && \
       (exec 3<>/dev/tcp/127.0.0.1/5555) 2>/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[run] $EP mapping server failed to start" | tee -a "$RUN_ROOT/run.log"
    echo mapping_start_failed > "$ER/eval.exit"
    continue
  fi
  echo "[run] $(date) $EP mapping server ready" | tee -a "$RUN_ROOT/run.log"

  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate habitat
  set -a
  source "$PROJECT/.env"
  set +a
  export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=5555
  export NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3
  export NAV_VLM_TRACE_INLINE_IMAGES=1
  export NAV_DEBUG_ROOT="$ER/debug_output"
  export NAV_RUN_ID="$EP"
  export PYTHONPATH="$PROJECT:$BENCH:$BENCH/evaluation/main"

  cd "$BENCH"
  SECONDS=0
  echo "[run] $(date) eval $EP start" | tee -a "$RUN_ROOT/run.log"
  python -u evaluation/main/run_eval.py \
    --config evaluation/main/hm3d_config.yaml \
    --dataset-dir "$DATASET" \
    --scene-root "$SCENE_ROOT" \
    --episode-id "$EP" \
    --same-floor-only --same-floor-height-threshold 0.6 \
    --agent agents.nav_agent:NavAgent \
    --action-protocol goat \
    --max-steps 200 \
    --max-steps-per-target 200 \
    --episode-timeout-seconds 1800 \
    --log-episodes --log-actions --log-prompts --log-positions \
    > "$ER/eval.log" 2>&1
  rc=$?
  echo "$SECONDS" > "$ER/eval.seconds"
  echo "$rc" > "$ER/eval.exit"
  echo "[run] $(date) eval $EP done rc=$rc" | tee -a "$RUN_ROOT/run.log"
done

pkill -f 'mapping.server --port 5555' 2>/dev/null || true
echo "[run] $(date) all episodes done" | tee -a "$RUN_ROOT/run.log"
