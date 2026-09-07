#!/usr/bin/env bash
# v27r2 retest: 重跑 v27r2 轮 6 个 ep（当前远端最新代码）
set -u
PROJECT=/root/autodl-tmp/vggt_nav_agent
BENCH=/root/autodl-tmp/habitat_benchmark_eval_20260827
RUN_ROOT=/root/autodl-tmp/runs_v27r2retest_20260906
DATASET="$BENCH/benchmark_v6/semantic"
SCENE_ROOT=/root/autodl-tmp/datasets/hm3d/val
PORT=5555
EPS="6s7QHgap2fW_ep0000010 yr17PDCnDDW_ep0000001 h1zeeAwLh9Z_ep0000011 q3zU7Yy5E5s_ep0000007 GLAQ4DNUx5U_ep0000014 DYehNKdT76V_ep0000008"

mkdir -p "$RUN_ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

for EP in $EPS; do
  ER="$RUN_ROOT/$EP"
  mkdir -p "$ER"
  echo "[run] $(date) === $EP: start mapping server ===" | tee -a "$RUN_ROOT/run.log"
  bash -c "
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate vggtslam
    set -a; source '$PROJECT/.env'; set +a
    export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=$PORT HF_HUB_OFFLINE=1
    export NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3 NAV_EMBED_DEVICE=cpu
    export NAV_VLM_TRACE_IMAGES=1 NAV_CAPTION_STORE_PATH='$ER/caption_store'
    export NAV_DEBUG_ROOT='$ER/debug_output' NAV_RUN_ID='$EP'
    export PYTHONPATH='$PROJECT:$BENCH:$BENCH/evaluation/main'
    cd '$PROJECT'
    exec bash scripts/run_mapping_server.sh --port $PORT > '$ER/mapping.log' 2>&1
  " &
  MAP_PID=$!
  sleep 45
  if ! kill -0 $MAP_PID 2>/dev/null; then
    echo mapping_start_failed > "$ER/eval.exit"
    echo "[run] $(date) mapping failed for $EP" | tee -a "$RUN_ROOT/run.log"
    continue
  fi
  # SAM 主链路完全依赖 SAM：轮询最多 180s 等 SAM 就绪/不可用
  sam_state=""
  for _ in $(seq 1 36); do
    if grep -q 'SAM 就绪' "$ER/mapping.log" 2>/dev/null; then
      sam_state=ok
      break
    fi
    if grep -q 'SAM 不可用' "$ER/mapping.log" 2>/dev/null; then
      sam_state=unavailable
      break
    fi
    if ! kill -0 $MAP_PID 2>/dev/null; then
      sam_state=crashed
      break
    fi
    sleep 5
  done
  if [ "$sam_state" != "ok" ]; then
    echo "[run] $(date) FATAL: SAM $sam_state for $EP" | tee -a "$RUN_ROOT/run.log"
    kill $MAP_PID 2>/dev/null || true
    wait $MAP_PID 2>/dev/null || true
    echo "sam_$sam_state" > "$ER/eval.exit"
    continue
  fi
  echo "[run] $(date) $EP mapping ready (SAM ok), eval start" | tee -a "$RUN_ROOT/run.log"
  conda activate habitat
  set -a; source "$PROJECT/.env"; set +a
  export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=$PORT
  export NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3
  export NAV_VLM_TRACE_INLINE_IMAGES=0 NAV_DEBUG_ROOT="$ER/debug_output" NAV_RUN_ID="$EP"
  export PYTHONPATH="$PROJECT:$BENCH:$BENCH/evaluation/main"
  cd "$BENCH"
  python -u evaluation/main/run_eval.py \
    --config evaluation/main/hm3d_config.yaml \
    --dataset-dir "$DATASET" --scene-root "$SCENE_ROOT" \
    --episode-id "$EP" --same-floor-only --same-floor-height-threshold 0.6 \
    --agent agents.nav_agent:NavAgent --action-protocol goat \
    --max-steps 300 --max-steps-per-target 300 \
    --episode-timeout-seconds 7200 --log-episodes --log-actions \
    --log-prompts --log-positions > "$ER/eval.log" 2>&1 &
  EVAL_PID=$!
  vlm_ok=0
  for _ in $(seq 1 24); do
    if grep -q 'VLM 战略层: enabled' "$ER/eval.log" 2>/dev/null; then
      vlm_ok=1
      break
    fi
    if grep -q 'VLM 战略层: disabled' "$ER/eval.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 $EVAL_PID 2>/dev/null; then
      break
    fi
    sleep 5
  done
  if [ "$vlm_ok" -ne 1 ]; then
    echo "[run] $(date) FATAL: decision VLM not enabled; abort $EP" | tee -a "$RUN_ROOT/run.log"
    kill $EVAL_PID $MAP_PID 2>/dev/null || true
    echo vlm_disabled > "$ER/eval.exit"
    wait $EVAL_PID 2>/dev/null; wait $MAP_PID 2>/dev/null
    continue
  fi
  echo "[run] $(date) $EP decision VLM enabled, eval running" | tee -a "$RUN_ROOT/run.log"
  wait $EVAL_PID
  rc=$?
  echo $rc > "$ER/eval.exit"
  kill $MAP_PID 2>/dev/null || true
  wait $MAP_PID 2>/dev/null || true
  echo "[run] $(date) $EP eval done rc=$rc" | tee -a "$RUN_ROOT/run.log"
done
echo "[run] $(date) ALL DONE" | tee -a "$RUN_ROOT/run.log"
