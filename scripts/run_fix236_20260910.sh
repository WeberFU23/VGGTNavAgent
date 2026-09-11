#!/usr/bin/env bash
# fix236: 基于 9/10 批次结果改进后的代码重测。
# 重测上次 description 4 个（bxsVRursffK 上次实为 image 类型，沿用原类型）
# + 随机新选 4 个（seed 20260910，避开已跑 episode）。
# 4 条 lane 各 2 个 ep，场间错开 240s，避免同时起 SAM 占满 GPU。
# 每 ep 上限：300 步 或 900s(15min)，先到即止。
set -u
PROJECT=/root/autodl-tmp/vggt_nav_agent
BENCH=/root/autodl-tmp/habitat_benchmark_eval_20260827
RUN_ROOT=/root/autodl-tmp/runs_fix236_20260910
DATASET="$BENCH/benchmark_v10/semantic"
SCENE_ROOT=/root/autodl-tmp/datasets/hm3d/val
STAGGER=240

mkdir -p "$RUN_ROOT"

run_ep() {
  local EP=$1 PORT=$2 GOAL_TYPE=$3
  local ER="$RUN_ROOT/$EP"
  mkdir -p "$ER"
  echo "[run] $(date) === $EP ($GOAL_TYPE): start mapping server (port $PORT) ===" | tee -a "$RUN_ROOT/run.log"
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
  local MAP_PID=$!
  sleep 45
  if ! kill -0 $MAP_PID 2>/dev/null; then
    echo mapping_start_failed > "$ER/eval.exit"
    echo "[run] $(date) mapping failed for $EP" | tee -a "$RUN_ROOT/run.log"
    return 1
  fi
  local sam_state=""
  for _ in $(seq 1 72); do
    if grep -q 'SAM 启动' "$ER/mapping.log" 2>/dev/null; then
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
    return 1
  fi
  echo "[run] $(date) $EP mapping ready (SAM ok, port $PORT), eval start" | tee -a "$RUN_ROOT/run.log"
  bash -c "
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate habitat
    set -a; source '$PROJECT/.env'; set +a
    export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=$PORT
    export NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3
    export NAV_VLM_TRACE_INLINE_IMAGES=0 NAV_DEBUG_ROOT='$ER/debug_output' NAV_RUN_ID='$EP'
    export PYTHONPATH='$PROJECT:$BENCH:$BENCH/evaluation/main'
    cd '$BENCH'
    exec python -u evaluation/main/run_eval.py \
      --config evaluation/main/hm3d_config.yaml \
      --dataset-dir '$DATASET' --scene-root '$SCENE_ROOT' \
      --episode-id '$EP' --goal-type $GOAL_TYPE --same-floor-only \
      --same-floor-height-threshold 0.6 \
      --agent agents.nav_agent:NavAgent --action-protocol goat \
      --max-steps 300 --max-steps-per-target 300 \
      --episode-timeout-seconds 900 --log-episodes --log-actions \
      --log-prompts --log-positions > '$ER/eval.log' 2>&1
  " &
  local EVAL_PID=$!
  local vlm_ok=0
  for _ in $(seq 1 24); do
    if grep -q 'VLM 策略包: enabled' "$ER/eval.log" 2>/dev/null; then
      vlm_ok=1
      break
    fi
    if grep -q 'VLM 策略包: disabled' "$ER/eval.log" 2>/dev/null; then
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
    return 1
  fi
  echo "[run] $(date) $EP decision VLM enabled, eval running" | tee -a "$RUN_ROOT/run.log"
  wait $EVAL_PID
  local rc=$?
  echo $rc > "$ER/eval.exit"
  kill $MAP_PID 2>/dev/null || true
  wait $MAP_PID 2>/dev/null || true
  echo "[run] $(date) $EP eval done rc=$rc" | tee -a "$RUN_ROOT/run.log"
  return $rc
}

run_lane() {
  local LANE=$1 PORT=$2; shift 2
  sleep $(( (LANE - 1) * STAGGER ))
  echo "[run] $(date) lane$LANE start (port $PORT): $*" | tee -a "$RUN_ROOT/run.log"
  for PAIR in "$@"; do
    local EP="${PAIR%%:*}" GT="${PAIR##*:}"
    run_ep "$EP" "$PORT" "$GT"
  done
  echo "[run] $(date) lane$LANE queue done" | tee -a "$RUN_ROOT/run.log"
}

# 重测：HY1NcmCgn3n/5cdEh9F2hJL/VBzV5z6i1WS (description) + bxsVRursffK (image)
# 随机（seed 20260910）新选：CrMo8WxCyVb (image) / Nfvxx8J5NCo (image) / GLAQ4DNUx5U (desc) / 7MXmsvcQjpJ (desc)
run_lane 1 5555 HY1NcmCgn3n_ep0000005:description CrMo8WxCyVb_ep0000020:image &
run_lane 2 5556 bxsVRursffK_ep0000018:image Nfvxx8J5NCo_ep0000015:image &
run_lane 3 5557 5cdEh9F2hJL_ep0000016:description GLAQ4DNUx5U_ep0000004:description &
run_lane 4 5558 VBzV5z6i1WS_ep0000002:description 7MXmsvcQjpJ_ep0000020:description &
wait
echo "[run] $(date) ALL DONE" | tee -a "$RUN_ROOT/run.log"