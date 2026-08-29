#!/usr/bin/env bash
# 3-episode check run: any / many / all, each with <4 targets.
# Based on run_10eps_semantic_verify.sh (Molmo pointing preflight included).
set -u

PROJECT=/root/autodl-tmp/vggt_nav_agent
BENCH=/root/autodl-tmp/habitat_benchmark_eval_20260827
RUN_ROOT=/root/autodl-tmp/runs_3ep_20260829
DATASET="$BENCH/benchmark_v3/semantic"
SCENE_ROOT=/root/autodl-tmp/datasets/hm3d/val
POINTING_URL=http://127.0.0.1:8000/v1
POINTING_MODEL=molmo-7b-d-0924
EPS="
5cdEh9F2hJL_ep0000006
6s7QHgap2fW_ep0000002
4ok3usBNeis_ep0000016
"

mkdir -p "$RUN_ROOT"

models_json=$(curl -fsS --max-time 10 "$POINTING_URL/models") || {
  echo "[run] POINTING_BACKEND_UNAVAILABLE: $POINTING_URL/models" \
    | tee -a "$RUN_ROOT/run.log"
  exit 20
}
MODELS_JSON="$models_json" POINTING_MODEL="$POINTING_MODEL" python -c \
  'import json,os,sys; ids=[str(x.get("id")) for x in json.loads(os.environ["MODELS_JSON"]).get("data",[])]; model=os.environ["POINTING_MODEL"]; print("[run] pointing models:", ids); sys.exit(0 if model in ids else 21)' \
  | tee -a "$RUN_ROOT/run.log"
preflight_rc=${PIPESTATUS[0]}
if [[ "$preflight_rc" -ne 0 ]]; then
  echo "[run] POINTING_BACKEND_UNAVAILABLE: model $POINTING_MODEL not loaded" \
    | tee -a "$RUN_ROOT/run.log"
  exit "$preflight_rc"
fi
echo "[run] $(date) pointing preflight passed" | tee -a "$RUN_ROOT/run.log"

for EP in $EPS; do
  ER="$RUN_ROOT/$EP"
  mkdir -p "$ER"
  echo "[run] $(date) === $EP: restart mapping server ===" \
    | tee -a "$RUN_ROOT/run.log"

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
    if grep -q 'pointing grounder 就绪' "$ER/mapping.log" 2>/dev/null && \
       (exec 3<>/dev/tcp/127.0.0.1/5555) 2>/dev/null; then
      ready=1
      break
    fi
    if grep -q 'PointingBackendUnavailable\|health check failed' \
         "$ER/mapping.log" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[run] $EP mapping/pointing server failed to start; abort run" \
      | tee -a "$RUN_ROOT/run.log"
    echo mapping_start_failed > "$ER/eval.exit"
    exit 22
  fi
  echo "[run] $(date) $EP mapping and pointing ready" \
    | tee -a "$RUN_ROOT/run.log"

  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate habitat
  set -a
  source "$PROJECT/.env"
  set +a
  export VGGT_SLAM_HOST=127.0.0.1 VGGT_SLAM_PORT=5555
  export NAV_EMBED_MODEL_PATH=/root/autodl-tmp/models/bge-m3
  export NAV_VLM_TRACE_INLINE_IMAGES=0
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
  if grep -q 'No episodes were evaluated' "$ER/eval.log"; then
    rc=23
  fi
  echo "$SECONDS" > "$ER/eval.seconds"
  echo "$rc" > "$ER/eval.exit"
  echo "[run] $(date) eval $EP done rc=$rc" | tee -a "$RUN_ROOT/run.log"
  if [[ "$rc" -eq 23 ]]; then
    echo "[run] invalid episode selection; abort run" | tee -a "$RUN_ROOT/run.log"
    exit 23
  fi
done

pkill -f 'mapping.server --port 5555' 2>/dev/null || true
echo "[run] $(date) all episodes done" | tee -a "$RUN_ROOT/run.log"
