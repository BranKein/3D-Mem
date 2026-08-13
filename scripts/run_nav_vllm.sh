#!/bin/bash
#
# Run the 3D-Mem RefON navigation evaluation against a vLLM server.
#
#   SMOKE=1 ./run_nav_vllm.sh Qwen/Qwen3.5-9B    # one episode per scene, do this first
#   ./run_nav_vllm.sh Qwen/Qwen3.5-9B            # the full split
#   MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B" ./run_nav_vllm.sh   # all three
#
# Starts a vLLM server, runs the evaluation against it, tears the server down, and
# repeats per model. Unlike the feasibility probes this evaluation drives habitat and
# a perception stack, so the two processes are pinned to different GPUs.
#
set -uo pipefail

REPO="${REPO:-$HOME/3D-Mem}"
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
PY="${PY:-$CONDA_BASE/envs/3dmem/bin/python}"
VLLM="${VLLM:-$CONDA_BASE/envs/vllm/bin/vllm}"
CFG="${CFG:-cfg/eval_refonbench_default.yaml}"
LOGDIR="${LOGDIR:-$REPO/results/vllm_nav_runs}"

# GPU split. src/scene_goatbench.py:74 hardcodes torch.device("cuda") for YOLO-World,
# SAM and CLIP, so the only way to keep them off the serving card is CUDA_VISIBLE_DEVICES
# on each process. With one GPU, set both to 0 and drop VLLM_UTIL to leave the
# perception stack room (0.55 is a reasonable start for a 9B in bf16 -- but on one
# 24 GB card a 9B in bf16 will not fit alongside it, so use QUANT=fp8 there).
VLLM_GPU="${VLLM_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-0}"
VLLM_UTIL="${VLLM_UTIL:-0.90}"

# bf16 by default: an A30 is compute capability 8.0 and has no fp8 tensor cores, so
# fp8 there only saves memory (via the Marlin path) and buys no speed. Set QUANT=fp8
# if you are memory-bound instead.
QUANT="${QUANT:-}"

# Images make these prompts far bigger than the text-only probes: egocentric views,
# frontier images, snapshot images and per-object crops all go in one request.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
PORT="${PORT:-8000}"
SMOKE="${SMOKE:-0}"

# Sampling. The evaluation builds its client at import time and cannot read cfg, so
# these arrive through the environment (src/const.py -> create_vlm_client).
#   max_tokens: the built-in default is 4096, below what a thinking reply needs
#               (median ~4900 tokens measured on Qwen3.5-2B).
#   temperature + presence_penalty: with thinking on, greedy decoding loops until the
#               budget is gone. Measured over 23 real prompts on Qwen3.5-2B:
#                 T0.6 pp0.0 -> 11/23 truncated;  T0.6 pp0.0 at 60k -> 13/23 (worse)
#                 T0.6 pp1.5 ->  1/23;            T0.7 pp1.5        ->  0/23
export VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-32768}"
export VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.7}"
export VLM_PRESENCE_PENALTY="${VLM_PRESENCE_PENALTY:-1.5}"

read -r -a MODEL_LIST <<< "${MODELS:-${1:-Qwen/Qwen3.5-9B}}"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

SERVER_PID=""
stop_server() {
    [ -z "$SERVER_PID" ] && return 0
    echo "[$(date +%H:%M:%S)] stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""; sleep 10
}
trap 'stop_server; exit 130' INT TERM

slug() { echo "$1" | sed 's|.*/||' | tr 'A-Z.' 'a-z_' | sed 's/[^a-z0-9]\+/_/g; s/^_//; s/_$//'; }

for MODEL in "${MODEL_LIST[@]}"; do
    SERVED="$(basename "$MODEL")"
    SLUG="$(slug "$MODEL")_vllm"
    EXP="exp_eval_refonbench_default_${SLUG}"
    echo
    echo "=== $MODEL -> results/$EXP ==="

    args=(--served-model-name "$SERVED" --max-model-len "$MAX_MODEL_LEN"
          --reasoning-parser qwen3 --gpu-memory-utilization "$VLLM_UTIL" --port "$PORT")
    [ -n "$QUANT" ] && args+=(--quantization "$QUANT")

    SERVER_LOG="$LOGDIR/server_${SLUG}.log"
    echo "[$(date +%H:%M:%S)] starting vLLM on GPU $VLLM_GPU -> $SERVER_LOG"
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" "$VLLM" serve "$MODEL" "${args[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    ready=0
    for _ in $(seq 1 360); do
        curl -sf -m 5 "http://localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 10
    done
    if [ "$ready" != 1 ]; then
        echo "  server never came up; last lines:"; tail -40 "$SERVER_LOG"; stop_server; continue
    fi
    echo "  server up"
    grep -aE "GPU KV cache size|Maximum concurrency" "$SERVER_LOG" | tail -2

    extra=()
    [ "$SMOKE" -gt 0 ] && extra=(--end_ratio 0.05)   # a slice, for a shakedown

    RUNLOG="$LOGDIR/run_${SLUG}.log"
    echo "[$(date +%H:%M:%S)] evaluating on GPU $EVAL_GPU (log: $RUNLOG)"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" VLM_PROVIDER=vllm VLLM_MODEL="$SERVED" VLLM_TIMEOUT=1800 \
        "$PY" run_refonbench_evaluation.py -cf "$CFG" --exp-name "$EXP" "${extra[@]}" \
        > "$RUNLOG" 2>&1
    echo "  exit $?"

    # The single most useful thing to look at before trusting any number: an empty
    # reply is scored as a wrong answer, so a run can look finished and mean nothing.
    grep -acE "call_vlm_api returns None" "$RUNLOG" | sed 's/^/  empty replies: /'
    stop_server
done

echo
echo "logs in $LOGDIR, results in $REPO/results/exp_eval_refonbench_default_*_vllm"
