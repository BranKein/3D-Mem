#!/bin/bash
#
# All three feasibility probes x three qwen3.5 sizes, served by vLLM with thinking on.
#
#   run_vllm_feasibility_all.sh              # the full sweep
#   SMOKE=2 run_vllm_feasibility_all.sh      # 2 episodes per shard, to shake out config
#
# One vLLM server per model (vLLM serves one model per process), three probes against
# it, then the server comes down and the next model goes up. Nine runs in total,
# landing in results/exp_feasibility{,_nav,_long}_refonbench_qwen3_5_{2,4,9}b_vllm/.
#
# Restartable: a probe whose results json already exists is skipped, so re-running
# after a crash picks up where it stopped.
#
# Two deliberate choices:
#
#   fp8 for all three sizes, not only 9B. 2B and 4B fit in bf16 easily, but a ladder
#   whose rungs are served at different precisions is not one condition -- the same
#   objection that already keeps these numbers apart from the ollama runs.
#
#   Thinking stays ON, so --reasoning-effort is never passed. The output directories
#   therefore carry no `nothink` tag, which together with the `_vllm` suffix is what
#   separates them from the existing ollama results.
#
set -uo pipefail

REPO="${REPO:-/home/yhkim/3D-Mem}"
PY="${PY:-$HOME/anaconda3/envs/3dmem/bin/python}"
VLLM="${VLLM:-$HOME/anaconda3/envs/vllm/bin/vllm}"
LOGDIR="${LOGDIR:-$REPO/results/vllm_runs}"

MODELS_DEFAULT="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
read -r -a MODELS <<< "${MODELS:-$MODELS_DEFAULT}"

# Bounds prompt + completion together. cfg.max_tokens is 32768 and thinking does use
# it, so the completion budget is what sets this -- the prompts themselves are a
# couple of thousand tokens even on the 50-subgoal long episodes.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-16}"            # vLLM batches continuously; 4 would leave it idle
SERVER_WAIT="${SERVER_WAIT:-3600}"  # a first start also downloads the weights

# vLLM refuses to start unless FREE memory is at least this fraction of TOTAL, so the
# ceiling is not 1.0 minus what we need -- it is 1.0 minus whatever anything else on
# the card holds. A desktop session (rustdesk, X) sits on ~2.5 GiB here, which already
# makes 0.90 (21.15 of 23.49 GiB) fail against 21.01 GiB free. 0.85 leaves that margin
# and still gives 9B fp8 about 11 GiB of KV cache.
GPU_UTIL="${GPU_UTIL:-0.85}"
SMOKE="${SMOKE:-0}"                 # >0: --episodes-per-scene N, for a quick shakedown

# The configs say temperature 0.0, which was right when every run had thinking off.
# With thinking on it is not: greedy decoding sends Qwen3.5 into a repetition loop
# that never emits the closing </think>, so every reply hits max_tokens with empty
# content and the runner drops the model. Measured on Qwen3.5-2B, same prompt:
#   0.0 -> length, 32768 tokens, no answer
#   0.6 -> stop,    2135 tokens, correct
#   0.7 -> stop,    3426 tokens, correct
# 0.6 is what Qwen recommends for thinking mode. Passed on the command line so the
# configs keep reproducing the existing ollama runs unchanged.
TEMPERATURE="${TEMPERATURE:-0.7}"

# The actual brake on the loop. Over 23 real prompts on Qwen3.5-2B, thinking on:
#   T0.6 pp0.0  max32k -> 11/23 truncated, 12/23 parsed
#   T0.6 pp0.0  max60k -> 13/23 truncated  (more budget is WORSE: it is a loop)
#   T0.6 pp1.5  max32k ->  1/23 truncated, 22/23 parsed
#   T0.7 pp1.5  max32k ->  0/23 truncated, 23/23 parsed, longest reply 11894 tokens
# This is why the max_tokens advice in the runner's own warning is a red herring here.
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1

# probe | runner | config | the results file written on success
PROBES=(
    "feasibility|run_refonbench_feasibility.py|cfg/eval_refonbench_feasibility.yaml|feasibility_results_incremental.json"
    "nav|run_refonbench_feasibility_nav.py|cfg/eval_refonbench_feasibility_nav.yaml|feasibility_nav_results.json"
    "long|run_refonbench_feasibility.py|cfg/eval_refonbench_feasibility_long.yaml|feasibility_results_incremental.json"
)
declare -A EXP=(
    [feasibility]="exp_feasibility_refonbench"
    [nav]="exp_feasibility_nav_refonbench"
    [long]="exp_feasibility_long_refonbench"
)

SERVER_PID=""

stop_server() {
    [ -z "$SERVER_PID" ] && return 0
    echo "[$(date +%H:%M:%S)] stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 60); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    sleep 10   # let the driver hand the VRAM back before the next model asks for it
}
trap 'stop_server; exit 130' INT TERM

free_gpu() {
    # ollama keeps the last model resident for five minutes. That is 4-6 GB the fp8
    # weights and their KV cache want. `ollama stop` unloads it without stopping the
    # server, so nothing about the ollama setup is lost.
    command -v ollama >/dev/null || return 0
    ollama ps 2>/dev/null | awk 'NR>1 {print $1}' | while read -r m; do
        [ -n "$m" ] && echo "  unloading ollama model $m" && ollama stop "$m"
    done
    sleep 5
}

wait_for_server() {
    local waited=0
    while [ "$waited" -lt "$SERVER_WAIT" ]; do
        if curl -sf -m 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "  server up after ${waited}s"
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  SERVER DIED -- see the log"; return 1; }
        sleep 10
        waited=$((waited + 10))
    done
    echo "  server did not become ready within ${SERVER_WAIT}s"
    return 1
}

slug_of() {  # Qwen/Qwen3.5-9B -> qwen3_5_9b_vllm, the tag model_slug() builds
    VLM_PROVIDER=vllm VLLM_MODEL="$1" "$PY" -c \
        "from run_refonbench_feasibility import model_slug; print(model_slug())"
}

echo "================================================================"
echo " vLLM feasibility sweep: ${#MODELS[@]} model(s) x ${#PROBES[@]} probes"
echo " thinking ON, fp8, max-model-len $MAX_MODEL_LEN, workers $WORKERS, gpu-util $GPU_UTIL"
echo " temperature $TEMPERATURE, presence_penalty $PRESENCE_PENALTY (the loop brake)"
[ "$SMOKE" -gt 0 ] && echo " SMOKE MODE: --episodes-per-scene $SMOKE"
echo " logs -> $LOGDIR"
echo "================================================================"

EXTRA=()
[ "$SMOKE" -gt 0 ] && EXTRA=(--episodes-per-scene "$SMOKE")

FAILED=()
for MODEL in "${MODELS[@]}"; do
    SERVED="$(basename "$MODEL")"
    SLUG="$(slug_of "$MODEL")"
    echo
    echo "=== $MODEL  (served as $SERVED, tag $SLUG) ==="

    # nothing to serve if all three probes already have results
    todo=0
    for spec in "${PROBES[@]}"; do
        IFS='|' read -r name _ _ resfile <<< "$spec"
        [ -f "results/${EXP[$name]}_${SLUG}/$resfile" ] || todo=$((todo + 1))
    done
    if [ "$todo" -eq 0 ]; then
        echo "  all three probes already have results, skipping"
        continue
    fi

    free_gpu
    SERVER_LOG="$LOGDIR/server_${SLUG}.log"
    echo "[$(date +%H:%M:%S)] starting vLLM -> $SERVER_LOG"
    "$VLLM" serve "$MODEL" \
        --served-model-name "$SERVED" \
        --quantization fp8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --reasoning-parser qwen3 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --port "$PORT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        FAILED+=("$SLUG: server never came up")
        tail -40 "$SERVER_LOG"
        stop_server
        continue
    fi

    for spec in "${PROBES[@]}"; do
        IFS='|' read -r name runner cfgfile resfile <<< "$spec"
        outdir="results/${EXP[$name]}_${SLUG}"
        if [ -f "$outdir/$resfile" ]; then
            echo "  [skip] $name -- $outdir already has results"
            continue
        fi
        runlog="$LOGDIR/run_${name}_${SLUG}.log"
        echo "[$(date +%H:%M:%S)] $name -> $outdir  (log: $runlog)"
        start=$(date +%s)
        # no --reasoning-effort: thinking stays on
        VLM_PROVIDER=vllm VLLM_MODEL="$SERVED" VLLM_TIMEOUT=1800 \
            "$PY" "$runner" -cf "$cfgfile" --workers "$WORKERS" \
            --temperature "$TEMPERATURE" --presence-penalty "$PRESENCE_PENALTY" \
            "${EXTRA[@]}" \
            > "$runlog" 2>&1
        rc=$?
        mins=$(( ($(date +%s) - start) / 60 ))
        if [ $rc -ne 0 ] || [ ! -f "$outdir/$resfile" ]; then
            echo "  FAILED ($name, $SLUG, rc=$rc, ${mins}m) -- tail:"
            tail -20 "$runlog"
            FAILED+=("$SLUG/$name (rc=$rc)")
        else
            echo "  done in ${mins}m"
        fi
    done

    stop_server
done

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo " all runs finished"
else
    echo " ${#FAILED[@]} run(s) failed:"
    printf '   - %s\n' "${FAILED[@]}"
fi
echo "================================================================"
ls -d results/exp_feasibility*_vllm 2>/dev/null
