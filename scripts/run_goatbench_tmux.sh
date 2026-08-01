#!/bin/bash
# Launch the GOAT-Bench val_unseen evaluation in a detached tmux session.
#
# Usage:
#   ./run_goatbench_tmux.sh                                # qwen2.5vl:7b
#   ./run_goatbench_tmux.sh qwen3-vl:8b-instruct-q8_0      # another ollama model
#   ./run_goatbench_tmux.sh qwen2.5vl:7b --clean           # wipe results/ first
#
# Attach with:  tmux attach -t goat_<model>
# Follow log:   tail -f goatbench_<model>.log

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HOME/anaconda3/envs/3dmem/bin/python"
CONFIG="cfg/eval_goatbench.yaml"

# The prompts carry ~20 images each, so a step can take a while on a big model.
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-1200}"
export VLM_PROVIDER="${VLM_PROVIDER:-ollama}"

# --------------------------------------------------------------------------- #
# inner mode: what actually runs inside the tmux session
# --------------------------------------------------------------------------- #
if [ "${1:-}" = "--inner" ]; then
    export OLLAMA_MODEL="$2"
    LOG="$3"
    cd "$REPO_DIR"

    echo "=== started $(date -Is) | model=$OLLAMA_MODEL | provider=$VLM_PROVIDER ===" | tee "$LOG"
    "$PYTHON" -u run_goatbench_evaluation.py -cf "$CONFIG" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    echo "=== finished $(date -Is) | exit=$status ===" | tee -a "$LOG"
    exit "$status"
fi

# --------------------------------------------------------------------------- #
# outer mode: preflight, then spawn the session
# --------------------------------------------------------------------------- #
MODEL="${1:-qwen2.5vl:7b}"
CLEAN="${2:-}"

# tmux session and log names are derived from the model, so two models can run
# one after another without overwriting each other's log. '.' is replaced too:
# tmux rewrites dots in session names, and the has-session check below would
# then never match the session it just created.
SLUG="$(echo "$MODEL" | tr ':/.' '___')"
SESSION="goat_${SLUG}"
LOG="$REPO_DIR/goatbench_${SLUG}.log"

cd "$REPO_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Attach with:"
    echo "  tmux attach -t $SESSION"
    echo "Or kill it with: tmux kill-session -t $SESSION"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found at $PYTHON (expected the '3dmem' conda env)"
    exit 1
fi

if [ "$VLM_PROVIDER" = "ollama" ]; then
    if ! curl -sf localhost:11434/api/tags >/dev/null; then
        echo "ERROR: ollama server is not reachable at localhost:11434"
        exit 1
    fi
    if ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
        echo "ERROR: '$MODEL' is not pulled. Run: ollama pull $MODEL"
        exit 1
    fi

    # The eval loads habitat-sim + YOLO-World + SAM on the same GPU (~11GB).
    # A model that leaves less than that free will OOM once SAM starts.
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    model_gb=$(ollama list | awk -v m="$MODEL" '$1==m {print $3}')
    echo "GPU free: ${free_mb} MiB | model: ${MODEL} (${model_gb}GB)"
    if [ "$free_mb" -lt 17000 ]; then
        echo "WARNING: less than ~17GB free. The VLM plus the perception stack"
        echo "         (habitat + YOLO-World + SAM, ~11GB) may not fit."
    fi
fi

if [ "$CLEAN" = "--clean" ]; then
    if [ -d results ]; then
        echo "Removing $(du -sh results | cut -f1) of previous results/ ..."
        rm -rf results
    fi
fi

tmux new-session -d -s "$SESSION" -c "$REPO_DIR" \
    "'$REPO_DIR/$(basename "${BASH_SOURCE[0]}")' --inner '$MODEL' '$LOG'; echo; echo '[run ended -- session kept open]'; exec bash"

echo
echo "Started GOAT-Bench val_unseen in tmux session '$SESSION'."
echo "  attach : tmux attach -t $SESSION"
echo "  log    : tail -f $LOG"
echo "  stop   : tmux kill-session -t $SESSION"
