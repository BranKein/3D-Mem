#!/bin/bash
# Launch the GOAT-Bench val_unseen evaluation in a detached tmux session.
#
# Run from anywhere; the repo root is located from this script's own path:
#   ./scripts/run_goatbench_tmux.sh                              # qwen2.5vl:7b
#   ./scripts/run_goatbench_tmux.sh qwen3-vl:30b-a3b-instruct    # another model
#   ./scripts/run_goatbench_tmux.sh qwen2.5vl:7b --clean         # wipe results/
#
# The conda env is found automatically; override with PYTHON=/path/to/python.
#
# Attach with:  tmux attach -t goat_<model>
# Follow log:   tail -f goatbench_<model>.log

set -uo pipefail

# --------------------------------------------------------------------------- #
# locate the repo, whatever directory this script was moved to
# --------------------------------------------------------------------------- #
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="$(dirname "$SCRIPT_PATH")"
while [ ! -f "$REPO_DIR/run_goatbench_evaluation.py" ] && [ "$REPO_DIR" != "/" ]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
if [ ! -f "$REPO_DIR/run_goatbench_evaluation.py" ]; then
    echo "ERROR: could not find run_goatbench_evaluation.py above $SCRIPT_PATH"
    exit 1
fi

CONFIG="cfg/eval_goatbench.yaml"

# The conda env lives in a different place on each machine, so probe for it.
# ENV_PREFIX below is derived from this, so it has to be right.
if [ -z "${PYTHON:-}" ]; then
    for p in "$HOME/.conda/envs/3dmem/bin/python" \
             "$HOME/anaconda3/envs/3dmem/bin/python" \
             "$HOME/miniconda3/envs/3dmem/bin/python"; do
        if [ -x "$p" ]; then PYTHON="$p"; break; fi
    done
fi
PYTHON="${PYTHON:-}"

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

    # tmux starts a fresh shell with no conda env activated, so $CONDA_PREFIX is
    # empty in here -- derive the prefix from $PYTHON instead of relying on
    # `conda activate` having run. Without this the system libstdc++ is used and
    # matplotlib fails on a missing GLIBCXX_3.4.29.
    ENV_PREFIX="$(dirname "$(dirname "$PYTHON")")"
    export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    # The conda env ships its own glvnd, so habitat's bindings resolve
    # libGLdispatch there while libEGL/libOpenGL come from the system. The two
    # dispatch tables do not match and GL entry points end up unpopulated:
    # habitat then aborts with "cannot retrieve OpenGL version: InvalidValue".
    # Preloading the system dispatch table wins over the baked-in RPATH.
    GLDISPATCH=/lib/x86_64-linux-gnu/libGLdispatch.so.0
    if [ -e "$GLDISPATCH" ]; then
        export LD_PRELOAD="$GLDISPATCH${LD_PRELOAD:+:$LD_PRELOAD}"
    fi

    {
        echo "=== started $(date -Is) | model=$OLLAMA_MODEL | provider=$VLM_PROVIDER ==="
        echo "    REPO_DIR        = $REPO_DIR"
        echo "    PYTHON          = $PYTHON"
        echo "    LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
        echo "    LD_PRELOAD      = ${LD_PRELOAD:-(unset)}"
    } | tee "$LOG"

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

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: could not find the '3dmem' conda env python."
    echo "       Set it explicitly, e.g. PYTHON=~/.conda/envs/3dmem/bin/python $0 $MODEL"
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
    "'$SCRIPT_PATH' --inner '$MODEL' '$LOG'; echo; echo '[run ended -- session kept open]'; exec bash"

echo
echo "Started GOAT-Bench val_unseen in tmux session '$SESSION'."
echo "  attach : tmux attach -t $SESSION"
echo "  log    : tail -f $LOG"
echo "  stop   : tmux kill-session -t $SESSION"
