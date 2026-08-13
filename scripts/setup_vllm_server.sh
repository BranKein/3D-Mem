#!/bin/bash
#
# Prepare a GPU server to run the 3D-Mem RefON navigation evaluation against vLLM.
#
#   ./setup_vllm_server.sh              # build the env, fetch weights, check the GPUs
#   ./setup_vllm_server.sh --check      # re-run the checks only, install nothing
#
# Run it once. It does NOT start a server and does NOT run an evaluation -- see
# run_nav_vllm.sh for that.
#
# What it assumes, because you said so: HM3D data is present, habitat already works
# in the `3dmem` env, the server has internet, and git pull works.
#
# What it does NOT touch: the `3dmem` environment. That one is pinned to python 3.9 /
# torch 2.0.1+cu118 / pytorch3d-pyt201 for habitat, and vLLM needs much newer. They do
# not need to share an environment -- this repo talks to vLLM over HTTP only, so
# `3dmem` needs nothing beyond the `openai` package it already has.
#
set -uo pipefail

# The conda *root*, not the binary -- but accept either, and fall back to asking
# conda itself, which is right far more often than a guess at $HOME.
CONDA_BASE="${CONDA_BASE:-}"
case "$CONDA_BASE" in
    */bin/conda) CONDA_BASE="${CONDA_BASE%/bin/conda}" ;;
    */condabin/conda) CONDA_BASE="${CONDA_BASE%/condabin/conda}" ;;
esac
if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    [ -z "$CONDA_BASE" ] && CONDA_BASE="$HOME/anaconda3"
fi
VLLM_ENV="${VLLM_ENV:-vllm}"
MODELS=("Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B")
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

say() { printf '\n=== %s\n' "$*"; }
warn() { printf '  !! %s\n' "$*"; }

# --------------------------------------------------------------------------- #
say "0. what is already here"
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi -- wrong machine?"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "  $N_GPU GPU(s)"
if [ "$N_GPU" -lt 2 ]; then
    warn "only one GPU. vLLM and the perception stack (YOLO-World + SAM + CLIP, which"
    warn "src/scene_goatbench.py pins to 'cuda' with no way to move it) will have to"
    warn "share it, and you must leave the perception stack room -- see run_nav_vllm.sh."
fi

# fp8 needs Ada (8.9) or newer for the tensor cores. vLLM accepts it further back
# (get_min_capability is 7.5) but falls through to the Marlin path, which stores the
# weights in 8 bits and converts back to bf16 to multiply: the memory is saved, the
# speed is not. On an A30 (8.0) with memory to spare, bf16 is the better trade.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ "${CAP:-0}" -ge 89 ]; then
    echo "  compute capability supports native fp8 -- QUANT=fp8 is worth it if memory is tight"
else
    echo "  compute capability ${CAP:-?} has no fp8 tensor cores -- use bf16 (the default)"
fi

if [ ! -d "$CONDA_BASE" ]; then
    echo "conda root not found at $CONDA_BASE"
    echo "  pass the root directory, not the binary:  CONDA_BASE=\$(conda info --base) $0"
    exit 1
fi
echo "  conda root: $CONDA_BASE"
if [ ! -w "$CONDA_BASE/envs" ]; then
    warn "$CONDA_BASE/envs is not writable. Either run with write access, or put the"
    warn "env somewhere you own and point the run script at it:"
    warn "  conda create -y -p \$HOME/envs/vllm python=3.12   (then VLLM=\$HOME/envs/vllm/bin/vllm)"
fi
if [ -x "$CONDA_BASE/envs/3dmem/bin/python" ]; then
    echo -n "  3dmem env: "; "$CONDA_BASE/envs/3dmem/bin/python" -c \
        "import sys,openai;print('python',sys.version.split()[0],'openai',openai.__version__)" \
        2>/dev/null || warn "3dmem exists but has no openai package -- pip install openai in it"
else
    warn "no 3dmem env at $CONDA_BASE/envs/3dmem -- this script does not build it"
fi

if [ "$CHECK_ONLY" = 1 ]; then
    say "check only, stopping here"
    exit 0
fi

# --------------------------------------------------------------------------- #
say "1. vLLM environment (separate from 3dmem, on purpose)"
if [ -x "$CONDA_BASE/envs/$VLLM_ENV/bin/python" ]; then
    echo "  $VLLM_ENV already exists, skipping creation"
else
    "$CONDA_BASE/bin/conda" create -y -n "$VLLM_ENV" python=3.12 || exit 1
fi
PY_VLLM="$CONDA_BASE/envs/$VLLM_ENV/bin/python"
"$PY_VLLM" -m pip install --upgrade pip
"$PY_VLLM" -m pip install vllm || { echo "vllm install failed"; exit 1; }
"$PY_VLLM" -c "
import vllm, torch
print('  vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)
print('  cuda available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())
"

say "2. Qwen3.5 support in this vLLM build"
"$PY_VLLM" -c "
from vllm.model_executor.models.registry import ModelRegistry
a = ModelRegistry.get_supported_archs()
need = ['Qwen3_5ForConditionalGeneration', 'Qwen3_5ForCausalLM']
for n in need:
    print(('  OK   ' if n in a else '  MISS ') + n)
if not any(n in a for n in need):
    raise SystemExit('this vLLM is too old for Qwen3.5 -- upgrade it')
"

say "3. model weights"
# The navigation prompts carry images, so these have to be the vision checkpoints --
# Qwen3.5 dense models are Qwen3_5ForConditionalGeneration and carry a vision_config,
# so the plain repo id is already the right one.
"$PY_VLLM" -m pip install -q "huggingface_hub[cli]"
for m in "${MODELS[@]}"; do
    echo "  fetching $m"
    "$CONDA_BASE/envs/$VLLM_ENV/bin/hf" download "$m" >/dev/null || \
        warn "download failed for $m"
done
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null

say "4. repo code"
echo "  the vLLM backend lives in src/vlm_client.py; make sure this checkout has it:"
echo "    git -C <repo> pull"
echo "    grep -q 'class VllmClient' <repo>/src/vlm_client.py && echo OK"

say "done"
cat <<'NEXT'

Next, in this order:

  1. Smoke test with ONE episode before anything long. Four separate silent failures
     showed up doing this on the other machine, each of which would have produced a
     full run of 0.0%:
       - the server refusing to start because something else held VRAM
       - thinking off by default, so --reasoning-parser ate every answer
       - temperature 0.0 sending the model into a repetition loop
       - presence_penalty 0.0 letting it loop anyway
     run_nav_vllm.sh sets all four correctly and takes SMOKE=1.

  2. Read results/<exp>/ and confirm the answers parse before scaling up.

NEXT
