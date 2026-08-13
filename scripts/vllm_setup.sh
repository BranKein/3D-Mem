#!/bin/bash
#
# Start a vLLM server for the RefON feasibility probes.
#
#   scripts/vllm_setup.sh                     # Qwen/Qwen3.5-9B, fp8
#   scripts/vllm_setup.sh Qwen/Qwen3.5-4B     # another model, same settings
#
# vLLM does NOT belong in the 3dmem conda environment: that one is pinned to
# python 3.9 / torch 2.0.1+cu118 / pytorch3d-pyt201 for habitat, and vLLM wants a
# newer python and torch. It does not need to share an environment either -- the
# repo only ever talks to it over HTTP -- so install it in its own env or container
# and leave 3dmem alone.
#
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-9B}"

# Bounds prompt + completion together, so it has to cover cfg.max_tokens (32768)
# on top of the prompt. The probes are text-only, which does not help: the
# completion budget is what dominates, and with thinking on the model does use it.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"

# fp8 halves the weights (9.7B: 18.1 GiB -> 9.0 GiB) and leaves room for the KV
# cache on a 24 GiB card. vLLM converts a bf16 checkpoint at load time, so no
# separate quantised model is needed -- which matters here, because Qwen ships
# prebuilt FP8 weights only from 27B up. The RTX 4090 (Ada, sm89) runs fp8 natively.
# Set QUANTIZATION= (empty) to serve bf16 instead; 2B and 4B fit either way.
QUANTIZATION="${QUANTIZATION:-fp8}"

# Splits the chain of thought into `reasoning_content` and leaves the answer alone
# in `content`. Without it the reply is `<think>...</think>` glued to the JSON and
# the runners score every answer as parse_failed. VllmClient strips the tags as a
# fallback and warns, but a reply cut off mid-thought has no closing tag.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

args=(
    --max-model-len "$MAX_MODEL_LEN"
    --reasoning-parser "$REASONING_PARSER"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    # keep the tag in results/ lined up with the ollama runs: without this the model
    # is named by its full repo id and the directory becomes ..._qwen_qwen3_5_9b_vllm
    --served-model-name "$(basename "$MODEL")"
)
[ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")

echo "serving $MODEL"
echo "  quantization   : ${QUANTIZATION:-bf16 (none)}"
echo "  max-model-len  : $MAX_MODEL_LEN"
echo "  reasoning      : on, parsed by '$REASONING_PARSER'"
echo
echo "then, in the 3dmem environment:"
echo "  VLM_PROVIDER=vllm VLLM_MODEL=$(basename "$MODEL") \\"
echo "      python run_refonbench_feasibility.py -cf cfg/eval_refonbench_feasibility.yaml --workers 8"
echo

exec vllm serve "$MODEL" "${args[@]}"
