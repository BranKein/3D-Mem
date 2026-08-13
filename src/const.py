import os

# about habitat scene
INVALID_SCENE_ID = []

# which VLM backend to use: "openai", "ollama", "vllm" or "anthropic"
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "ollama")

# about chatgpt api (also works for any OpenAI-compatible endpoint)
END_POINT = os.environ.get("OPENAI_END_POINT", "")
OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# about ollama, used when VLM_PROVIDER == "ollama"
# note that the context length is NOT set here: it is a property of the ollama
# server (the OLLAMA_CONTEXT_LENGTH environment variable of `ollama serve`),
# and the prompts contain many images, so it needs to be large
OLLAMA_END_POINT = os.environ.get("OLLAMA_END_POINT", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 600))

# about vllm, used when VLM_PROVIDER == "vllm". The model is a HuggingFace repo id
# ("Qwen/Qwen3.5-9B") unless the server was started with --served-model-name.
#
# As with ollama, the context length is a server setting rather than a per-request one,
# but the flag bounds prompt AND completion together: --max-model-len has to cover
# cfg.max_tokens (32768 in the feasibility configs) on top of the prompt, or vLLM
# rejects the request. See scripts/vllm_setup.sh.
VLLM_END_POINT = os.environ.get("VLLM_END_POINT", "http://localhost:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", 600))

# Sampling defaults for callers that have no config file to read -- the navigation
# evaluations build their client at import time (src/eval_utils_gpt_*.py) and cannot
# reach cfg. They matter most with thinking on: the built-in max_tokens of 4096 is
# below what a thinking reply needs (measured median ~4900 on Qwen3.5-2B), and
# presence_penalty 0 lets the model loop instead of answering. An explicit argument
# from the caller always wins over these.
VLM_MAX_TOKENS = os.environ.get("VLM_MAX_TOKENS")
VLM_TEMPERATURE = os.environ.get("VLM_TEMPERATURE")
VLM_PRESENCE_PENALTY = os.environ.get("VLM_PRESENCE_PENALTY")
VLM_REASONING_EFFORT = os.environ.get("VLM_REASONING_EFFORT")

# about the Anthropic API, used when VLM_PROVIDER == "anthropic". The SDK resolves the
# key itself from ANTHROPIC_API_KEY (or an `ant auth login` profile), so there is no key
# setting here -- unlike OPENAI_KEY, which this repo passes explicitly.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
ANTHROPIC_TIMEOUT = float(os.environ.get("ANTHROPIC_TIMEOUT", 600))
