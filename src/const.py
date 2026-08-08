import os

# about habitat scene
INVALID_SCENE_ID = []

# which VLM backend to use for the evaluation: "openai", "ollama" or "anthropic"
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

# about the Anthropic API, used when VLM_PROVIDER == "anthropic". The SDK resolves the
# key itself from ANTHROPIC_API_KEY (or an `ant auth login` profile), so there is no key
# setting here -- unlike OPENAI_KEY, which this repo passes explicitly.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
ANTHROPIC_TIMEOUT = float(os.environ.get("ANTHROPIC_TIMEOUT", 600))
