import os

# about habitat scene
INVALID_SCENE_ID = []

# which VLM backend to use for the evaluation: "openai" or "ollama"
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "openai")

# about chatgpt api (also works for any OpenAI-compatible endpoint)
END_POINT = os.environ.get("OPENAI_END_POINT", "")
OPENAI_KEY = os.environ.get("OPENAI_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# about ollama, used when VLM_PROVIDER == "ollama"
OLLAMA_END_POINT = os.environ.get("OLLAMA_END_POINT", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
# the prompts contain many images, so the default context length is far too
# small: anything beyond it is silently truncated by ollama
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 32768))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 600))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
