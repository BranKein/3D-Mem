"""Provider-agnostic VLM client used by the evaluation scripts.

The evaluation code builds prompts as a list of "content" tuples, where each
tuple is either ``(text,)`` or ``(text, base64_png_image)``.  A client takes
such a list plus a system prompt and returns the model response as a string.

Two backends are provided:
    - "openai": the OpenAI API or any OpenAI-compatible endpoint (GPT-4o, ...)
    - "ollama": a local ollama server, through its OpenAI-compatible /v1 API

The backend and its settings are read from src/const.py, which in turn can be
overridden with environment variables (see that file).
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from src.const import (
    END_POINT,
    OLLAMA_END_POINT,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OPENAI_KEY,
    OPENAI_MODEL,
    VLM_PROVIDER,
)

logger = logging.getLogger(__name__)

Content = Sequence[Tuple]


class VLMClient(ABC):
    """Common retry logic and interface for all backends."""

    def __init__(
        self,
        model: str,
        max_tries: int = 5,
        rate_limit_wait: float = 30,
        error_wait: float = 60,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
    ):
        self.model = model
        self.max_tries = max_tries
        self.rate_limit_wait = rate_limit_wait
        self.error_wait = error_wait
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def call(self, sys_prompt: str, contents: Content) -> Optional[str]:
        """Query the model, retrying on failure. Returns None if all tries fail."""
        retry_count = 0
        while retry_count < self.max_tries:
            try:
                return self._chat(sys_prompt, contents)
            except Exception as e:
                wait = self.error_wait
                if self._is_rate_limit_error(e):
                    print(f"Rate limit error, waiting for {self.rate_limit_wait}s")
                    wait = self.rate_limit_wait
                else:
                    print("Error: ", e)
                time.sleep(wait)
                retry_count += 1
                continue

        return None

    @abstractmethod
    def _chat(self, sys_prompt: str, contents: Content) -> str:
        """Single (non-retried) request to the model."""

    def _is_rate_limit_error(self, e: Exception) -> bool:
        return False


class OpenAIClient(VLMClient):
    """The OpenAI API, or any OpenAI-compatible chat completion endpoint."""

    def __init__(self, model=None, end_point=None, api_key=None, timeout=None, **kwargs):
        super().__init__(model=model or OPENAI_MODEL, **kwargs)
        import openai
        from openai import OpenAI

        self._openai = openai
        end_point = END_POINT if end_point is None else end_point
        client_kwargs = {
            # an empty base_url would be passed through as-is, so fall back to
            # the SDK default (the official API) when it is not configured
            "base_url": end_point or None,
            "api_key": OPENAI_KEY if api_key is None else api_key,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def format_content(contents: Content) -> List[dict]:
        formated_content = []
        for c in contents:
            formated_content.append({"type": "text", "text": c[0]})
            if len(c) == 2:
                formated_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{c[1]}",
                            "detail": "high",
                        },
                    }
                )
        return formated_content

    def _chat(self, sys_prompt: str, contents: Content) -> str:
        message_text = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": self.format_content(contents)},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=message_text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return completion.choices[0].message.content

    def _is_rate_limit_error(self, e: Exception) -> bool:
        return isinstance(e, self._openai.RateLimitError)


class OllamaClient(OpenAIClient):
    """A local ollama server, through its OpenAI-compatible /v1 API.

    Ollama's native /api/chat endpoint takes the images of a message as a
    separate list instead of interleaving them with the text, and the model
    then cannot tell which image belongs to which "Snapshot i" label (verified
    on ollama 0.32.5 / qwen2.5vl:7b: it mismatches the images). The /v1 API
    keeps text and images interleaved, so it is the one used here.

    Note that /v1 ignores per-request context length: it is set on the server,
    with the OLLAMA_CONTEXT_LENGTH environment variable of `ollama serve`.
    """

    def __init__(self, model=None, end_point=None, timeout=None, **kwargs):
        end_point = OLLAMA_END_POINT if end_point is None else end_point
        super().__init__(
            model=model or OLLAMA_MODEL,
            end_point=f"{end_point.rstrip('/')}/v1",
            api_key="ollama",  # ollama ignores the key, but it must be non-empty
            timeout=OLLAMA_TIMEOUT if timeout is None else timeout,
            **kwargs,
        )
        self._warn_if_model_missing()

    def _warn_if_model_missing(self):
        """Fail loudly at startup instead of after the first prompt."""
        try:
            available = [m.id for m in self.client.models.list().data]
        except Exception as e:
            logger.warning(f"Could not reach the ollama server at {self.client.base_url}: {e}")
            return
        if self.model not in available:
            logger.warning(
                f"Model '{self.model}' is not available in ollama (found: {available}). "
                f"Run `ollama pull {self.model}` first."
            )


BACKENDS = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}


def create_vlm_client(provider: Optional[str] = None, **kwargs) -> VLMClient:
    """Build the VLM client for the configured provider.

    Extra keyword arguments are forwarded to the backend, so a caller can e.g.
    use different retry intervals without changing the global configuration.
    """
    provider = (provider or VLM_PROVIDER).lower()
    if provider not in BACKENDS:
        raise ValueError(
            f"Unknown VLM provider '{provider}', expected one of {sorted(BACKENDS)}"
        )
    client = BACKENDS[provider](**kwargs)
    logger.info(f"Using {provider} VLM backend with model {client.model}")
    return client


if __name__ == "__main__":
    # smoke test: python -m src.vlm_client
    logging.basicConfig(level=logging.INFO)
    client = create_vlm_client(max_tries=1)
    print(
        client.call("You are a helpful assistant.", [("Say 'hello' and nothing else.",)])
    )
