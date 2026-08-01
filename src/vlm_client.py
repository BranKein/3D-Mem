"""Provider-agnostic VLM client used by the evaluation scripts.

The evaluation code builds prompts as a list of "content" tuples, where each
tuple is either ``(text,)`` or ``(text, base64_png_image)``.  A client takes
such a list plus a system prompt and returns the model response as a string.

Two backends are provided:
    - "openai": any OpenAI-compatible chat completion endpoint (GPT-4o, ...)
    - "ollama": a local ollama server (native /api/chat endpoint)

The backend and its settings are read from src/const.py, which in turn can be
overridden with environment variables (see that file).
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from src.const import (
    END_POINT,
    OLLAMA_END_POINT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TIMEOUT,
    OPENAI_KEY,
    OPENAI_MODEL,
    VLM_PROVIDER,
)

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
    """OpenAI (or any OpenAI-compatible) chat completion endpoint."""

    def __init__(self, model=None, end_point=None, api_key=None, **kwargs):
        super().__init__(model=model or OPENAI_MODEL, **kwargs)
        import openai
        from openai import OpenAI

        self._openai = openai
        end_point = END_POINT if end_point is None else end_point
        self.client = OpenAI(
            # an empty base_url would be passed through as-is, so fall back to
            # the SDK default (the official API) when it is not configured
            base_url=end_point or None,
            api_key=OPENAI_KEY if api_key is None else api_key,
        )

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


class OllamaClient(VLMClient):
    """Local ollama server, through its native /api/chat endpoint.

    Ollama takes the images of a message as a separate list of base64 strings
    instead of interleaving them with the text, so the texts are concatenated
    in order and the images are appended in the same order.
    """

    def __init__(
        self,
        model=None,
        end_point=None,
        num_ctx=None,
        timeout=None,
        keep_alive=None,
        **kwargs,
    ):
        super().__init__(model=model or OLLAMA_MODEL, **kwargs)
        import requests

        self._requests = requests
        self.end_point = (OLLAMA_END_POINT if end_point is None else end_point).rstrip(
            "/"
        )
        self.num_ctx = OLLAMA_NUM_CTX if num_ctx is None else num_ctx
        self.timeout = OLLAMA_TIMEOUT if timeout is None else timeout
        self.keep_alive = OLLAMA_KEEP_ALIVE if keep_alive is None else keep_alive

    @staticmethod
    def format_content(contents: Content) -> Tuple[str, List[str]]:
        text, images = "", []
        for c in contents:
            text += c[0]
            if len(c) == 2:
                images.append(c[1])
        return text, images

    def _chat(self, sys_prompt: str, contents: Content) -> str:
        text, images = self.format_content(contents)
        user_message = {"role": "user", "content": text}
        if images:
            user_message["images"] = images

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                user_message,
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        response = self._requests.post(
            f"{self.end_point}/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        if "message" not in result:
            raise RuntimeError(f"Unexpected ollama response: {json.dumps(result)[:500]}")
        return result["message"]["content"]


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
    logging.info(f"Using {provider} VLM backend with model {client.model}")
    return client


if __name__ == "__main__":
    # smoke test: python -m src.vlm_client
    logging.basicConfig(level=logging.INFO)
    client = create_vlm_client(max_tries=1)
    print(client.call("You are a helpful assistant.", [("Say 'hello' and nothing else.",)]))
