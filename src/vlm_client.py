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
import threading
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
        empty_response_wait: float = 2,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        reasoning_effort: Optional[str] = None,
        max_length_stops: int = 5,
    ):
        self.model = model
        self.max_tries = max_tries
        self.rate_limit_wait = rate_limit_wait
        self.error_wait = error_wait
        self.empty_response_wait = empty_response_wait
        self.reasoning_effort = reasoning_effort
        #: give up on a model once this many replies were cut off by the token budget
        self.max_length_stops = max_length_stops
        self.length_stops = 0
        self._length_lock = threading.Lock()

    def note_length_stop(self):
        with self._length_lock:
            self.length_stops += 1
            return self.length_stops

    @property
    def gave_up(self) -> bool:
        """True once the model has been cut off `max_length_stops` times."""
        return self.max_length_stops and self.length_stops >= self.max_length_stops
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def call(self, sys_prompt: str, contents: Content) -> Optional[str]:
        """Query the model, retrying on failure. Returns None if all tries fail."""
        retry_count = 0
        while retry_count < self.max_tries:
            try:
                response = self._chat(sys_prompt, contents)
                # An empty body is not an answer. A reasoning model that spends its whole
                # token budget thinking returns content="" with no exception raised, and
                # the caller can only score that as a wrong answer -- so retry it like
                # any other failed try instead of passing the blank through. Observed
                # sporadically with gemma4:26b (12 of 200 prompts, unrelated to prompt
                # length); a plain retry is usually enough.
                if response is not None and not str(response).strip():
                    print("Empty response from the model, retrying")
                    time.sleep(self.empty_response_wait)
                    retry_count += 1
                    continue
                return response
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
        extra = {}
        if self.reasoning_effort is not None:
            # A reasoning model can spend the whole budget thinking and return empty
            # content. qwen3.5:0.8b did exactly that on this task: 16384 completion
            # tokens, 66k characters of `reasoning`, content "", 240s per call. With
            # reasoning_effort="none" the same prompt answers in 0.5s and 46 tokens.
            extra["reasoning_effort"] = self.reasoning_effort
        completion = self.client.chat.completions.create(
            model=self.model,
            extra_body=extra or None,
            messages=message_text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            # Ran out of budget mid-answer. Retrying is pointless -- the same prompt
            # gets the same truncation -- so this is a failed question, and a model that
            # keeps doing it is not answering this task at all.
            self.note_length_stop()
            return None
        return choice.message.content

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
