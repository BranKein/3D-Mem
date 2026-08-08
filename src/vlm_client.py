"""Provider-agnostic VLM client used by the evaluation scripts.

The evaluation code builds prompts as a list of "content" tuples, where each
tuple is either ``(text,)`` or ``(text, base64_png_image)``.  A client takes
such a list plus a system prompt and returns the model response as a string.

Three backends are provided:
    - "openai": the OpenAI API or any OpenAI-compatible endpoint (GPT-4o, ...)
    - "ollama": a local ollama server, through its OpenAI-compatible /v1 API
    - "anthropic": the Anthropic API, through the official SDK

The backend and its settings are read from src/const.py, which in turn can be
overridden with environment variables (see that file).
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from src.const import (
    ANTHROPIC_MODEL,
    ANTHROPIC_TIMEOUT,
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
        self.consecutive_length_stops = 0
        #: replies declined by a provider policy classifier (Anthropic `stop_reason`)
        self.refusals = 0
        self._length_lock = threading.Lock()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def note_length_stop(self):
        """A reply cut off by the token budget. Counted consecutively, on purpose.

        The question itself is a failure either way. What the streak decides is whether
        the *model* is worth continuing with, and only an unbroken run means "this model
        cannot answer this task" -- which is what it looked like when qwen3.5:2b was cut
        off on every single call. Counting cumulatively instead threw away two otherwise
        healthy runs over a 0.3% blip rate: 5 stops in 1679 answered queries.
        """
        with self._length_lock:
            self.length_stops += 1
            self.consecutive_length_stops += 1
            return self.consecutive_length_stops

    def note_good_reply(self):
        with self._length_lock:
            self.consecutive_length_stops = 0

    @property
    def gave_up(self) -> bool:
        """True once the model was cut off `max_length_stops` times in a row."""
        return (
            bool(self.max_length_stops)
            and self.consecutive_length_stops >= self.max_length_stops
        )

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
        self.note_good_reply()
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


class AnthropicClient(VLMClient):
    """The Anthropic API, through the official SDK.

    Not an OpenAI-compatible shim: the request shape differs enough that reusing
    OpenAIClient would misreport the two things this repo's runners depend on --
    a reply cut off by the token budget, and an empty body.
    """

    def __init__(self, model=None, timeout=None, effort=None, **kwargs):
        super().__init__(model=model or ANTHROPIC_MODEL, **kwargs)
        import anthropic

        self._anthropic = anthropic
        # No api_key argument: the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
        # or an `ant auth login` profile on its own, and passing an empty string would
        # shadow all three.
        self.client = anthropic.Anthropic(
            timeout=ANTHROPIC_TIMEOUT if timeout is None else timeout
        )
        #: output_config.effort, when the model supports it. Haiku 4.5 rejects it.
        self.effort = effort

    @staticmethod
    def format_content(contents: Content) -> List[dict]:
        blocks = []
        for c in contents:
            blocks.append({"type": "text", "text": c[0]})
            if len(c) == 2:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": c[1],
                        },
                    }
                )
        return blocks

    def _chat(self, sys_prompt: str, contents: Content) -> str:
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": self.format_content(contents)}],
        }
        # temperature is accepted on Haiku 4.5 and the other pre-4.6 models, and
        # rejected outright on Opus 4.7+/Sonnet 5 -- send it only when it would be
        # honoured, so the same client works across both.
        if not self._drops_sampling_params():
            params["temperature"] = self.temperature
            params["top_p"] = self.top_p
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.reasoning_effort == "none":
            # the cross-provider "turn thinking off" switch; on models where thinking
            # is on by default this is what keeps the run cheap
            params["thinking"] = {"type": "disabled"}

        # Stream: the SDK refuses non-streaming requests whose max_tokens implies a
        # long generation, and these configs run at 32768.
        with self.client.messages.stream(**params) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "max_tokens":
            self.note_length_stop()
            return None
        if message.stop_reason == "refusal":
            # A policy decline, not an outage. Return None rather than "" so it skips
            # the empty-body retry -- the same prompt gets the same decline, and five
            # 2s retries per refused question would only slow the run down. It lands in
            # the records as an unanswered question, same as any other empty body.
            self.refusals += 1
            print(f"Refusal from {self.model} (stop_reason=refusal); scoring as no answer")
            self.note_good_reply()
            return None
        self.note_good_reply()
        return "".join(b.text for b in message.content if b.type == "text")

    def _drops_sampling_params(self) -> bool:
        m = self.model
        return any(
            tag in m
            for tag in ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5", "mythos-5")
        )

    def _is_rate_limit_error(self, e: Exception) -> bool:
        return isinstance(e, self._anthropic.RateLimitError)


BACKENDS = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
    "anthropic": AnthropicClient,
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
