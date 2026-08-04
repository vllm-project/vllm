# SPDX-License-Identifier: Apache-2.0
"""Async streaming request sender for ``lmcache bench engine``."""

# Standard
from collections.abc import Callable
import collections.abc
import os
import time

# Third Party
from openai import AsyncOpenAI

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import RequestResult
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Callback signature: (result, response_text) -> None
OnFinishedCallback = Callable[[RequestResult, str], None]


def _normalize_url(engine_url: str) -> str:
    """Ensure *engine_url* has a scheme and ends with ``/v1``."""
    url = engine_url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _extract_content(chunk: object, completions_mode: bool) -> str:
    """Return text content from a streaming chunk, or ``""`` if none.

    Ported from Tensormesh-Benchmark ``streaming_utils.py``.
    """
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""

    choice = choices[0]

    if completions_mode:
        text = getattr(choice, "text", None)
        return text if text is not None else ""

    # Chat mode: delta.content, with fallback for reasoning_content
    delta = getattr(choice, "delta", None)
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if content is not None:
        return content
    # Fallback for reasoning models
    for attr in ("reasoning_content", "reasoning"):
        fallback = getattr(delta, attr, None)
        if fallback is not None:
            return fallback
    return ""


class RequestSender:
    """Async streaming request sender for inference engines.

    Each ``send_request`` call is a self-contained coroutine.
    Concurrency is controlled externally by the workload module.
    """

    def __init__(
        self,
        engine_url: str,
        model: str,
        completions_mode: bool = False,
        on_finished: list[OnFinishedCallback] = [],  # noqa: B006
        ignore_eos: bool = False,
    ) -> None:
        self._model = model
        self._completions_mode = completions_mode
        self._on_finished = list(on_finished)
        self._ignore_eos = ignore_eos

        base_url = _normalize_url(engine_url)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = "sk-dummy"
            logger.debug("API key source: default dummy key")
        else:
            logger.debug("API key source: OPENAI_API_KEY env var")

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=None,
        )

    def add_on_finished_callback(self, callback: OnFinishedCallback) -> None:
        """Register a callback to be invoked when a request finishes."""
        self._on_finished.append(callback)

    async def send_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
    ) -> RequestResult:
        """Send a single streaming request and return the result.

        Streams the response via SSE, measures TTFT, decode speed, and
        total latency.  Extracts token counts from server usage reports.
        After collecting the result, invokes all registered
        ``on_finished`` callbacks.
        """
        submit_time = time.time()
        first_token_time = 0.0
        first_chunk_time = 0.0
        tokens: list[str] = []
        num_input_tokens = 0
        num_output_tokens = 0

        try:
            response = await self._create_stream(messages, max_tokens)

            async for chunk in response:
                if not first_chunk_time:
                    first_chunk_time = time.time()

                # Extract usage from final chunk
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", 0)
                    ct = getattr(usage, "completion_tokens", 0)
                    if pt:
                        num_input_tokens = pt
                    if ct:
                        num_output_tokens = ct

                content = _extract_content(chunk, self._completions_mode)
                if content:
                    if not first_token_time:
                        first_token_time = time.time()
                    tokens.append(content)

            finish_time = time.time()
            if first_token_time == 0.0 and num_output_tokens > 0:
                # Empty-content stream (common with max_tokens=1, e.g. the
                # single token is EOS): use first chunk arrival as TTFT —
                # closer to engine prefill completion than finish_time.
                first_token_time = first_chunk_time or finish_time
            successful = first_token_time > 0.0
            ttft = (first_token_time - submit_time) if successful else -1.0
            request_latency = finish_time - submit_time
            decode_time = (finish_time - first_token_time) if successful else 0.0
            num_output = num_output_tokens if num_output_tokens > 0 else len(tokens)
            decode_speed = (num_output / decode_time) if decode_time > 0 else 0.0

            result = RequestResult(
                request_id=request_id,
                successful=successful,
                ttft=ttft,
                request_latency=request_latency,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output,
                decode_speed=decode_speed,
                submit_time=submit_time,
                first_token_time=first_token_time,
                finish_time=finish_time,
                error="",
            )
            response_text = "".join(tokens)

        except Exception as e:
            finish_time = time.time()
            result = RequestResult(
                request_id=request_id,
                successful=False,
                ttft=-1.0,
                request_latency=finish_time - submit_time,
                num_input_tokens=0,
                num_output_tokens=0,
                decode_speed=0.0,
                submit_time=submit_time,
                first_token_time=0.0,
                finish_time=finish_time,
                error=str(e),
            )
            response_text = ""
            logger.debug(
                "Request %s failed: %s",
                request_id,
                e,
            )

        for cb in self._on_finished:
            cb(result, response_text)

        return result

    async def send_warmup_request(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1,
    ) -> RequestResult:
        """Send a warmup request (``max_tokens=1`` by default)."""
        return await self.send_request(
            request_id,
            messages,
            max_tokens=max_tokens,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> collections.abc.AsyncIterator:
        """Dispatch the streaming API call (chat or completions).

        When ``ignore_eos`` is set on the sender, ``{"ignore_eos": true}`` is
        added to the request body (a vLLM sampling extension) so generation
        always runs for the full ``max_tokens`` instead of stopping at the
        model's EOS token. This makes decode-throughput numbers reproducible.
        """
        # Attach extra_body only when ignore_eos is set; otherwise send the
        # plain request so no vLLM-specific field reaches non-vLLM backends.
        extra: dict[str, dict[str, bool]] = {}
        if self._ignore_eos:
            extra["extra_body"] = {"ignore_eos": True}
        if self._completions_mode:
            prompt = messages[0]["content"] if messages else ""
            return await self._client.completions.create(
                model=self._model,
                prompt=prompt,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.0,
                stream_options={"include_usage": True},
                **extra,
            )
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.0,
            stream_options={"include_usage": True},
            **extra,
        )
