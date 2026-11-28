# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temperature fallback for greedy transcription, as in HF Whisper
``generate_with_fallback`` (transformers.models.whisper.generation_whisper)."""

import math
import zlib
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

# HF long-form defaults.
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 1.35
LOGPROB_THRESHOLD = -1.0


def compression_ratio(token_ids: Sequence[int], vocab_size: int) -> float:
    width = int(math.log2(vocab_size) / 8) + 1
    raw = b"".join(t.to_bytes(width, "little") for t in token_ids)
    return len(raw) / len(zlib.compress(raw))


def needs_fallback(completion: CompletionOutput, vocab_size: int) -> bool:
    token_ids = completion.token_ids
    # Unlike HF, vLLM strips EOS, so an immediate EOS yields no tokens at all.
    if not token_ids:
        return True
    if compression_ratio(token_ids, vocab_size) > COMPRESSION_RATIO_THRESHOLD:
        return True
    logprob = completion.cumulative_logprob
    return logprob is not None and logprob / len(token_ids) < LOGPROB_THRESHOLD


async def generate_with_temperature_fallback(
    generate: Callable[..., AsyncGenerator[RequestOutput, None]],
    prompt: Any,
    sampling_params: SamplingParams,
    request_id: str,
    vocab_size: int,
    **kwargs: Any,
) -> AsyncGenerator[RequestOutput, None]:
    """Run ``generate`` at increasing temperatures until the output passes the
    compression ratio and logprob checks, then yield the accepted attempt."""
    for i, temperature in enumerate(TEMPERATURES):
        params = sampling_params.clone()
        params.temperature = temperature
        if params.logprobs is None:
            # Needed to populate cumulative_logprob.
            params.logprobs = 0
        attempt_id = request_id if i == 0 else f"{request_id}-fallback-{i}"
        outputs = [out async for out in generate(prompt, params, attempt_id, **kwargs)]
        is_last = i == len(TEMPERATURES) - 1
        if is_last or not needs_fallback(outputs[-1].outputs[0], vocab_size):
            for out in outputs:
                yield out
            return
