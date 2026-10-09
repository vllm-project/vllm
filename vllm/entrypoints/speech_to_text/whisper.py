# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF Whisper generate_with_fallback for STT transcription + optional LLM.

Source: transformers.models.whisper.generation_whisper
  WhisperGenerationMixin.generate, generate_with_fallback, _need_fallback

Offline: WhisperGenerationMixin.generate wraps super().generate() (LLM.generate).
Do not auto-hook the generic LLM class.

Serving: /v1/audio/transcriptions only (OpenAIServingTranscription). Translation
and SpeechToTextBaseServing keep engine_client.generate.

    from vllm.entrypoints.speech_to_text.whisper import WhisperLLM

    llm = WhisperLLM(model="openai/whisper-large-v3")
    outputs = llm.generate(
        prompts,
        sampling_params,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=1.35,
        logprob_threshold=-1.0,
    )

Empty token_ids return gzip inf so, when a compression threshold is set,
immediate-EOT completions retry. HF keeps EOS so empty gzip is 0.0.
"""

from __future__ import annotations

import copy
import math
import zlib
from collections.abc import AsyncGenerator, Sequence

from vllm.sampling_params import SamplingParams

# Long-form defaults from HuggingFace
# ``transformers.models.whisper.generation_whisper``
# (``generate_with_fallback`` / ``_need_fallback``).
# OpenAI whisper.cpp used gzip 2.4; we match HF, not that OpenAI constant.
COMPRESSION_RATIO_THRESHOLD = 1.35  # retry when token-id gzip ratio exceeds this
LOGPROB_THRESHOLD = -1.0  # retry when mean token logprob is below this
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # greedy first, then +0.2 up to 1.0
# Multilingual Whisper tokenizer size (English-only is 51864). Gzip packs
# each token into ceil(log2(vocab_size)/8) bytes; STT serving may have no
# tokenizer yet, so this is the fallback width.
WHISPER_VOCAB_SIZE = 51865


def compression_ratio(token_ids: Sequence[int], vocab_size: int) -> float:
    if not token_ids:
        return float("inf")
    # Bits-per-token / 8 → whole bytes, then +1 so ids always fit.
    width = int(math.log2(max(vocab_size, 2)) / 8) + 1
    raw = b"".join(int(t).to_bytes(width, "little", signed=False) for t in token_ids)
    return len(raw) / max(len(zlib.compress(raw)), 1)


def needs_fallback(
    token_ids: Sequence[int],
    vocab_size: int,
    avg_logprob: float | None = None,
    *,
    compression_ratio_threshold: float | None = None,
    logprob_threshold: float | None = None,
) -> bool:
    """HF `_need_fallback`: no-op unless a threshold is set.

    Empty vLLM `token_ids` (EOS stripped) gzip as inf, so they retry only when
    ``compression_ratio_threshold`` or ``logprob_threshold`` is set.
    """
    if not token_ids:
        return compression_ratio_threshold is not None or logprob_threshold is not None
    if (
        compression_ratio_threshold is not None
        and compression_ratio(token_ids, vocab_size) > compression_ratio_threshold
    ):
        return True
    return (
        logprob_threshold is not None
        and avg_logprob is not None
        and avg_logprob < logprob_threshold
    )


def gzip_vocab_size(tokenizer) -> int:
    if tokenizer is None:
        return WHISPER_VOCAB_SIZE
    return int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))


def is_whisper_model(model_cls) -> bool:
    return "whisper" in getattr(model_cls, "__name__", "").lower()


def should_gzip_fallback(model_cls, request) -> bool:
    """OpenAI T=0 gzip ladder: Whisper transcription, non-streaming only."""
    if not is_whisper_model(model_cls):
        return False
    if bool(getattr(request, "stream", False)):
        return False
    t = getattr(request, "temperature", None)
    return t is None or float(t) == 0.0


def _avg_logprob(completion) -> float | None:
    ids = getattr(completion, "token_ids", None) or []
    lp = getattr(completion, "cumulative_logprob", None)
    if lp is None:
        return None
    return float(lp) / max(len(ids), 1)


def _sampling_params_at_temperature(
    sampling_params: SamplingParams, temperature: float
) -> SamplingParams:
    clone_fn = getattr(sampling_params, "clone", None)
    clone = clone_fn() if callable(clone_fn) else copy.copy(sampling_params)
    clone.temperature = float(temperature)
    if temperature > 0:
        top_p = float(getattr(clone, "top_p", 0) or 0)
        clone.top_p = top_p if top_p > 0 else 1.0
    return clone


async def generate_chunk_with_gzip_fallback(
    engine_generate,
    engine_input,
    sampling_params: SamplingParams,
    request_id: str,
    *,
    vocab_size: int,
    **generate_kwargs,
) -> AsyncGenerator:
    """HF temperature ladder; yield the kept attempt as an engine generator."""
    for t_idx, temperature in enumerate(TEMPERATURES):
        sp = _sampling_params_at_temperature(sampling_params, temperature)
        rid = request_id if t_idx == 0 else f"{request_id}-fb-{t_idx}"
        buffered = []
        async for output in engine_generate(engine_input, sp, rid, **generate_kwargs):
            buffered.append(output)
            if getattr(output, "finished", True):
                break
        if not buffered or not getattr(buffered[-1], "outputs", None):
            continue
        completion = buffered[-1].outputs[0]
        retry = needs_fallback(
            list(completion.token_ids),
            vocab_size,
            _avg_logprob(completion),
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
        )
        if t_idx == len(TEMPERATURES) - 1 or not retry:
            for output in buffered:
                yield output
            return


class WhisperGenerationMixin:
    """HF ``WhisperGenerationMixin``: wrap ``super().generate()``.

    Generic ``LLM.generate`` is ``GenerationMixin.generate`` and stays
    model-agnostic. Subclass: ``class WhisperLLM(WhisperGenerationMixin, LLM)``.
    """

    def generate(
        self,
        prompts,
        sampling_params: SamplingParams | None = None,
        *,
        temperature: float | Sequence[float] | None = None,
        compression_ratio_threshold: float | None = None,
        logprob_threshold: float | None = None,
        use_tqdm: bool = True,
        **kwargs,
    ):
        """HF ``generate`` + ``generate_with_fallback``.

        Retry only if a threshold is set (HF long-form). Both thresholds
        ``None`` is one ``super().generate()`` (HF short-form).
        """
        if sampling_params is None:
            getter = getattr(self, "get_default_sampling_params", None)
            sampling_params = getter() if callable(getter) else SamplingParams()

        temperatures: tuple[float, ...]
        if temperature is None:
            t = getattr(sampling_params, "temperature", None)
            temperatures = (0.0 if t is None else float(t),)
        elif isinstance(temperature, (int, float)):
            temperatures = (float(temperature),)
        else:
            temperatures = tuple(float(x) for x in temperature)

        prompts_list = prompts if isinstance(prompts, (list, tuple)) else [prompts]
        super_generate = super().generate  # type: ignore[misc]
        get_tokenizer = getattr(self, "get_tokenizer", None)
        if not callable(get_tokenizer):
            raise AttributeError("WhisperGenerationMixin requires get_tokenizer()")
        tokenizer = get_tokenizer()
        vocab_size = gzip_vocab_size(tokenizer)

        n = len(prompts_list)
        finals = [None] * n
        pending = list(range(n))
        cur_prompts = list(prompts_list)

        for t_idx, t in enumerate(temperatures):
            sp = _sampling_params_at_temperature(sampling_params, t)
            outputs = super_generate(
                cur_prompts, sampling_params=sp, use_tqdm=use_tqdm, **kwargs
            )
            still: list[int] = []
            still_prompts = []
            last = t_idx == len(temperatures) - 1
            for local_i, req_i in enumerate(pending):
                out = outputs[local_i]
                completion = out.outputs[0]
                retry = needs_fallback(
                    list(completion.token_ids),
                    vocab_size,
                    _avg_logprob(completion),
                    compression_ratio_threshold=compression_ratio_threshold,
                    logprob_threshold=logprob_threshold,
                )
                if last or not retry:
                    finals[req_i] = out
                else:
                    still.append(req_i)
                    still_prompts.append(cur_prompts[local_i])
            if not still:
                break
            pending = still
            cur_prompts = still_prompts

        return finals


def __getattr__(name: str):
    if name == "WhisperLLM":
        from vllm.entrypoints.llm import LLM

        class WhisperLLM(WhisperGenerationMixin, LLM):  # type: ignore[misc]
            """HF WhisperForConditionalGeneration analog."""

        return WhisperLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
