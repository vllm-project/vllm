# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF Whisper generate_with_fallback, as a mixin around LLM.generate.

Source: transformers.models.whisper.generation_whisper
  WhisperGenerationMixin.generate, generate_with_fallback, _need_fallback

HF does **not** put this on GenerationMixin.generate. Users call
WhisperForConditionalGeneration.generate(..., temperature=(0, 0.2, ...),
compression_ratio_threshold=1.35, logprob_threshold=-1.0). Inner decode is
super().generate(). Thresholds default to None: one greedy pass, no retry.

vLLM analog: WhisperGenerationMixin.generate wraps super().generate()
(LLM.generate). Do not auto-hook the generic LLM class.

    from vllm.whisper_generation import WhisperLLM  # or WhisperGenerationMixin

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

import math
import zlib
from collections.abc import Sequence

from vllm.sampling_params import SamplingParams

# HF generation_whisper.py docs: 1.35 / -1.0 are common. OpenAI used gzip 2.4.
COMPRESSION_RATIO_THRESHOLD = 1.35
LOGPROB_THRESHOLD = -1.0
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def compression_ratio(token_ids: Sequence[int], vocab_size: int) -> float:
    if not token_ids:
        return float("inf")
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


def _avg_logprob(completion) -> float | None:
    ids = getattr(completion, "token_ids", None) or []
    ntok = max(len(ids), 1)
    lp = getattr(completion, "cumulative_logprob", None)
    if lp is None:
        return None
    return float(lp) / ntok


def _clone_sampling_params(sampling_params: SamplingParams) -> SamplingParams:
    clone = getattr(sampling_params, "clone", None)
    if callable(clone):
        return clone()
    import copy

    return copy.copy(sampling_params)


def _normalize_temperatures(
    temperature: float | Sequence[float] | None,
    sampling_params: SamplingParams | None,
) -> tuple[float, ...]:
    if temperature is None:
        t = getattr(sampling_params, "temperature", None)
        return (0.0 if t is None else float(t),)
    if isinstance(temperature, (int, float)):
        return (float(temperature),)
    return tuple(float(x) for x in temperature)


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

        prompts_list = prompts if isinstance(prompts, (list, tuple)) else [prompts]
        super_generate = super().generate  # type: ignore[misc]
        temperatures = _normalize_temperatures(temperature, sampling_params)
        get_tokenizer = getattr(self, "get_tokenizer", None)
        if not callable(get_tokenizer):
            raise AttributeError("WhisperGenerationMixin requires get_tokenizer()")
        tokenizer = get_tokenizer()
        vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))

        n = len(prompts_list)
        finals = [None] * n
        pending = list(range(n))
        cur_prompts = list(prompts_list)

        for t_idx, t in enumerate(temperatures):
            sp = _clone_sampling_params(sampling_params)
            sp.temperature = float(t)
            if t > 0:
                top_p = float(getattr(sp, "top_p", 0) or 0)
                sp.top_p = top_p if top_p > 0 else 1.0

            outputs = super_generate(
                cur_prompts, sampling_params=sp, use_tqdm=use_tqdm, **kwargs
            )

            still: list[int] = []
            still_prompts = []
            last = t_idx == len(temperatures) - 1
            for local_i, req_i in enumerate(pending):
                out = outputs[local_i]
                completion = out.outputs[0]
                ids = list(completion.token_ids)
                retry = needs_fallback(
                    ids,
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
