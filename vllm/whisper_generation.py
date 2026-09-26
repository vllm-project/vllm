# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF Whisper generate_with_fallback, for vLLM LLM.generate.

Source: transformers.models.whisper.generation_whisper
  generate_with_fallback, _need_fallback, _retrieve_compression_ratio

Do not put this in model_executor/models/whisper.py (architecture/forward only).
Call from LLM.generate when the loaded arch is Whisper.

Empty token_ids return gzip inf so immediate-EOT completions retry. HF keeps
EOS in the sequence so its gzip on an empty tensor is 0.0 and would not retry.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Sequence

from vllm.sampling_params import SamplingParams

# HF generation_whisper.py docs: 1.35 is common. OpenAI transcribe() used 2.4.
COMPRESSION_RATIO_THRESHOLD = 1.35
LOGPROB_THRESHOLD = -1.0
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

WHISPER_ARCH = "WhisperForConditionalGeneration"


def is_whisper_llm(llm: Any) -> bool:
    try:
        cfg = getattr(llm, "model_config", None)
        if cfg is None:
            engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
            cfg = getattr(engine, "model_config", None)
        if cfg is None:
            return False
        if getattr(cfg, "architecture", None) == WHISPER_ARCH:
            return True
        arch = getattr(cfg, "architectures", None) or getattr(
            getattr(cfg, "hf_config", None), "architectures", None
        )
        if arch and WHISPER_ARCH in arch:
            return True
        model_type = getattr(getattr(cfg, "hf_config", None), "model_type", "")
        return model_type == "whisper"
    except Exception:
        return False


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
) -> bool:
    if not token_ids:
        return True
    if compression_ratio(token_ids, vocab_size) > COMPRESSION_RATIO_THRESHOLD:
        return True
    if avg_logprob is not None and avg_logprob < LOGPROB_THRESHOLD:
        return True
    return False


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


def generate_whisper_with_fallback(
    llm: Any,
    prompts,
    sampling_params: SamplingParams,
    *,
    generate_once,
    use_tqdm: bool = False,
):
    """Retry looping / low-logprob hyps at higher temperature.

    ``generate_once`` must be the inner LLM.generate (no fallback) to avoid
    recursion. First pass is always T=0 (HF greedy), then 0.2 … 1.0.
    """
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))

    n = len(prompts)
    finals = [None] * n
    pending = list(range(n))
    cur_prompts = list(prompts)

    for t_idx, temperature in enumerate(TEMPERATURES):
        sp = _clone_sampling_params(sampling_params)
        sp.temperature = float(temperature)
        if temperature > 0:
            top_p = float(getattr(sp, "top_p", 0) or 0)
            sp.top_p = top_p if top_p > 0 else 1.0

        outputs = generate_once(
            cur_prompts, sampling_params=sp, use_tqdm=use_tqdm
        )

        still: list[int] = []
        still_prompts = []
        last = t_idx == len(TEMPERATURES) - 1
        for local_i, req_i in enumerate(pending):
            out = outputs[local_i]
            completion = out.outputs[0]
            ids = list(completion.token_ids)
            if last or not needs_fallback(ids, vocab_size, _avg_logprob(completion)):
                finals[req_i] = out
            else:
                still.append(req_i)
                still_prompts.append(cur_prompts[local_i])
        if not still:
            break
        pending = still
        cur_prompts = still_prompts

    return finals
