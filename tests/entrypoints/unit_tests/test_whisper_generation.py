# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from vllm.whisper_generation import (
    COMPRESSION_RATIO_THRESHOLD,
    LOGPROB_THRESHOLD,
    compression_ratio,
    needs_fallback,
)


def test_empty_token_ids_need_fallback():
    assert math.isinf(compression_ratio([], vocab_size=51865))
    assert needs_fallback([], vocab_size=51865) is True


def test_looping_tokens_exceed_gzip_threshold():
    loop = [42, 43, 44] * 80
    ratio = compression_ratio(loop, vocab_size=51865)
    assert ratio > COMPRESSION_RATIO_THRESHOLD
    assert needs_fallback(loop, vocab_size=51865) is True


def test_diverse_tokens_skip_fallback():
    tokens = list(range(1, 40))
    assert compression_ratio(tokens, vocab_size=51865) <= COMPRESSION_RATIO_THRESHOLD
    assert needs_fallback(tokens, vocab_size=51865, avg_logprob=-0.2) is False


def test_low_logprob_needs_fallback():
    tokens = list(range(1, 20))
    assert needs_fallback(
        tokens, vocab_size=51865, avg_logprob=LOGPROB_THRESHOLD - 0.1
    )
    assert not needs_fallback(
        tokens, vocab_size=51865, avg_logprob=LOGPROB_THRESHOLD + 0.1
    )


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _LLM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_is_whisper_llm_detects_architecture():
    from vllm.whisper_generation import is_whisper_llm

    assert is_whisper_llm(
        _LLM(model_config=_Cfg(architecture="WhisperForConditionalGeneration"))
    )
    assert is_whisper_llm(
        _LLM(model_config=_Cfg(hf_config=_Cfg(model_type="whisper")))
    )
    assert not is_whisper_llm(_LLM(model_config=_Cfg(architecture="LlamaForCausalLM")))
    assert not is_whisper_llm(_LLM())
