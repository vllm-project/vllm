# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from vllm.entrypoints.speech_to_text.whisper import (
    COMPRESSION_RATIO_THRESHOLD,
    LOGPROB_THRESHOLD,
    TEMPERATURES,
    WHISPER_VOCAB_SIZE,
    WhisperGenerationMixin,
    compression_ratio,
    needs_fallback,
)


def test_empty_token_ids_gzip_inf_but_no_retry_without_threshold():
    assert math.isinf(compression_ratio([], vocab_size=WHISPER_VOCAB_SIZE))
    assert needs_fallback([], vocab_size=WHISPER_VOCAB_SIZE) is False
    assert needs_fallback(
        [],
        vocab_size=WHISPER_VOCAB_SIZE,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
    )


def test_looping_tokens_exceed_gzip_threshold():
    loop = [42, 43, 44] * 80
    ratio = compression_ratio(loop, vocab_size=WHISPER_VOCAB_SIZE)
    assert ratio > COMPRESSION_RATIO_THRESHOLD
    assert needs_fallback(
        loop,
        vocab_size=WHISPER_VOCAB_SIZE,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
    )
    assert needs_fallback(loop, vocab_size=WHISPER_VOCAB_SIZE) is False


def test_diverse_tokens_skip_fallback():
    tokens = list(range(1, 40))
    assert (
        compression_ratio(tokens, vocab_size=WHISPER_VOCAB_SIZE)
        <= COMPRESSION_RATIO_THRESHOLD
    )
    assert (
        needs_fallback(
            tokens,
            vocab_size=WHISPER_VOCAB_SIZE,
            avg_logprob=-0.2,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
        )
        is False
    )


def test_low_logprob_needs_fallback():
    tokens = list(range(1, 20))
    assert needs_fallback(
        tokens,
        vocab_size=WHISPER_VOCAB_SIZE,
        avg_logprob=LOGPROB_THRESHOLD - 0.1,
        logprob_threshold=LOGPROB_THRESHOLD,
    )
    assert not needs_fallback(
        tokens,
        vocab_size=WHISPER_VOCAB_SIZE,
        avg_logprob=LOGPROB_THRESHOLD + 0.1,
        logprob_threshold=LOGPROB_THRESHOLD,
    )


class _C:
    def __init__(self, ids, text, lp):
        self.token_ids = ids
        self.text = text
        self.cumulative_logprob = lp


class _Out:
    def __init__(self, ids, text, lp):
        self.outputs = [_C(ids, text, lp)]


class _Tok:
    vocab_size = WHISPER_VOCAB_SIZE


def _whisper_llm(on_generate):
    class Inner:
        def get_tokenizer(self):
            return _Tok()

        def generate(self, prompts, sampling_params=None, use_tqdm=False, **kwargs):
            return on_generate(prompts, sampling_params)

    class WhisperInner(WhisperGenerationMixin, Inner):  # type: ignore[misc]
        pass

    return WhisperInner()


def test_no_retry_when_thresholds_unset_hf_shortform():
    from vllm.sampling_params import SamplingParams

    looping = [7, 8, 9, 10] * 80
    temps: list[float] = []

    def on_generate(prompts, sampling_params):
        temps.append(float(sampling_params.temperature))
        return [_Out(looping, "loop", -4.0) for _ in prompts]

    outs = _whisper_llm(on_generate).generate(
        ["p0"],
        SamplingParams(temperature=0, max_tokens=256),
        use_tqdm=False,
    )
    assert temps == [0.0]
    assert outs[0].outputs[0].text == "loop"


def test_mixin_generate_retries_loop_then_keeps_clean():
    from vllm.sampling_params import SamplingParams

    looping = [7, 8, 9, 10] * 80
    unique = list(range(40, 120))
    temps: list[float] = []

    def on_generate(prompts, sampling_params):
        temps.append(float(sampling_params.temperature))
        if sampling_params.temperature == 0.0:
            return [_Out(looping, "i am a man of the law " * 40, -4.0) for _ in prompts]
        return [_Out(unique, "doctor of laws", -8.0) for _ in prompts]

    outs = _whisper_llm(on_generate).generate(
        ["p0", "p1"],
        SamplingParams(temperature=0, max_tokens=256),
        temperature=TEMPERATURES,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD,
        use_tqdm=False,
    )
    assert temps[:2] == [0.0, 0.2]
    assert all(o.outputs[0].text == "doctor of laws" for o in outs)
