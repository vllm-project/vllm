# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gzip/logprob temperature ladder on the transcription engine generate path."""

import asyncio

from vllm.entrypoints.speech_to_text.whisper import (
    WHISPER_VOCAB_SIZE,
    generate_chunk_with_gzip_fallback,
    should_gzip_fallback,
)


class _SP:
    def __init__(self, temperature=0.0, top_p=1.0):
        self.temperature = temperature
        self.top_p = top_p

    def clone(self):
        return _SP(self.temperature, self.top_p)


class _C:
    def __init__(self, ids, text, lp):
        self.token_ids = ids
        self.text = text
        self.cumulative_logprob = lp


class _Out:
    def __init__(self, ids, text, lp, finished=True):
        self.outputs = [_C(ids, text, lp)]
        self.finished = finished


class _Req:
    def __init__(self, temperature=0.0, stream=False):
        self.temperature = temperature
        self.stream = stream


class _Whisper:
    __name__ = "WhisperForConditionalGeneration"


class _Qwen:
    __name__ = "Qwen3ASRForConditionalGeneration"


def test_stt_fallback_retries_loop_then_yields_kept_attempt():
    looping = [7, 8, 9, 10] * 80
    unique = list(range(40, 120))
    temps: list[float] = []
    rids: list[str] = []

    async def engine_generate(_prompt, sampling_params, request_id, **_kw):
        temps.append(float(sampling_params.temperature))
        rids.append(request_id)
        if sampling_params.temperature == 0.0:
            yield _Out(looping, "loop", -4.0)
        else:
            yield _Out(unique, "partial", -8.0, finished=False)
            yield _Out(unique, "ok", -8.0, finished=True)

    async def _run():
        return [
            out
            async for out in generate_chunk_with_gzip_fallback(
                engine_generate,
                {"prompt": "chunk"},
                _SP(),
                "transcribe-1",
                vocab_size=WHISPER_VOCAB_SIZE,
            )
        ]

    outs = asyncio.run(_run())
    assert temps[:2] == [0.0, 0.2]
    assert rids[0] == "transcribe-1"
    assert rids[1] == "transcribe-1-fb-1"
    assert [o.outputs[0].text for o in outs] == ["partial", "ok"]
    assert outs[-1].finished is True


def test_should_gzip_fallback_transcription_whisper_t0_only():
    assert should_gzip_fallback(_Whisper, _Req(temperature=0.0, stream=False))
    assert should_gzip_fallback(_Whisper, _Req(temperature=None, stream=False))
    assert not should_gzip_fallback(_Whisper, _Req(temperature=0.2, stream=False))
    assert not should_gzip_fallback(_Whisper, _Req(temperature=0.0, stream=True))
    assert not should_gzip_fallback(_Qwen, _Req(temperature=0.0, stream=False))
