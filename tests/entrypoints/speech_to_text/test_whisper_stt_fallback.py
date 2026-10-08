# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gzip/logprob temperature ladder on the STT engine generate path."""

import asyncio

from vllm.entrypoints.whisper import generate_chunk_with_gzip_fallback


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
    def __init__(self, ids, text, lp):
        self.outputs = [_C(ids, text, lp)]
        self.finished = True


def test_stt_fallback_retries_loop_then_keeps_clean():
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
            yield _Out(unique, "ok", -8.0)

    async def _run():
        return [
            out
            async for out in generate_chunk_with_gzip_fallback(
                engine_generate,
                {"prompt": "chunk"},
                _SP(),
                "transcribe-1",
                vocab_size=51865,
            )
        ]

    outs = asyncio.run(_run())
    assert temps[:2] == [0.0, 0.2]
    assert rids[0] == "transcribe-1"
    assert rids[1] == "transcribe-1-fb-1"
    assert len(outs) == 1
    assert outs[0].outputs[0].text == "ok"
