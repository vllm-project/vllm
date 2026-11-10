# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.speech_to_text.base.fallback import (
    TEMPERATURES,
    generate_with_temperature_fallback,
    needs_fallback,
)
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_test

VOCAB_SIZE = 51866
LOOPING = [7, 8, 9, 10] * 80
DIVERSE = list(range(1000, 50000, 613))


def _completion(token_ids, cumulative_logprob=None):
    return SimpleNamespace(
        token_ids=token_ids, cumulative_logprob=cumulative_logprob, text=""
    )


@pytest.mark.parametrize(
    ("token_ids", "cumulative_logprob", "expected"),
    [
        ([], None, True),
        (LOOPING, None, True),
        (DIVERSE, None, False),
        (DIVERSE, -0.5 * len(DIVERSE), False),
        (DIVERSE, -1.5 * len(DIVERSE), True),
    ],
)
def test_needs_fallback(token_ids, cumulative_logprob, expected):
    completion = _completion(token_ids, cumulative_logprob)
    assert needs_fallback(completion, VOCAB_SIZE) is expected


@pytest.mark.asyncio
async def test_retries_until_clean_output():
    calls: list[tuple[float, str]] = []

    async def generate(prompt, params, request_id, **kwargs):
        calls.append((params.temperature, request_id))
        token_ids = LOOPING if params.temperature < 0.4 else DIVERSE
        yield SimpleNamespace(outputs=[_completion(token_ids)])

    sampling_params = SamplingParams(temperature=0.0)
    outputs = [
        out
        async for out in generate_with_temperature_fallback(
            generate, {"prompt": "x"}, sampling_params, "req", VOCAB_SIZE
        )
    ]

    assert calls == [(0.0, "req"), (0.2, "req-fallback-1"), (0.4, "req-fallback-2")]
    assert [out.outputs[0].token_ids for out in outputs] == [DIVERSE]
    # The caller's params are left untouched.
    assert sampling_params.temperature == 0.0
    assert sampling_params.logprobs is None


@pytest.mark.asyncio
async def test_keeps_last_attempt_when_all_fail():
    temperatures: list[float] = []

    async def generate(prompt, params, request_id, **kwargs):
        temperatures.append(params.temperature)
        yield SimpleNamespace(outputs=[_completion(LOOPING)])

    outputs = [
        out
        async for out in generate_with_temperature_fallback(
            generate,
            {"prompt": "x"},
            SamplingParams(temperature=0.0),
            "req",
            VOCAB_SIZE,
        )
    ]

    assert temperatures == list(TEMPERATURES)
    assert len(outputs) == 1
