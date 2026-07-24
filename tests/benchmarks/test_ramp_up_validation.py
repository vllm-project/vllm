# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ramp-up argument validation in `vllm bench serve`.

The `--ramp-up-strategy linear` path used to crash at runtime when
`--ramp-up-start-rps 0` was passed: `_get_current_request_rate` returns
`0.0` at `progress=0`, which then fails the `current_request_rate > 0.0`
assertion and would otherwise hit a `ZeroDivisionError` in the gamma
delay computation. The exponential path already validated this, but the
linear path did not. These tests pin down both the runtime symptom and
the validation that prevents it.
"""

import argparse
import asyncio

import pytest

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.serve import (
    _get_current_request_rate,
    get_request,
    main_async,
)


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for `main_async` validation paths.

    Only fields that the early validation in `main_async` reads need to be
    populated; the rest can stay as defaults that the test never reaches.
    """
    defaults: dict = {
        "seed": 0,
        "plot_timeline": False,
        "timeline_itl_thresholds": "25,50",
        "ramp_up_strategy": None,
        "ramp_up_start_rps": None,
        "ramp_up_end_rps": None,
        "request_rate": float("inf"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_linear_ramp_up_with_zero_start_rps_is_rejected() -> None:
    """Linear ramp-up with start_rps=0 must be rejected up front.

    Without validation, the first request would receive rate=0 from
    `_get_current_request_rate`, which then fails the `> 0` assertion
    inside `get_request`. Reject at the CLI boundary instead.
    """
    args = _make_args(
        ramp_up_strategy="linear",
        ramp_up_start_rps=0,
        ramp_up_end_rps=10,
    )
    with pytest.raises(ValueError, match="start RPS must be greater than 0"):
        asyncio.run(main_async(args))


def test_exponential_ramp_up_with_zero_start_rps_is_rejected() -> None:
    """Exponential ramp-up with start_rps=0 must still be rejected.

    Regression guard for the existing validation, which previously
    lived in its own branch and is now unified with the linear path.
    """
    args = _make_args(
        ramp_up_strategy="exponential",
        ramp_up_start_rps=0,
        ramp_up_end_rps=10,
    )
    with pytest.raises(ValueError, match="start RPS must be greater than 0"):
        asyncio.run(main_async(args))


def test_linear_ramp_up_first_request_rate_with_zero_start_is_zero() -> None:
    """Document the underlying broken behaviour at idx=0.

    `_get_current_request_rate` returns `start_rps` at `progress=0` for
    linear ramp. With `start=0`, the first request's rate is exactly 0,
    which is the value that breaks downstream invariants.
    """
    rate = _get_current_request_rate(
        ramp_up_strategy="linear",
        ramp_up_start_rps=0,
        ramp_up_end_rps=10,
        request_index=0,
        total_requests=5,
        request_rate=float("inf"),
    )
    assert rate == 0.0


def test_get_request_rejects_zero_rate_from_linear_ramp_up() -> None:
    """`get_request` must surface the zero-rate first-request case.

    Even if validation is bypassed (e.g. programmatic callers), the
    runtime assertion in `get_request` must still trip rather than
    sleep forever or divide by zero.
    """
    reqs = [
        SampleRequest(prompt=f"p{i}", prompt_len=1, expected_output_len=1)
        for i in range(3)
    ]

    async def _drive() -> None:
        gen = get_request(
            input_requests=reqs,
            request_rate=float("inf"),
            ramp_up_strategy="linear",
            ramp_up_start_rps=0,
            ramp_up_end_rps=10,
        )
        async for _ in gen:
            pass

    with pytest.raises(AssertionError, match="non-positive request rate"):
        asyncio.run(_drive())
