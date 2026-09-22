# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for max_model_len fan-out on the V2 GPU runner (upstream #57034).

When the engine auto-fits ``max_model_len`` down to what the KV cache can hold, it
broadcasts the reduced value to the workers. Every copy the runner cached before
memory profiling has to follow, or the cudagraph capture path keeps sizing
worst-case sequences from the pre-auto-fit length while the attention metadata
builders have already sized their buffers from the reduced one.

``ModelState`` builds its copy in ``load_model``, i.e. before profiling, so it is
one of those caches.

CPU only: ``update_max_model_len`` is plain attribute assignment, so the bound
function runs against a stand-in with the same attributes; no model is loaded and
no kernel is launched.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

FULL_LEN = 1048576
FITTED_LEN = 980224


def _runner_stand_in() -> SimpleNamespace:
    """A runner whose caches all start at the pre-auto-fit length."""
    return SimpleNamespace(
        max_model_len=FULL_LEN,
        req_states=SimpleNamespace(max_model_len=FULL_LEN),
        model_state=SimpleNamespace(max_model_len=FULL_LEN),
    )


def test_update_max_model_len_reaches_model_state():
    runner = _runner_stand_in()

    GPUModelRunner.update_max_model_len(runner, FITTED_LEN)

    # The capture path reads ModelState.max_model_len; leaving it at FULL_LEN is
    # what trips DeepSeek-V4 sparse MLA's C128A assertion during capture.
    assert runner.model_state.max_model_len == FITTED_LEN


def test_update_max_model_len_leaves_no_cache_behind():
    runner = _runner_stand_in()

    GPUModelRunner.update_max_model_len(runner, FITTED_LEN)

    stale = {
        name: value.max_model_len if isinstance(value, SimpleNamespace) else value
        for name, value in vars(runner).items()
    }
    assert set(stale.values()) == {FITTED_LEN}, f"stale copies remain: {stale}"
