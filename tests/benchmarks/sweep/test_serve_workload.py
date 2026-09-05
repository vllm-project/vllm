# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import sys
import types

import numpy as np
import pytest

# `serve_workload` imports `DEFAULT_NUM_PROMPTS` from `vllm.benchmarks.datasets`
# at module load, which pulls in the heavy tokenizer/transformers stack. The
# rate-anchor logic under test does not need it, so stub the module before
# import to keep this a pure-Python unit test (no GPU / model deps).
_datasets_stub = types.ModuleType("vllm.benchmarks.datasets")
_datasets_stub.DEFAULT_NUM_PROMPTS = 1000
sys.modules.setdefault("vllm.benchmarks.datasets", _datasets_stub)

serve_workload = importlib.import_module("vllm.benchmarks.sweep.serve_workload")

_round_workload_value = serve_workload._round_workload_value


def _sweep_levels(serial_avg, batch_avg, workload_var, workload_iters):
    """Mirror the anchor + intermediate computation in `explore_comb_workloads`."""
    serial_value = _round_workload_value(serial_avg, workload_var)
    batch_value = _round_workload_value(batch_avg, workload_var)

    inter = np.linspace(serial_value, batch_value, workload_iters)[1:-1]
    inter = sorted({_round_workload_value(v, workload_var) for v in inter})

    return sorted({serial_value, batch_value, *inter})


class TestRoundWorkloadValue:
    def test_request_rate_keeps_sub_one_values(self):
        # Regression for the "non-positive request rate" crash: a sub-1
        # throughput must not collapse to 0 (which asserts in serve.py).
        assert _round_workload_value(0.425, "request_rate") == 0.425
        assert _round_workload_value(0.21, "request_rate") == 0.21

    def test_request_rate_never_non_positive(self):
        for value in (0.0, 0.0001, -0.5):
            assert _round_workload_value(value, "request_rate") > 0.0

    def test_max_concurrency_is_integer_and_at_least_one(self):
        value = _round_workload_value(0.425, "max_concurrency")
        assert isinstance(value, int)
        assert value == 1

        value = _round_workload_value(0.0, "max_concurrency")
        assert value == 1

    def test_max_concurrency_rounds_normally_above_one(self):
        assert _round_workload_value(7.4, "max_concurrency") == 7
        assert _round_workload_value(7.6, "max_concurrency") == 8


class TestSweepLevels:
    def test_sub_one_request_rate_does_not_emit_zero(self):
        # serial ~0.21 req/s, batch ~0.425 req/s (measured on an RTX 4090 with
        # 8K-token inputs, from issue #47651). Before the fix this produced
        # request_rate levels [0, 1], crashing on the 0 level.
        levels = _sweep_levels(0.21, 0.425, "request_rate", workload_iters=10)

        assert all(level > 0.0 for level in levels)
        # Sub-1 resolution is preserved instead of collapsing to {0, 1}.
        assert any(0.0 < level < 1.0 for level in levels)
        assert len(levels) > 2

    def test_request_rate_above_one_still_works(self):
        levels = _sweep_levels(2.0, 12.0, "request_rate", workload_iters=5)
        assert all(level > 0.0 for level in levels)
        assert min(levels) == pytest.approx(2.0)
        assert max(levels) == pytest.approx(12.0)

    def test_max_concurrency_never_emits_zero(self):
        # With `--workload-var max_concurrency`, a floored batch anchor could
        # emit max_concurrency=0, which is not a valid workload level either.
        levels = _sweep_levels(0.4, 0.9, "max_concurrency", workload_iters=10)
        assert all(isinstance(level, int) and level >= 1 for level in levels)
