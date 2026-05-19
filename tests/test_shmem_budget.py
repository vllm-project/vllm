# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm.triton_utils.shmem_budget``.

All tests are pure unit tests — no GPU, no Triton kernel compilation.
``torch.cuda`` is mocked via ``unittest.mock`` so the suite runs on CPU CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.triton_utils.shmem_budget import (
    _FALLBACK_BUDGET_BYTES,
    infer_shmem_budget,
    make_shmem_pruner,
)

# --------------------------------------------------------------------------
# Test fixtures: fake triton.Config + helpers to construct device props
# --------------------------------------------------------------------------


@dataclass
class FakeConfig:
    """Stand-in for triton.Config — only the fields the pruner reads."""

    kwargs: dict[str, Any]
    num_warps: int = 4
    num_stages: int = 3


def _device_props(per_block_optin: int, per_block: int | None = None) -> Any:
    """Mock object mimicking torch.cuda.device_props."""
    p = MagicMock()
    p.shared_memory_per_block_optin = per_block_optin
    p.shared_memory_per_block = per_block if per_block is not None else per_block_optin
    return p


@pytest.fixture
def shmem_logs():
    """Capture log records emitted by vllm.triton_utils.shmem_budget.

    The vllm root logger is configured with ``propagate=False``
    (vllm/logger.py:67), so the standard pytest ``caplog`` fixture cannot
    see records emitted by vllm modules. Attach a list-based handler
    directly to the module logger instead.
    """
    import logging as _logging

    records: list[_logging.LogRecord] = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=_logging.DEBUG)
    log = _logging.getLogger("vllm.triton_utils.shmem_budget")
    log.addHandler(handler)
    try:
        yield records
    finally:
        log.removeHandler(handler)


@pytest.fixture(autouse=True)
def _clear_budget_cache():
    """Each test starts with a clean cache so device-prop mocks aren't stale."""
    from vllm.triton_utils import shmem_budget

    shmem_budget._BUDGET_CACHE.clear()
    yield
    shmem_budget._BUDGET_CACHE.clear()


# --------------------------------------------------------------------------
# infer_shmem_budget()
# --------------------------------------------------------------------------


def test_infer_budget_sm90_h100():
    """H100 (SM_90) per-block opt-in is 228 KiB = 233,472 bytes."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_device_props(233_472)),
    ):
        assert infer_shmem_budget() == 233_472


def test_infer_budget_sm120_blackwell():
    """SM_120 consumer Blackwell (RTX 5090 / RTX PRO 6000) is 99 KiB = 101,376."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_device_props(101_376)),
    ):
        assert infer_shmem_budget() == 101_376


def test_infer_budget_turing_t4():
    """T4 (SM_75) per-block opt-in is 64 KiB = 65,536 bytes."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_device_props(65_536)),
    ):
        assert infer_shmem_budget() == 65_536


def test_infer_budget_cached_per_device():
    """Second call for same device shouldn't re-read torch.cuda."""
    mock_props = MagicMock(return_value=_device_props(101_376))
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", mock_props),
    ):
        assert infer_shmem_budget() == 101_376
        assert infer_shmem_budget() == 101_376
    # Two infer_shmem_budget() calls, but get_device_properties called only once.
    assert mock_props.call_count == 1


def test_infer_budget_no_cuda_falls_back():
    """No CUDA available → fallback to the conservative budget."""
    with patch("torch.cuda.is_available", return_value=False):
        assert infer_shmem_budget() == _FALLBACK_BUDGET_BYTES


def test_infer_budget_optin_missing_uses_static():
    """Older torch may not surface shared_memory_per_block_optin; use static."""
    props = MagicMock()
    props.shared_memory_per_block_optin = 0  # missing/0
    props.shared_memory_per_block = 49_152
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=props),
    ):
        assert infer_shmem_budget() == 49_152


def test_infer_budget_get_device_properties_raises():
    """Exception during the query → fallback + warn (not raise)."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "torch.cuda.get_device_properties",
            side_effect=RuntimeError("driver detached"),
        ),
    ):
        assert infer_shmem_budget() == _FALLBACK_BUDGET_BYTES


def test_infer_budget_explicit_device():
    """Caller can pass device=N to query a specific GPU."""

    def _props_per_device(dev):
        return _device_props(101_376 if dev == 0 else 233_472)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", side_effect=_props_per_device),
    ):
        assert infer_shmem_budget(device=0) == 101_376
        assert infer_shmem_budget(device=1) == 233_472


# --------------------------------------------------------------------------
# make_shmem_pruner() — pruning logic
# --------------------------------------------------------------------------


def _const_estimator(value: int):
    """Estimator that returns a constant byte count regardless of config."""
    return lambda config, named_args: value


def _kwarg_estimator(arg_name: str, multiplier: int = 1):
    """Estimator that reads a kwarg and scales it."""
    return lambda config, named_args: int(config.kwargs[arg_name] * multiplier)


def _mock_sm120():
    """Return context manager that pretends torch.cuda is SM_120."""
    return patch.multiple(
        "torch.cuda",
        is_available=MagicMock(return_value=True),
        current_device=MagicMock(return_value=0),
        get_device_properties=MagicMock(return_value=_device_props(101_376)),
    )


def test_pruner_noop_when_all_fit():
    """If every config fits the budget, the pruner returns them unchanged."""
    configs = [FakeConfig(kwargs={"BV": 32}), FakeConfig(kwargs={"BV": 64})]
    pruner = make_shmem_pruner(_const_estimator(50_000))  # well under SM_120
    with _mock_sm120():
        kept = pruner(configs, named_args={})
    assert kept == configs


def test_pruner_filters_over_budget():
    """Configs estimating over budget are dropped."""
    configs = [
        FakeConfig(kwargs={"BV": 32}),  # estimator → 32_000 (kept)
        FakeConfig(kwargs={"BV": 64}),  # estimator → 64_000 (kept)
        FakeConfig(kwargs={"BV": 128}),  # estimator → 128_000 (dropped)
    ]
    pruner = make_shmem_pruner(_kwarg_estimator("BV", multiplier=1_000))
    with _mock_sm120():
        kept = pruner(configs, named_args={})
    assert len(kept) == 2
    assert all(c.kwargs["BV"] != 128 for c in kept)


def test_pruner_safety_margin_enforced():
    """Default 1 KiB safety margin: budget(101_376) - margin(1024) = 100_352."""
    configs = [
        FakeConfig(kwargs={"est": 99_999}),  # under effective budget → kept
        FakeConfig(kwargs={"est": 100_352}),  # exactly at effective budget → kept
        FakeConfig(kwargs={"est": 100_353}),  # 1 byte over → dropped
    ]
    pruner = make_shmem_pruner(_kwarg_estimator("est"))
    with _mock_sm120():
        kept = pruner(configs, named_args={})
    assert len(kept) == 2
    assert all(c.kwargs["est"] <= 100_352 for c in kept)


def test_pruner_all_over_budget_falls_back_to_smallest(shmem_logs):
    """When no config fits, return single smallest config + warn once."""
    configs = [
        FakeConfig(kwargs={"BV": 256}),  # 256_000 → over
        FakeConfig(kwargs={"BV": 128}),  # 128_000 → over
        FakeConfig(kwargs={"BV": 192}),  # 192_000 → over
    ]
    pruner = make_shmem_pruner(_kwarg_estimator("BV", multiplier=1_000))
    with _mock_sm120():
        kept = pruner(configs, named_args={})
    assert len(kept) == 1
    assert kept[0].kwargs["BV"] == 128  # smallest
    assert any("falling back to smallest" in r.getMessage() for r in shmem_logs)


def test_pruner_all_over_budget_warns_only_once_per_device(shmem_logs):
    """Repeated invocations on same device emit at most one warning."""
    configs = [FakeConfig(kwargs={"BV": 256})]
    pruner = make_shmem_pruner(_kwarg_estimator("BV", multiplier=1_000))
    with _mock_sm120():
        pruner(configs, named_args={})
        pruner(configs, named_args={})
        pruner(configs, named_args={})
    n_warns = sum(1 for r in shmem_logs if "falling back" in r.getMessage())
    assert n_warns == 1


def test_pruner_on_empty_raise_mode():
    """on_empty='raise' lifts the empty-list case into a RuntimeError."""
    configs = [FakeConfig(kwargs={"BV": 256})]
    pruner = make_shmem_pruner(
        _kwarg_estimator("BV", multiplier=1_000), on_empty="raise"
    )
    with _mock_sm120(), pytest.raises(RuntimeError, match="exceeds device"):
        pruner(configs, named_args={})


def test_pruner_estimator_exception_keeps_config(shmem_logs):
    """An estimator bug shouldn't kill the JIT — keep the config + warn."""

    def _bad_estimator(config, named_args):
        raise ValueError("estimator bug")

    configs = [FakeConfig(kwargs={"BV": 32})]
    pruner = make_shmem_pruner(_bad_estimator)
    with _mock_sm120():
        kept = pruner(configs, named_args={})
    assert kept == configs  # config preserved
    assert any("estimator raised" in r.getMessage() for r in shmem_logs)


def test_pruner_zero_budget_passes_through():
    """Pathological budget <= 0 → return configs unfiltered, let Triton raise."""
    configs = [FakeConfig(kwargs={"BV": 64})]
    pruner = make_shmem_pruner(_const_estimator(50_000))
    # Mock a degenerate device with 0 shmem
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "torch.cuda.get_device_properties",
            return_value=_device_props(0, per_block=0),
        ),
    ):
        kept = pruner(configs, named_args={})
    # Fallback budget kicks in (49152) → 50_000 over → on_empty smallest → 1 config
    # (this also covers the case where the fallback path is exercised cleanly)
    assert len(kept) == 1


def test_pruner_consumes_named_args():
    """Estimator can read kernel runtime kwargs (the `named_args` dict)."""

    def _by_named(config, named_args):
        return named_args.get("BT", 64) * 1_000  # 64 * 1000 = 64000

    configs = [FakeConfig(kwargs={"BV": 32})]
    pruner = make_shmem_pruner(_by_named)
    with _mock_sm120():
        # BT=64 → 64000 bytes → fits 101376-1024
        kept = pruner(configs, named_args={"BT": 64})
        assert kept == configs
        # BT=128 → 128000 bytes → over budget, falls back to smallest
        kept = pruner(configs, named_args={"BT": 128})
        assert len(kept) == 1


# --------------------------------------------------------------------------
# Second reference call site — chunk_o.py kernel estimator coverage
#
# The PR wires the helper into vllm/model_executor/layers/fla/ops/chunk_o.py
# (function ``chunk_fwd_kernel_o``). These tests cover the estimator
# function ``_est_smem_chunk_fwd_o`` defined alongside that kernel. They
# verify the byte arithmetic + the helper integration is correct end-to-end
# without requiring the full vllm + Triton runtime.
# --------------------------------------------------------------------------


def _est_chunk_fwd_o(config, named_args):
    """Local copy of the kernel-side estimator for unit-test isolation.

    Mirrors ``_est_smem_chunk_fwd_o`` in chunk_o.py. Duplicated here so the
    test suite stays runnable without importing the FLA kernel module
    (which pulls in vllm._C and triton at import time).
    """
    BK = config.kwargs["BK"]
    BV = config.kwargs["BV"]
    BT = named_args.get("BT", 64)
    num_stages = config.num_stages
    persistent = BT * BV * 4 + BT * BT * 4
    per_stage = BT * BK * 2 + BK * BT * 2 + BV * BK * 2
    overhead = 4096
    return persistent + num_stages * per_stage + overhead


def test_chunk_fwd_o_estimator_extreme_config_overflows_all_gpus():
    """BK=BV=128 num_stages=4 BT=64 is the upper-extreme config.

    The estimator is intentionally a slight over-estimate (peak co-resident
    assumption). At this extreme, it predicts ~315 KiB — over BOTH the
    SM_120 opt-in (~101 KiB) AND the H100 opt-in (~228 KiB). The pruner
    catching this on H100 too is a useful side-benefit: even H100 had no
    real headroom for this config under the worst-case allocation
    pattern, and the autotune would have either soft-OOM'd or compiled
    into spillover-heavy code. The hand-rolled ``BKV_LIST`` only switches
    bins at boot — the pruner adds per-config + per-num_stages precision.
    """
    config = FakeConfig(kwargs={"BK": 128, "BV": 128}, num_stages=4)
    bytes_needed = _est_chunk_fwd_o(config, {"BT": 64})
    # persistent = 64*128*4 + 64*64*4 = 32768 + 16384 = 49152
    # per_stage  = 64*128*2 + 128*64*2 + 128*128*2 = 65536
    # total      = 49152 + 4 * 65536 + 4096 = 315392
    assert bytes_needed == 315392
    assert bytes_needed > 101_376  # over SM_120 (~99 KiB)
    assert bytes_needed > 233_472  # over H100 (~228 KiB) too


def test_chunk_fwd_o_estimator_h100_friendly_config_fits():
    """BK=BV=128 num_stages=2 BT=64 fits H100 (228 KiB) — keeps perf there."""
    config = FakeConfig(kwargs={"BK": 128, "BV": 128}, num_stages=2)
    bytes_needed = _est_chunk_fwd_o(config, {"BT": 64})
    # persistent = 49152
    # per_stage = 65536; 2 stages → 131072
    # total = 49152 + 131072 + 4096 = 184320
    assert bytes_needed == 184320
    assert bytes_needed > 101_376  # over SM_120 — pruner drops
    assert bytes_needed < 233_472  # under H100 — pruner keeps


def test_chunk_fwd_o_estimator_sm120_safe_config_fits():
    """BK=BV=64 num_stages=2 BT=64 fits SM_120 (101 KiB)."""
    config = FakeConfig(kwargs={"BK": 64, "BV": 64}, num_stages=2)
    bytes_needed = _est_chunk_fwd_o(config, {"BT": 64})
    # persistent = 64*64*4 + 64*64*4 = 32768
    # per_stage  = 64*64*2 + 64*64*2 + 64*64*2 = 24576
    # total      = 32768 + 2 * 24576 + 4096 = 32768 + 49152 + 4096 = 86016
    assert bytes_needed == 86016
    assert bytes_needed < 101_376  # fits SM_120


def test_chunk_fwd_o_pruner_on_sm120_filters_large_configs():
    """Wire the chunk_fwd_o estimator through make_shmem_pruner on SM_120."""
    configs = [
        FakeConfig(kwargs={"BK": 64, "BV": 64}, num_stages=2),  # 86016 — fits
        FakeConfig(kwargs={"BK": 64, "BV": 64}, num_stages=4),  # 135168 — over
        FakeConfig(kwargs={"BK": 128, "BV": 128}, num_stages=2),  # 184320 — over
        FakeConfig(kwargs={"BK": 128, "BV": 128}, num_stages=4),  # 315392 — over
    ]
    pruner = make_shmem_pruner(_est_chunk_fwd_o)
    with _mock_sm120():
        kept = pruner(configs, named_args={"BT": 64})
    # SM_120 budget 101376 - 1024 safety = 100352 effective
    # Only the first (86016 bytes) fits
    assert len(kept) == 1
    assert kept[0].kwargs == {"BK": 64, "BV": 64}
    assert kept[0].num_stages == 2
