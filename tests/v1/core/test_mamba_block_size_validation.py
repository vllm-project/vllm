# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""validate_block_size mamba-mode scheduler-budget gating.

Align mode splits prefill chunks on block boundaries, so its constraints key
on block_size. All mode with a kernel-chunk-aligned prefill split (GDN/FLA:
``mamba_all_mode_prefill_align_size`` set) clips chunk ends to align-size
multiples instead, so its constraints key on the (much smaller) align size —
a budget below it would floor every prefill step to 0 tokens and hang. All
mode WITHOUT the field (SSD-style kernels with in-kernel realignment) has no
split and therefore no scheduler-budget constraint.
"""

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig


def _fake_cfg(
    mode,
    block_size,
    align_size=None,
    max_num_batched_tokens=32768,
    long_prefill_token_threshold=0,
    disable_chunked_mm_input=False,
):
    """Minimal stand-in exposing exactly the fields validate_block_size reads."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_cache_mode=mode,
            mamba_all_mode_prefill_align_size=align_size,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            dcp_kv_cache_interleave_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
            disable_chunked_mm_input=disable_chunked_mm_input,
        ),
    )


def _validate(cfg):
    # Call unbound so the minimal stand-in can act as self.
    VllmConfig.validate_block_size(cfg)


# --------------------------- REGRESSION (align/none) ---------------------------


def test_align_oversized_block_still_asserts():
    with pytest.raises(AssertionError, match="align mode"):
        _validate(_fake_cfg("align", block_size=65536, max_num_batched_tokens=32768))


def test_align_valid_config_still_passes():
    _validate(_fake_cfg("align", block_size=544, max_num_batched_tokens=32768))


def test_none_mode_unaffected_by_oversized_block():
    _validate(_fake_cfg("none", block_size=65536, max_num_batched_tokens=32768))


# ------------------------------- NEW (all-mode) --------------------------------


def test_all_budget_below_align_size_asserts():
    """all + chunk-aligned split: a budget below the align size hangs the
    scheduler (every mid-prompt bite floors to 0) and must fail validation."""
    with pytest.raises(AssertionError, match="all mode"):
        _validate(
            _fake_cfg("all", block_size=576, align_size=64, max_num_batched_tokens=32)
        )


def test_all_valid_config_passes():
    _validate(
        _fake_cfg("all", block_size=576, align_size=64, max_num_batched_tokens=32768)
    )


def test_all_block_size_may_exceed_budget():
    """Unlike align, all-mode only needs the budget to cover one kernel chunk:
    block_size larger than the budget is fine (blocks fill across steps)."""
    _validate(
        _fake_cfg("all", block_size=65536, align_size=64, max_num_batched_tokens=1024)
    )


def test_all_without_split_field_has_no_budget_constraint():
    """all-mode kernels with in-kernel realignment (SSD-style) never take the
    scheduler split; no align size -> no constraint."""
    _validate(
        _fake_cfg("all", block_size=65536, align_size=None, max_num_batched_tokens=32)
    )


def test_all_long_prefill_threshold_below_align_size_asserts():
    with pytest.raises(AssertionError):
        _validate(
            _fake_cfg(
                "all",
                block_size=576,
                align_size=64,
                long_prefill_token_threshold=32,
            )
        )


def test_all_disable_chunked_mm_input_asserts():
    with pytest.raises(AssertionError, match="Chunked MM input"):
        _validate(
            _fake_cfg(
                "all", block_size=576, align_size=64, disable_chunked_mm_input=True
            )
        )
