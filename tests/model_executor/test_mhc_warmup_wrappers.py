# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the mHC VllmJitKernel wrappers.

These tests do not require CUDA / TileLang. They exercise only the
dispatch / get_warmup_keys path (AST tracer + dedup logic) by stubbing
the deep_gemm / compute_mhc_dispatch dependencies, and verify the
compile-key set is the expected sparse subset rather than the full
token range.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.warmup import (
    HC_HEAD_FUSED_KERNEL,
    MHC_FUSED_POST_PRE_KERNEL,
    MHC_PRE_KERNEL,
    HcHeadFusedKernel,
    MhcFusedPostPreKernel,
    MhcKernelConstants,
    MhcPreKernel,
)
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------

# Mock a 132-SM GPU (H100). compute_mhc_dispatch mirrors the real heuristic
# minus the torch.cuda.get_device_properties call so it runs without CUDA.
N_SMS = 132


def _fake_compute_mhc_dispatch(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    use_deep_gemm: bool,
    *,
    is_broadcast: bool = False,
    is_fused: bool = False,
):
    """CPU-only mock of compute_mhc_dispatch (no CUDA device query)."""
    hc_hidden_size = hc_mult * hidden_size

    if is_fused:
        use_small_fma = num_tokens <= 16
        if use_small_fma:
            tile_n = 2 if num_tokens < 8 else 3
            n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
        else:
            tile_n = 1
            if use_deep_gemm:
                grid_size = max(1, math.ceil(num_tokens / 64))
                k = hc_hidden_size
                n_splits = N_SMS // grid_size
                num_block_k = math.ceil(k / 64)
                n_splits = min(n_splits, num_block_k // 4)
                n_splits = max(n_splits, 1)
            else:
                n_splits = 1
    elif is_broadcast:
        use_small_fma = False
        tile_n = 1
        grid_size = max(1, math.ceil(num_tokens / 64))
        k = hidden_size
        n_splits = N_SMS // grid_size
        num_block_k = math.ceil(k / 64)
        n_splits = min(n_splits, num_block_k // 4)
        n_splits = max(n_splits, 1)
    else:
        use_small_fma = False
        tile_n = 1
        if use_deep_gemm:
            grid_size = max(1, math.ceil(num_tokens / 64))
            k = hc_hidden_size
            n_splits = N_SMS // grid_size
            num_block_k = math.ceil(k / 64)
            n_splits = min(n_splits, num_block_k // 4)
            n_splits = max(n_splits, 1)
        else:
            n_splits = 1

    return SimpleNamespace(
        n_splits=n_splits, tile_n=tile_n, use_small_fma=use_small_fma
    )


def _vllm_config(max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )


_DEFAULT_CONSTANTS = MhcKernelConstants(
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    norm_eps=1e-6,
)


@pytest.fixture
def _patch_deep_gemm():
    """Stub deep_gemm support + compute_mhc_dispatch so tests are CPU-only."""
    with (
        mock.patch(
            "vllm.model_executor.kernels.mhc.warmup._is_deep_gemm_supported",
            return_value=True,
        ),
        mock.patch(
            "vllm.model_executor.kernels.mhc.warmup._compute_mhc_dispatch",
            side_effect=_fake_compute_mhc_dispatch,
        ),
    ):
        yield


# -----------------------------------------------------------------------------
# CompileKey shape tests
# -----------------------------------------------------------------------------


def test_mhc_pre_kernel_compile_key_fields() -> None:
    fields = {f.name for f in dataclasses.fields(MhcPreKernel.CompileKey)}
    assert fields == {
        "hidden_size",
        "hc_mult",
        "n_splits",
        "use_norm_weight",
        "use_deep_gemm",
        "is_broadcast",
    }


def test_mhc_fused_post_pre_kernel_compile_key_fields() -> None:
    fields = {f.name for f in dataclasses.fields(MhcFusedPostPreKernel.CompileKey)}
    assert fields == {
        "hidden_size",
        "hc_mult",
        "n_splits",
        "tile_n",
        "use_small_fma",
        "use_norm_weight",
        "use_deep_gemm",
    }


def test_hc_head_fused_kernel_compile_key_fields() -> None:
    fields = {f.name for f in dataclasses.fields(HcHeadFusedKernel.CompileKey)}
    assert fields == {"hidden_size", "hc_mult"}


# -----------------------------------------------------------------------------
# MhcKernelConstants tests
# -----------------------------------------------------------------------------


def test_mhc_kernel_constants_fields() -> None:
    fields = {f.name for f in dataclasses.fields(MhcKernelConstants)}
    assert fields == {
        "hc_post_mult_value",
        "sinkhorn_repeat",
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "norm_eps",
    }


# -----------------------------------------------------------------------------
# Compile-key enumeration tests
# -----------------------------------------------------------------------------


def test_mhc_pre_kernel_dedupes_token_range_to_n_splits_set(
    _patch_deep_gemm: object,
) -> None:
    """A 16k token range must collapse to ~36 keys, not 16384 keys."""
    cfg = _vllm_config(max_tokens=16384)
    keys = MHC_PRE_KERNEL.get_warmup_keys(
        cast(VllmConfig, cfg),
        hidden_size=4096,
        hc_mult=4,
        use_norm_weight=True,
        is_broadcast_values=[False, True],
        constants=_DEFAULT_CONSTANTS,
    )
    assert len(keys) < 64, f"expected sparse key set, got {len(keys)}"
    assert len(keys) > 0

    for k in keys:
        assert k.hidden_size == 4096
        assert k.hc_mult == 4
        assert k.use_norm_weight is True
        assert k.use_deep_gemm is True
        assert k.n_splits >= 1

    broadcast_keys = [k for k in keys if k.is_broadcast]
    non_broadcast_keys = [k for k in keys if not k.is_broadcast]
    assert broadcast_keys, "expected broadcast keys"
    assert non_broadcast_keys, "expected non-broadcast keys"
    assert len(keys) == len(non_broadcast_keys) + len(broadcast_keys)


def test_mhc_pre_kernel_no_broadcast_keys_when_no_broadcast(
    _patch_deep_gemm: object,
) -> None:
    """When is_broadcast_values=[False], no broadcast keys."""
    cfg = _vllm_config(max_tokens=256)
    keys = MHC_PRE_KERNEL.get_warmup_keys(
        cast(VllmConfig, cfg),
        hidden_size=4096,
        hc_mult=4,
        use_norm_weight=False,
        is_broadcast_values=[False],
        constants=_DEFAULT_CONSTANTS,
    )
    assert keys, "expected at least one key"
    assert not any(k.is_broadcast for k in keys)


def test_mhc_fused_post_pre_kernel_covers_small_fma_and_big_path(
    _patch_deep_gemm: object,
) -> None:
    """The fused wrapper must cover small_fma and big path."""
    cfg = _vllm_config(max_tokens=16384)
    keys = MHC_FUSED_POST_PRE_KERNEL.get_warmup_keys(
        cast(VllmConfig, cfg),
        hidden_size=4096,
        hc_mult=4,
        use_norm_weight=True,
        constants=_DEFAULT_CONSTANTS,
    )

    small_fma_keys = [k for k in keys if k.use_small_fma]
    big_path_keys = [k for k in keys if not k.use_small_fma]
    assert small_fma_keys, "missing use_small_fma=True key"
    assert big_path_keys, "missing use_small_fma=False key"

    small_n_splits = {k.n_splits for k in small_fma_keys}
    assert 8 in small_n_splits, f"missing n_splits=8, got {small_n_splits}"
    assert 4 in small_n_splits, f"missing n_splits=4, got {small_n_splits}"

    small_tile_ns = {k.tile_n for k in small_fma_keys}
    assert small_tile_ns == {2, 3}, f"expected tile_n in {{2, 3}}, got {small_tile_ns}"


def test_hc_head_fused_kernel_dedupes_to_single_key(
    _patch_deep_gemm: object,
) -> None:
    """hc_head deduplicates to exactly one key."""
    cfg = _vllm_config(max_tokens=16384)
    keys = HC_HEAD_FUSED_KERNEL.get_warmup_keys(
        cast(VllmConfig, cfg),
        hidden_size=7168,
        hc_mult=4,
        use_norm_weight=True,
        constants=_DEFAULT_CONSTANTS,
    )
    assert len(keys) == 1, f"expected 1 key, got {len(keys)}: {keys}"
    assert keys[0] == HcHeadFusedKernel.CompileKey(
        hidden_size=7168,
        hc_mult=4,
    )


# -----------------------------------------------------------------------------
# CompileKey is frozen + hashable
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper,kwargs",
    [
        (
            MHC_PRE_KERNEL,
            dict(
                hidden_size=7168,
                hc_mult=4,
                use_norm_weight=True,
                use_deep_gemm=True,
                num_tokens=128,
                is_broadcast=False,
            ),
        ),
        (
            MHC_FUSED_POST_PRE_KERNEL,
            dict(
                hidden_size=4096,
                hc_mult=4,
                use_norm_weight=True,
                use_deep_gemm=True,
                num_tokens=8,
            ),
        ),
        (
            HC_HEAD_FUSED_KERNEL,
            dict(
                hidden_size=7168,
                hc_mult=4,
                use_norm_weight=True,
                num_tokens=4,
            ),
        ),
    ],
)
def test_compile_key_is_frozen_and_hashable(
    _patch_deep_gemm: object, wrapper: VllmJitKernel, kwargs: dict
) -> None:
    """CompileKey must be frozen + hashable so dedup works."""
    key = wrapper.dispatch(**kwargs)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        key.hidden_size = 0  # type: ignore[misc]
    assert {key, replace(key)} == {key}


# -----------------------------------------------------------------------------
# VllmJitKernel contract: dispatch matches get_warmup_keys expansion
# -----------------------------------------------------------------------------


def test_dispatch_matches_get_warmup_keys_expansion(
    _patch_deep_gemm: object,
) -> None:
    """For every token size t in [1, max_tokens], dispatch(t) must produce a
    key that appears in get_warmup_keys()."""
    cfg = _vllm_config(max_tokens=256)
    keys = MHC_FUSED_POST_PRE_KERNEL.get_warmup_keys(
        cast(VllmConfig, cfg),
        hidden_size=4096,
        hc_mult=4,
        use_norm_weight=True,
        constants=_DEFAULT_CONSTANTS,
    )
    key_set = set(keys)
    for t in range(1, 257):
        k = MHC_FUSED_POST_PRE_KERNEL.dispatch(
            num_tokens=t,
            hidden_size=4096,
            hc_mult=4,
            use_norm_weight=True,
            use_deep_gemm=True,
        )
        assert k in key_set, f"token={t} produced key {k} not in warmup set"
