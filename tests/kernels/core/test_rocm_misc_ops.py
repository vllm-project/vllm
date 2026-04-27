# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm platform and env-var checks not owned by the backend-specific files.

The following tests check:
- architecture predicates and platform capability
- active ROCm env vars that still feed runtime code
- the ROCm custom paged-attention eligibility predicate

Direct attention kernel numerics live in the attention test files. The ROCm
AITER flash-attention backend behavior lives in
``tests/kernels/attention/test_rocm_aiter_fa.py``.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# Small helpers -----------------------------------------------------------


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _set_rocm_arch(monkeypatch: pytest.MonkeyPatch, gcn_arch: str):
    import vllm.platforms.rocm as rocm_platform

    monkeypatch.setattr(rocm_platform, "_GCN_ARCH", gcn_arch)
    monkeypatch.setattr(
        rocm_platform,
        "_ON_GFX1X",
        any(gfx in gcn_arch for gfx in ["gfx11", "gfx12"]),
    )
    monkeypatch.setattr(rocm_platform, "_ON_GFX12X", "gfx12" in gcn_arch)
    monkeypatch.setattr(
        rocm_platform,
        "_ON_MI3XX",
        any(gfx in gcn_arch for gfx in ["gfx942", "gfx950"]),
    )
    monkeypatch.setattr(
        rocm_platform,
        "_ON_GFX9",
        any(gfx in gcn_arch for gfx in ["gfx90a", "gfx942", "gfx950"]),
    )
    monkeypatch.setattr(rocm_platform, "_ON_GFX90A", "gfx90a" in gcn_arch)
    monkeypatch.setattr(rocm_platform, "_ON_GFX942", "gfx942" in gcn_arch)
    monkeypatch.setattr(rocm_platform, "_ON_GFX950", "gfx950" in gcn_arch)
    rocm_platform.use_rocm_custom_paged_attention.cache_clear()
    return rocm_platform


# Env vars with active runtime hooks -------------------------------------


def test_shuffle_kv_cache_env_default():
    """The shuffled KV-cache env default should remain disabled until
    explicitly requested."""
    import vllm.envs as envs

    assert envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT is False


@pytest.mark.parametrize("enabled", [True, False])
def test_shuffle_kv_cache_env_propagates_to_rocm_aiter_ops(enabled, monkeypatch):
    """The shuffled KV-cache env should propagate through env reload and the
    cached AITER state refresh."""
    import vllm.envs as envs
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1" if enabled else "0")
    importlib.reload(envs)
    rocm_aiter_ops.refresh_env_variables()

    assert envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT is enabled
    assert rocm_aiter_ops.is_shuffle_kv_cache_enabled() is enabled


def test_sleep_mem_chunk_size_default():
    """The public ROCm sleep-memory chunk env should keep its documented
    default value."""
    import vllm.envs as envs

    assert envs.VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE == 256


@pytest.mark.parametrize("chunk_size", [128, 256, 512, 1024])
def test_sleep_mem_chunk_size_env_parses_ints(chunk_size, monkeypatch):
    """The ROCm sleep-memory chunk env should parse integer overrides
    predictably."""
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE", str(chunk_size))
    importlib.reload(envs)

    assert chunk_size == envs.VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE


def test_skinny_gemm_env_default():
    """The ROCm skinny-GEMM env should stay enabled by default."""
    import vllm.envs as envs

    assert envs.VLLM_ROCM_USE_SKINNY_GEMM is True


@pytest.mark.parametrize("enabled", [True, False])
def test_skinny_gemm_env_controls_rocm_fp8_scaled_mm_support(
    enabled,
    monkeypatch,
):
    """The skinny-GEMM env should directly gate the ROCm FP8 scaled-mm kernel
    eligibility check."""
    import vllm.envs as envs
    from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
        ROCmFP8ScaledMMLinearKernel,
    )

    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1" if enabled else "0")
    importlib.reload(envs)
    monkeypatch.setattr("vllm.platforms.rocm.on_mi3xx", lambda: True)

    supported, reason = ROCmFP8ScaledMMLinearKernel.is_supported()

    assert supported is enabled
    if enabled:
        assert reason is None
    else:
        assert reason == "requires VLLM_ROCM_USE_SKINNY_GEMM to be enabled."


# ROCm architecture and platform contracts -------------------------------


@pytest.mark.parametrize(
    (
        "gcn_arch",
        "expect_on_gfx9",
        "expect_on_mi3xx",
        "expect_on_gfx942",
        "expect_on_gfx950",
        "expect_supports_mx",
        "expect_supports_fp8",
        "expect_custom_allreduce",
        "expect_fp8_dtype",
    ),
    [
        pytest.param(
            "gfx90a",
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            torch.float8_e4m3fn,
            id="gfx90a",
        ),
        pytest.param(
            "gfx942",
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            torch.float8_e4m3fnuz,
            id="gfx942",
        ),
        pytest.param(
            "gfx950",
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            torch.float8_e4m3fn,
            id="gfx950",
        ),
        pytest.param(
            "gfx1100",
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            torch.float8_e4m3fn,
            id="gfx1100",
        ),
        pytest.param(
            "gfx1201",
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            torch.float8_e4m3fn,
            id="gfx1201",
        ),
    ],
)
def test_rocm_architecture_contracts(
    gcn_arch,
    expect_on_gfx9,
    expect_on_mi3xx,
    expect_on_gfx942,
    expect_on_gfx950,
    expect_supports_mx,
    expect_supports_fp8,
    expect_custom_allreduce,
    expect_fp8_dtype,
    monkeypatch,
):
    """The ROCm arch predicates and platform capability helpers should stay in
    sync for the supported arch families."""
    rocm_platform = _set_rocm_arch(monkeypatch, gcn_arch)

    assert rocm_platform.on_gfx9() is expect_on_gfx9
    assert rocm_platform.on_mi3xx() is expect_on_mi3xx
    assert rocm_platform.on_gfx942() is expect_on_gfx942
    assert rocm_platform.on_gfx950() is expect_on_gfx950

    assert current_platform.supports_mx() is expect_supports_mx
    assert current_platform.supports_fp8() is expect_supports_fp8
    assert current_platform.use_custom_allreduce() is expect_custom_allreduce
    assert current_platform.fp8_dtype() == expect_fp8_dtype


# Custom paged-attention eligibility -------------------------------------


def test_rocm_custom_paged_attention_gfx9_supported_case(monkeypatch):
    """The gfx9 custom paged-attention predicate should accept its documented
    fast-path configuration."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx942")

    assert rocm_platform.use_rocm_custom_paged_attention(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"head_size": 256}, id="reject-head-size"),
        pytest.param({"block_size": 8}, id="reject-block-size"),
        pytest.param({"gqa_ratio": 17}, id="reject-gqa"),
        pytest.param({"sliding_window": 128}, id="reject-sliding-window"),
        pytest.param({"max_seq_len": 128 * 1024 + 16}, id="reject-seq-len"),
        pytest.param({"sinks": torch.ones(1)}, id="reject-sinks"),
    ],
)
def test_rocm_custom_paged_attention_gfx9_rejects_unsupported_cases(
    kwargs,
    monkeypatch,
):
    """The gfx9 custom paged-attention predicate should reject unsupported
    shapes and features instead of over-claiming support."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx950")

    params = dict(
        qtype=torch.float16,
        head_size=64,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        sinks=None,
    )
    params.update(kwargs)

    assert not rocm_platform.use_rocm_custom_paged_attention(**params)


def test_rocm_custom_paged_attention_gfx1x_supported_case(monkeypatch):
    """The gfx1x custom paged-attention predicate should accept the narrower
    RDNA fast-path it advertises."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx1201")

    assert rocm_platform.use_rocm_custom_paged_attention(
        qtype=torch.float16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"head_size": 64}, id="reject-head-size"),
        pytest.param({"gqa_ratio": 2}, id="reject-gqa"),
        pytest.param({"kv_cache_dtype": "fp8"}, id="reject-kv-cache-dtype"),
        pytest.param({"alibi_slopes": torch.ones(8)}, id="reject-alibi"),
        pytest.param({"sinks": torch.ones(1)}, id="reject-sinks"),
    ],
)
def test_rocm_custom_paged_attention_gfx1x_rejects_unsupported_cases(
    kwargs,
    monkeypatch,
):
    """The gfx1x custom paged-attention predicate should stay honest about the
    restrictions in the RDNA path."""
    rocm_platform = _set_rocm_arch(monkeypatch, "gfx1100")

    params = dict(
        qtype=torch.bfloat16,
        head_size=128,
        block_size=16,
        gqa_ratio=4,
        max_seq_len=4096,
        sliding_window=0,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        sinks=None,
    )
    params.update(kwargs)

    assert not rocm_platform.use_rocm_custom_paged_attention(**params)
