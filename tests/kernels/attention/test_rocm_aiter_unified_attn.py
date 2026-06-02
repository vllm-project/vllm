# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-specific tests for the AITER unified-attention backend.

This file owns ROCm-only wiring for unified attention:
- backend static contract (block/head sizes, attention types)
- env gate for Triton unified attention
- representative decode correctness and determinism

Broader kernel coverage lives in
``tests/kernels/attention/test_triton_unified_attention.py``.
Backend selection lives in
``tests/v1/attention/test_rocm_attention_backends_selection.py``.
"""

import importlib
from typing import Any

import pytest
import torch

from tests.kernels.attention.test_triton_unified_attention import ref_paged_attn
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

requires_mi3xx = pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")

NUM_BLOCKS = 1024
FP8_DTYPE = current_platform.fp8_dtype()
DECODE_ATOL, DECODE_RTOL = 1.5e-2, 1e-2
FP8_ATOL, FP8_RTOL = 0.02, 0.0

DECODE_CASES = [
    pytest.param(64, torch.float16, 16, id="fp16_h64_b16"),
    pytest.param(128, torch.bfloat16, 64, id="bf16_h128_b64"),
]


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _require_aiter() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


def _run_aiter_unified_attention(case: dict[str, Any], *, max_seqlen_q: int = 1):
    from aiter.ops.triton.unified_attention import unified_attention

    output = torch.zeros_like(case["query"])
    unified_attention(
        q=case["query"],
        k=case["key_cache"],
        v=case["value_cache"],
        out=output,
        cu_seqlens_q=case["cu_seqlens_q"],
        max_seqlen_q=max_seqlen_q,
        seqused_k=case["seq_lens_tensor"],
        max_seqlen_k=case["max_seq_len"],
        softmax_scale=case["scale"],
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=case["block_tables"],
        softcap=0,
        q_descale=None,
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        sinks=None,
        output_scale=None,
    )
    return output


def _make_decode_case(
    *,
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    seq_lens: list[int],
    kv_cache_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    torch.set_default_device("cuda")
    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = len(seq_lens)
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    if kv_cache_dtype is None:
        key_cache = torch.randn(
            NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
    else:
        key_cache = torch.clamp(
            torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size),
            -1.0,
            1.0,
        ).to(kv_cache_dtype)
        value_cache = torch.clamp(
            torch.randn(NUM_BLOCKS, block_size, num_kv_heads, head_size),
            -1.0,
            1.0,
        ).to(kv_cache_dtype)

    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "seq_lens_tensor": seq_lens_tensor,
        "cu_seqlens_q": cu_seqlens_q,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "scale": scale,
        "max_seq_len": max_seq_len,
        "query_dtype": dtype,
    }


def _ref_decode_output(case: dict[str, Any]) -> torch.Tensor:
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    if key_cache.dtype != case["query_dtype"]:
        key_cache = key_cache.to(case["query_dtype"])
        value_cache = value_cache.to(case["query_dtype"])
    return ref_paged_attn(
        query=case["query"],
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * len(case["seq_lens"]),
        kv_lens=case["seq_lens"],
        block_tables=case["block_tables"],
        scale=case["scale"],
    )


def test_unified_attn_backend_contract():
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    backend = RocmAiterUnifiedAttentionBackend
    assert backend.get_preferred_block_size(16) == 64
    assert backend.supports_block_size(16)
    assert backend.supports_block_size(64)
    assert not backend.supports_block_size(15)
    assert backend.supports_head_size(32)
    assert backend.supports_head_size(256)
    assert not backend.supports_head_size(16)
    assert backend.supports_mm_prefix()
    assert backend.supports_sink()
    assert backend.forward_includes_kv_cache_update is False
    for attn_type in (
        AttentionType.DECODER,
        AttentionType.ENCODER,
        AttentionType.ENCODER_ONLY,
        AttentionType.ENCODER_DECODER,
    ):
        assert backend.supports_attn_type(attn_type)


def test_unified_attn_backend_validates_kv_cache_block_size():
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.get_kv_cache_shape(8, 16, 8, 128) == (
        2,
        8,
        16,
        8,
        128,
    )
    with pytest.raises(ValueError, match="Block size must be a multiple of 16"):
        RocmAiterUnifiedAttentionBackend.get_kv_cache_shape(8, 15, 8, 128)


@requires_mi3xx
@pytest.mark.parametrize(
    ("use_aiter", "use_unified", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_unified_attn_env_flags_control_enablement(
    use_aiter, use_unified, expected, monkeypatch
):
    from vllm._aiter_ops import rocm_aiter_ops

    _require_aiter()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv(
            "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION",
            "1" if use_unified else "0",
        )
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_triton_unified_attn_enabled() is expected

    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


@requires_mi3xx
@pytest.mark.parametrize(("head_size", "dtype", "block_size"), DECODE_CASES)
def test_unified_attn_decode_matches_reference(head_size, dtype, block_size):
    _require_aiter()
    torch.manual_seed(0)

    case = _make_decode_case(
        head_size=head_size,
        dtype=dtype,
        block_size=block_size,
        seq_lens=[128, 256, 384, 512],
    )
    output = _run_aiter_unified_attention(case)
    output_ref = _ref_decode_output(case)
    torch.testing.assert_close(output, output_ref, atol=DECODE_ATOL, rtol=DECODE_RTOL)


@requires_mi3xx
def test_unified_attn_decode_determinism():
    _require_aiter()
    torch.manual_seed(3)

    case = _make_decode_case(
        head_size=128,
        dtype=torch.bfloat16,
        block_size=16,
        seq_lens=[64, 128],
    )
    reference = _run_aiter_unified_attention(case)
    for run in range(1, 4):
        output = _run_aiter_unified_attention(case)
        assert torch.equal(reference, output), (
            f"Run {run}: decode output differs from run 0 "
            f"(max diff = {(output - reference).abs().max().item():.2e})"
        )


@requires_mi3xx
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.parametrize("head_size", [64, 128])
def test_unified_attn_decode_fp8_kv_cache(head_size):
    _require_aiter()
    torch.manual_seed(5)

    case = _make_decode_case(
        head_size=head_size,
        dtype=torch.bfloat16,
        block_size=16,
        seq_lens=[128, 256],
        kv_cache_dtype=FP8_DTYPE,
    )
    output = _run_aiter_unified_attention(case)
    output_ref = _ref_decode_output(case)
    torch.testing.assert_close(output, output_ref, atol=FP8_ATOL, rtol=FP8_RTOL)
