# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm kernel correctness tests for AITER unified attention.

Compares ``aiter.ops.triton.unified_attention`` against ``ref_paged_attn`` under
decode, prefill, and mixed batches with varied shapes.
"""

from typing import Any

import pytest
import torch

from tests.kernels.attention.test_triton_unified_attention import ref_paged_attn
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.utils.torch_utils import set_random_seed

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only"),
]

NUM_Q_HEADS = 8
NUM_KV_HEADS = 8
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 64]
DTYPES = [torch.bfloat16, torch.float16]
FP8_DTYPE = current_platform.fp8_dtype()

# (query_len, kv_len) per sequence; kv_len <= 512 (2D kernel path)
MIXED_SEQ_LENS = [
    [(1, 128), (5, 18), (129, 463)],
    [(10, 256), (5, 64), (32, 128)],
]
DECODE_SEQ_LENS = [
    [(1, 128), (1, 256), (1, 384), (1, 512)],
]

DEFAULT_ATOL, DEFAULT_RTOL = 1.5e-2, 1e-2
FP8_KV_ATOL, FP8_KV_RTOL = 1.5e-1, 1.5e-1

# kv_cache_dtype, k_scale, v_scale, atol, rtol
KV_CACHE_CONFIGS = [
    pytest.param(None, 1.0, 1.0, DEFAULT_ATOL, DEFAULT_RTOL, id="native"),
    pytest.param(
        FP8_DTYPE,
        0.5,
        0.25,
        FP8_KV_ATOL,
        FP8_KV_RTOL,
        id="fp8_kv",
        marks=pytest.mark.skipif(
            not current_platform.supports_fp8(),
            reason="FP8 not supported on this hardware",
        ),
    ),
]


def _require_aiter() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("aiter is required on supported ROCm hardware for this test")


def _make_case(
    *,
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    num_blocks: int = 2048,
    kv_cache_dtype: torch.dtype | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> dict[str, Any]:
    torch.set_default_device("cuda")

    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_seqs = len(seq_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), NUM_Q_HEADS, head_size, dtype=dtype)
    if kv_cache_dtype is None:
        key_cache = torch.randn(
            num_blocks, block_size, NUM_KV_HEADS, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
    else:
        key_cache = torch.clamp(
            torch.randn(num_blocks, block_size, NUM_KV_HEADS, head_size),
            -1.0,
            1.0,
        ).to(kv_cache_dtype)
        value_cache = torch.clamp(
            torch.randn(num_blocks, block_size, NUM_KV_HEADS, head_size),
            -1.0,
            1.0,
        ).to(kv_cache_dtype)

    cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )

    descale_shape = (num_seqs, NUM_KV_HEADS)
    k_descale = torch.full(descale_shape, k_scale, dtype=torch.float32, device="cuda")
    v_descale = torch.full(descale_shape, v_scale, dtype=torch.float32, device="cuda")

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "seq_lens_tensor": seq_lens_tensor,
        "cu_seqlens_q": cu_seqlens_q,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "scale": scale,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
        "query_dtype": dtype,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def _run_aiter_unified_attention(case: dict[str, Any]) -> torch.Tensor:
    from aiter.ops.triton.unified_attention import unified_attention

    output = torch.empty_like(case["query"])
    unified_attention(
        q=case["query"],
        k=case["key_cache"],
        v=case["value_cache"],
        out=output,
        cu_seqlens_q=case["cu_seqlens_q"],
        max_seqlen_q=case["max_query_len"],
        seqused_k=case["seq_lens_tensor"],
        max_seqlen_k=case["max_kv_len"],
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


def _ref_output(case: dict[str, Any]) -> torch.Tensor:
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    if key_cache.dtype != case["query_dtype"]:
        key_cache = key_cache.to(case["query_dtype"]) * case["k_scale"]
        value_cache = value_cache.to(case["query_dtype"]) * case["v_scale"]

    return ref_paged_attn(
        query=case["query"],
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=case["query_lens"],
        kv_lens=case["kv_lens"],
        block_tables=case["block_tables"],
        scale=case["scale"],
    )


def _assert_matches_reference(
    case: dict[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    output = _run_aiter_unified_attention(case)
    output_ref = _ref_output(case)
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_lens", MIXED_SEQ_LENS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype,k_scale,v_scale,atol,rtol", KV_CACHE_CONFIGS)
@torch.inference_mode()
def test_aiter_unified_attn_mixed_batch(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: torch.dtype | None,
    k_scale: float,
    v_scale: float,
    atol: float,
    rtol: float,
) -> None:
    """Decode + prefill sequences in one batch."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    _assert_matches_reference(case, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_lens", DECODE_SEQ_LENS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype,k_scale,v_scale,atol,rtol", KV_CACHE_CONFIGS)
@torch.inference_mode()
def test_aiter_unified_attn_decode(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: torch.dtype | None,
    k_scale: float,
    v_scale: float,
    atol: float,
    rtol: float,
) -> None:
    """Single-token decode."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    _assert_matches_reference(case, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(256, 256), (128, 512)],
        [(64, 128), (32, 256), (16, 512)],
    ],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_cache_dtype,k_scale,v_scale,atol,rtol", KV_CACHE_CONFIGS)
@torch.inference_mode()
def test_aiter_unified_attn_prefill(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    kv_cache_dtype: torch.dtype | None,
    k_scale: float,
    v_scale: float,
    atol: float,
    rtol: float,
) -> None:
    """Prefill-only batches with query_len > 1."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    _assert_matches_reference(case, atol=atol, rtol=rtol)
