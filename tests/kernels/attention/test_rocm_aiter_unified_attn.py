# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm kernel correctness tests for AITER unified attention.

Compares ``aiter.ops.triton.unified_attention`` against ``ref_paged_attn`` under
decode, prefill, and mixed batches with varied shapes.
"""

from typing import Any, Literal

import pytest
import torch

from tests.kernels.attention.test_triton_unified_attention import ref_paged_attn
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

_SKIP_NON_MI3XX = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_mi3xx

    _SKIP_NON_MI3XX = not on_mi3xx()

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-specific tests"),
    pytest.mark.skipif(_SKIP_NON_MI3XX, reason="MI300/MI350 ROCm only"),
]

NUM_Q_HEADS = 8
NUM_KV_HEADS = 8
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 64]
DTYPES = [torch.bfloat16, torch.float16]
FP8_DTYPE = current_platform.fp8_dtype()

# (query_len, kv_len) per sequence
MIXED_SEQ_LENS = [
    [(1, 128), (5, 18), (129, 463)],
    [(10, 256), (5, 64), (32, 128)],
    [(1, 1024), (5, 18), (129, 1328)],
]
DECODE_SEQ_LENS = [
    [(1, 128), (1, 256), (1, 384), (1, 512)],
    [(1, 1024), (1, 1536), (1, 2048)],
]
PREFILL_SEQ_LENS = [
    [(256, 256), (128, 512)],
    [(64, 128), (32, 256), (16, 512)],
    [(256, 1024), (128, 2048)],
]

DEFAULT_ATOL, DEFAULT_RTOL = 1.5e-2, 1e-2
FP8_ATOL, FP8_RTOL = 1.5e-1, 1.5e-1
# Non-unity scale so q_descale handling is exercised explicitly.
Q_SCALE = 0.75
K_SCALE, V_SCALE = 0.5, 0.25

Fp8Variant = Literal["fp8_kv", "fp8_query", "fp8_query_kv"]

FP8_VARIANTS = [
    pytest.param("fp8_kv", id="fp8_kv"),
    pytest.param("fp8_query", id="fp8_query"),
    pytest.param("fp8_query_kv", id="fp8_query_kv"),
]

FP8_SEQ_LENS = [
    MIXED_SEQ_LENS[0],
    DECODE_SEQ_LENS[0],
    DECODE_SEQ_LENS[1],
    PREFILL_SEQ_LENS[0],
    PREFILL_SEQ_LENS[2],
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
    q_dtype: torch.dtype | None = None,
    q_scale: float = Q_SCALE,
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

    kernel_query = query
    q_descale = None
    if q_dtype is not None:
        q_descale = torch.tensor(q_scale, dtype=torch.float32, device="cuda")
        kernel_query = (query / q_scale).to(q_dtype)

    return {
        "query": query,
        "kernel_query": kernel_query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "seq_lens_tensor": seq_lens_tensor,
        "cu_seqlens_q": cu_seqlens_q,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "scale": scale,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
        "query_dtype": dtype,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def _make_fp8_case(
    *,
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    variant: Fp8Variant,
) -> dict[str, Any]:
    use_fp8_kv = variant in ("fp8_kv", "fp8_query_kv")
    use_fp8_query = variant in ("fp8_query", "fp8_query_kv")
    return _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=FP8_DTYPE if use_fp8_kv else None,
        k_scale=K_SCALE if use_fp8_kv else 1.0,
        v_scale=V_SCALE if use_fp8_kv else 1.0,
        q_dtype=FP8_DTYPE if use_fp8_query else None,
    )


def _run_aiter_unified_attention(case: dict[str, Any]) -> torch.Tensor:
    from aiter.ops.triton.unified_attention import unified_attention

    kernel_query = case["kernel_query"]
    # Kernel writes high-precision output even when Q is FP8 (matches vLLM usage).
    output = torch.empty_like(case["query"])
    unified_attention(
        q=kernel_query,
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
        q_descale=case["q_descale"],
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
@torch.inference_mode()
def test_aiter_unified_attn_mixed_batch(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """Decode + prefill sequences in one batch (native dtypes)."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
    )
    _assert_matches_reference(case)


@pytest.mark.parametrize("seq_lens", DECODE_SEQ_LENS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_aiter_unified_attn_decode(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """Single-token decode (native dtypes)."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
    )
    _assert_matches_reference(case)


@pytest.mark.parametrize("seq_lens", PREFILL_SEQ_LENS)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16])
@torch.inference_mode()
def test_aiter_unified_attn_prefill(
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
) -> None:
    """Prefill-only batches with query_len > 1 (native dtypes)."""
    _require_aiter()
    set_random_seed(0)

    case = _make_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        dtype=torch.bfloat16,
    )
    _assert_matches_reference(case)


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.parametrize("variant", FP8_VARIANTS)
@pytest.mark.parametrize("seq_lens", FP8_SEQ_LENS)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16, 64])
@torch.inference_mode()
def test_aiter_unified_attn_fp8(
    variant: Fp8Variant,
    seq_lens: list[tuple[int, int]],
    head_size: int,
    block_size: int,
) -> None:
    """FP8 KV cache, FP8 query, or both; compared at bf16 reference precision."""
    _require_aiter()
    set_random_seed(0)

    case = _make_fp8_case(
        seq_lens=seq_lens,
        head_size=head_size,
        block_size=block_size,
        variant=variant,
    )
    _assert_matches_reference(case, atol=FP8_ATOL, rtol=FP8_RTOL)
