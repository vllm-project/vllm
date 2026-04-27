# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-specific tests for the AITER Flash Attention backend.

This file owns the ROCm backend contract and the ROCm-specific execution paths
wired through ``vllm.v1.attention.backends.rocm_aiter_fa``:
- backend contract and env gating
- representative prefill, multi-batch, decode, and FP8-KV execution
- direct kernel stress for large block tables and sliding-window masking
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

DTYPES = [torch.bfloat16, torch.float16]
HEAD_SIZES = [64, 128, 256]
NUM_HEADS_PAIRS = [(8, 8), (16, 4)]  # (num_q_heads, num_kv_heads) - tests GQA
BLOCK_SIZE = 16
NUM_BLOCKS = 2048
DIRECT_NUM_BLOCKS = [2048, 32768]
DIRECT_NUM_HEADS = (8, 2)
DIRECT_HEAD_SIZE = 128
DIRECT_DTYPE = torch.bfloat16
HEAD_SIZE_TEST_NUM_HEADS_PAIRS = [(16, 16), (16, 4)]
# Prefill seq lens: (query_len, kv_len). Exclude single-token decode (q=1)
# because flash_attn_varlen_func is a prefill kernel; q_len=1 with short kv
# triggers kernel limitations (MAE > 0.1 for head != 128 in BF16, all heads in FP16).
# Single-token decode is covered separately in test_aiter_mha_decode_single_token.
SEQ_LENS = [(8, 512), (32, 1024)]
DIRECT_SEQ_LENS = [(10, 1328), (5, 18), (129, 463)]


# Reference implementation -------------------------------------------------
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Naive reference paged attention using einsum."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        if sliding_window is not None:
            window_mask = (
                torch.triu(
                    torch.ones(query_len, kv_len),
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= window_mask
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# Small test helpers ------------------------------------------------------
def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _print_close_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    abs_diff = (actual - expected).abs().float().flatten()
    expected_abs = expected.abs().float().flatten()
    allowed = atol + rtol * expected_abs
    within = abs_diff <= allowed

    total = abs_diff.numel()
    passed = int(within.sum().item())
    failed = total - passed

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = torch.quantile(abs_diff, 0.99).item()
    p999_abs = torch.quantile(abs_diff, 0.999).item()
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    print(
        "[rocm_aiter_fa] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(0.0, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )


# Direct kernel helper ----------------------------------------------------
def _run_direct_flash_attn_case(
    *,
    seq_lens: list[tuple[int, int]],
    num_blocks: int,
    sliding_window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(0)

    query_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    num_query_heads, num_kv_heads = DIRECT_NUM_HEADS
    total_query_tokens = sum(query_lens)
    total_kv_tokens = sum(kv_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = DIRECT_HEAD_SIZE**-0.5

    query = torch.randn(
        total_query_tokens,
        num_query_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    key_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_num_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (len(seq_lens), max_num_blocks_per_seq), dtype=torch.int32
    )

    token_to_batch = torch.empty(total_kv_tokens, dtype=torch.int32)
    seq_starts = torch.zeros(len(seq_lens), dtype=torch.int32)
    token_index = 0
    for batch_index, kv_len in enumerate(kv_lens):
        token_to_batch[token_index : token_index + kv_len] = batch_index
        token_index += kv_len

    gathered_key = torch.empty(
        total_kv_tokens,
        num_kv_heads,
        DIRECT_HEAD_SIZE,
        dtype=DIRECT_DTYPE,
    )
    gathered_value = torch.empty_like(gathered_key)
    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=total_kv_tokens,
    )

    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
    )
    return output, ref


# Backend contract tests --------------------------------------------------
def test_aiter_mha_backend_contract():
    """The ROCm backend advertises the dtypes, shapes, and attention types it
    is designed to handle."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    assert AiterFlashAttentionBackend.get_name() == "FLASH_ATTN"
    assert AiterFlashAttentionBackend.supported_dtypes == [
        torch.float16,
        torch.bfloat16,
    ]
    assert AiterFlashAttentionBackend.supported_kv_cache_dtypes == [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]
    assert AiterFlashAttentionBackend.get_supported_kernel_block_sizes() == [16, 32]
    assert AiterFlashAttentionBackend.get_supported_head_sizes() == [64, 128, 256]
    assert AiterFlashAttentionBackend.supports_attn_type(AttentionType.DECODER)
    assert not AiterFlashAttentionBackend.supports_attn_type(AttentionType.ENCODER)
    assert not AiterFlashAttentionBackend.supports_attn_type(
        AttentionType.ENCODER_DECODER
    )


def test_aiter_mha_backend_validates_kv_cache_block_size():
    """The backend should reject KV cache shapes that cannot be gathered
    correctly by the ROCm kernel."""
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    assert AiterFlashAttentionBackend.get_kv_cache_shape(8, 16, 8, 128) == (
        2,
        8,
        16,
        8,
        128,
    )
    with pytest.raises(ValueError, match="Block size must be a multiple of 16"):
        AiterFlashAttentionBackend.get_kv_cache_shape(8, 15, 8, 128)


def test_aiter_mha_backend_supports_compute_capability_matches_mi3xx_probe():
    """The backend should trust the ROCm MI3xx probe instead of the raw torch
    capability tuple."""
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    assert (
        AiterFlashAttentionBackend.supports_compute_capability(DeviceCapability(0, 0))
        is on_mi3xx()
    )


# Env and platform gate tests ---------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(
    ("use_aiter", "use_mha", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_aiter_mha_env_flags_control_enablement(
    use_aiter, use_mha, expected, monkeypatch
):
    """Both the global AITER flag and the MHA-specific flag must be enabled
    before the ROCm MHA path is considered active."""
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MHA", "1" if use_mha else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_mha_enabled() is expected

    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


def test_aiter_mha_platform_gate_matches_install_and_arch():
    """The global AITER availability check should only open on ROCm MI3xx when
    the aiter package is installed."""
    from vllm._aiter_ops import IS_AITER_FOUND, is_aiter_found_and_supported

    assert is_aiter_found_and_supported() is (
        current_platform.is_rocm() and on_mi3xx() and IS_AITER_FOUND
    )


# Kernel path tests -------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", HEAD_SIZE_TEST_NUM_HEADS_PAIRS)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
def test_aiter_fa_head_sizes(head_size, dtype, num_heads, seq_lens):
    """AITER flash attention should stay accurate across supported head sizes.

    This is the backend-owned head-size matrix for ``rocm_aiter_fa``. The
    mixed ROCm head-size file now keeps only ``rocm_attn`` and ``triton_attn``
    coverage.
    """
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_q_heads, num_kv_heads = num_heads
    query_len, kv_len = seq_lens
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        "head_sizes "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_varlen_paged_kv(head_size, num_heads, seq_lens, dtype):
    """Test AITER flash attention varlen with paged KV cache.

    Exercises: VLLM_ROCM_USE_AITER, VLLM_ROCM_USE_AITER_MHA
    """
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_q_heads, num_kv_heads = num_heads
    query_len, kv_len = seq_lens
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        "varlen_paged_kv "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_multi_batch(num_heads, head_size, dtype):
    """Test AITER flash attention with multiple sequences in a batch."""
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(42)

    num_q_heads, num_kv_heads = num_heads
    seq_lens = [(4, 128), (2, 256), (8, 64)]
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_seqs = len(seq_lens)
    scale = head_size**-0.5

    total_q = sum(query_lens)
    total_kv = sum(kv_lens)

    query = torch.randn(total_q, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seq_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    max_kv_len = max(kv_lens)
    max_num_blocks = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks), dtype=torch.int32
    )

    token_to_batch = torch.empty(total_kv, dtype=torch.int32)
    seq_starts = torch.zeros(num_seqs, dtype=torch.int32)
    tok_idx = 0
    for b, kl in enumerate(kv_lens):
        token_to_batch[tok_idx : tok_idx + kl] = b
        tok_idx += kl

    gathered_key = torch.empty(total_kv, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=total_kv,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max_kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        "multi_batch "
        f"dtype={dtype} head_size={head_size} "
        f"num_heads={num_heads} seq_lens={seq_lens}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("num_blocks", DIRECT_NUM_BLOCKS)
def test_aiter_fa_large_block_table_matches_reference(num_blocks):
    """The direct paged-KV path should stay stable for both normal and very
    large block tables."""
    atol = 2e-2
    rtol = 2e-2
    _assert_aiter_supported()

    output, ref = _run_direct_flash_attn_case(
        seq_lens=DIRECT_SEQ_LENS,
        num_blocks=num_blocks,
    )
    _print_close_stats(
        f"direct_varlen_paged_kv num_blocks={num_blocks}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
def test_aiter_fa_sliding_window_matches_reference():
    """The direct kernel should respect the same sliding-window causal mask as
    the naive reference implementation."""
    atol = 2e-2
    rtol = 2e-2
    _assert_aiter_supported()

    output, ref = _run_direct_flash_attn_case(
        seq_lens=[(8, 523), (24, 37), (3, 2011)],
        num_blocks=2048,
        sliding_window=256,
    )
    _print_close_stats(
        "sliding_window num_blocks=2048 window=256",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


# Decode path test --------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        pytest.param(
            torch.float16,
            marks=pytest.mark.xfail(
                reason=(
                    "aiter bug #2229: flash_attn_varlen_func currently "
                    "miscomputes FP16 single-token decode on MI3xx "
                    "(validated on gfx950; max abs diff 1.95 vs atol 1.5e-2). "
                    "Remove xfail when the AITER decode kernel is fixed. "
                    "https://github.com/ROCm/aiter/issues/2229"
                ),
                strict=True,
            ),
        ),
    ],
)
def test_aiter_mha_decode_single_token(dtype):
    """Test AITER MHA for decode (single query token per sequence).

    BF16 is the working reference configuration here. FP16 remains xfail until
    the upstream AITER single-token decode bug is fixed.
    """
    atol = 1.5e-2
    rtol = 1e-2
    _assert_aiter_supported()
    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_q_heads, num_kv_heads = 8, 8
    head_size = 128
    kv_len = 512
    scale = head_size**-0.5

    query = torch.randn(1, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0, 1], dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32)

    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)

    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache,
        value_cache=value_cache,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32),
        v_scales=torch.ones(1, dtype=torch.float32),
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=1,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _print_close_stats(
        f"decode_single_token dtype={dtype} kv_len={kv_len}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


# FP8 KV cache test -------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize("dtype", DTYPES)
def test_aiter_mha_varlen_fp8_kv(dtype):
    """AITER flash attention with FP8 KV cache matches reference on BF16-cast KV.

    cp_mha_gather_cache is called with dequant=True to dequantize FP8 to dtype
    before passing to flash_attn_varlen_func.  We compare to ref_paged_attn on
    the dtype-cast KV cache with FP8 quantization tolerance.

    Exercises: VLLM_ROCM_USE_AITER, VLLM_ROCM_USE_AITER_MHA, FP8 KV path.
    """
    atol = 0.15
    rtol = 0.05
    _assert_aiter_supported()
    if not current_platform.supports_fp8():
        pytest.skip("FP8 not supported on this hardware")

    import aiter

    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    FP8_DTYPE = current_platform.fp8_dtype()

    torch.set_default_device("cuda")
    set_random_seed(10)

    num_q_heads, num_kv_heads = 8, 8
    head_size = 128
    query_len, kv_len = 4, 128
    scale = head_size**-0.5

    query = torch.randn(query_len, num_q_heads, head_size, dtype=dtype)
    # FP8 KV cache; clamp to stay in FP8 representable range
    key_cache_fp8 = torch.clamp(
        torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)
    value_cache_fp8 = torch.clamp(
        torch.randn(NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32)
    max_num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(0, NUM_BLOCKS, (1, max_num_blocks), dtype=torch.int32)
    token_to_batch = torch.zeros(kv_len, dtype=torch.int32)
    seq_starts = torch.zeros(1, dtype=torch.int32)

    # Gather and dequantize FP8 KV to dtype
    gathered_key = torch.empty(kv_len, num_kv_heads, head_size, dtype=dtype)
    gathered_value = torch.empty_like(gathered_key)
    k_scales = torch.ones(1, dtype=torch.float32)
    v_scales = torch.ones(1, dtype=torch.float32)

    cp_mha_gather_cache(
        key_cache=key_cache_fp8,
        value_cache=value_cache_fp8,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=k_scales,
        v_scales=v_scales,
        cu_seqlens_kv=cu_seq_lens,
        token_to_batch=token_to_batch,
        seq_starts=seq_starts,
        dequant=True,
        kv_cache_layout="NHD",
        total_tokens=kv_len,
    )

    output = torch.empty_like(query)
    aiter.flash_attn_varlen_func(
        q=query,
        k=gathered_key,
        v=gathered_value,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_seq_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=kv_len,
        min_seqlen_q=1,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        return_lse=False,
        out=output,
    )

    # Reference: ref_paged_attn on dtype-cast KV (simulates perfect dequant)
    key_cache_ref = key_cache_fp8.to(dtype)
    value_cache_ref = value_cache_fp8.to(dtype)
    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache_ref,
        value_cache=value_cache_ref,
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    # FP8 quantization + dequantization introduces noise
    _print_close_stats(
        f"varlen_fp8_kv dtype={dtype} query_len={query_len} kv_len={kv_len}",
        output,
        ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)
