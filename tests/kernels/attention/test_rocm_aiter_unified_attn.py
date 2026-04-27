# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-specific tests for the AITER unified-attention backend.

This file owns the parts of unified attention that are specific to the ROCm
backend class and the ROCm AITER paged-KV path:
- backend-specific static contract such as block-size, head-size, and
  attention-type support
- the helper env gate for Triton unified attention
- representative ROCm paged-KV decode correctness
- FP8 KV-cache decode correctness
- decode determinism

Direct unified-attention kernel stress, larger shape matrices, sliding-window
coverage, and output-scale paths live in
``tests/kernels/attention/test_triton_unified_attention.py``.

ROCm attention registry and selector wiring live in
``tests/v1/attention/test_rocm_attention_backends_selection.py``.
"""

import importlib
import warnings

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

BLOCK_SIZE = 16
NUM_BLOCKS = 1024
FP8_DTYPE = current_platform.fp8_dtype()

DECODE_CASES = [
    pytest.param(64, torch.float16, 16, id="fp16_h64_b16"),
    pytest.param(128, torch.bfloat16, 64, id="bf16_h128_b64"),
]


# Reference implementation -------------------------------------------------
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Naive reference paged attention using einsum."""
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i, query_len in enumerate(query_lens):
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, repeat, dim=1)
            v = torch.repeat_interleave(v, repeat, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
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


def _assert_fp8_supported() -> None:
    assert current_platform.supports_fp8(), (
        "FP8 is required on this ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _quantile(values: torch.Tensor, q: float) -> float:
    return torch.quantile(values, q).item()


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
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    print(
        "[rocm_aiter_unified_attention] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(0.0, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"max_abs={abs_diff.max().item():.6g} "
        f"mean_abs={abs_diff.mean().item():.6g} "
        f"p99_abs={_quantile(abs_diff, 0.99):.6g} "
        f"p999_abs={_quantile(abs_diff, 0.999):.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )


def _assert_abs_error_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    tight_atol: float,
    max_atol: float,
    pass_rate: float,
    max_fail_rate: float,
) -> None:
    abs_diff = (actual.float() - expected.float()).abs().flatten()
    total = abs_diff.numel()
    within_tight_count = int((abs_diff <= tight_atol).sum().item())
    fail_count = total - within_tight_count
    above_max_count = int((abs_diff > max_atol).sum().item())
    within_tight = within_tight_count / total
    above_max = above_max_count / total
    allowed_fail_rate = 1.0 - pass_rate

    msg = (
        "[rocm_aiter_unified_attention] "
        f"{label}: "
        f"abs<={tight_atol:g} pass={within_tight:.4%} "
        f"({within_tight_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"max_abs={abs_diff.max().item():.6g} "
        f"mean_abs={abs_diff.mean().item():.6g} "
        f"p99_abs={_quantile(abs_diff, 0.99):.6g} "
        f"p999_abs={_quantile(abs_diff, 0.999):.6g}"
    )
    print(msg)
    if within_tight < 1.0:
        warnings.warn(msg, stacklevel=2)
    assert within_tight >= pass_rate, msg
    assert above_max <= max_fail_rate, msg


def _make_decode_case(
    *,
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    seq_lens: list[int],
):
    torch.set_default_device("cuda")
    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = len(seq_lens)
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

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
    }


# Backend contract tests --------------------------------------------------
def test_unified_attn_backend_contract():
    """The ROCm unified-attention backend should advertise the static contract
    its callers rely on."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend,
    )

    assert RocmAiterUnifiedAttentionBackend.get_preferred_block_size(16) == 64
    assert RocmAiterUnifiedAttentionBackend.supports_block_size(16)
    assert RocmAiterUnifiedAttentionBackend.supports_block_size(64)
    assert not RocmAiterUnifiedAttentionBackend.supports_block_size(15)
    assert RocmAiterUnifiedAttentionBackend.supports_head_size(32)
    assert RocmAiterUnifiedAttentionBackend.supports_head_size(256)
    assert not RocmAiterUnifiedAttentionBackend.supports_head_size(16)
    assert RocmAiterUnifiedAttentionBackend.supports_mm_prefix()
    assert RocmAiterUnifiedAttentionBackend.supports_sink()
    assert RocmAiterUnifiedAttentionBackend.forward_includes_kv_cache_update is False
    assert RocmAiterUnifiedAttentionBackend.supports_attn_type(AttentionType.DECODER)
    assert RocmAiterUnifiedAttentionBackend.supports_attn_type(AttentionType.ENCODER)
    assert RocmAiterUnifiedAttentionBackend.supports_attn_type(
        AttentionType.ENCODER_ONLY
    )
    assert RocmAiterUnifiedAttentionBackend.supports_attn_type(
        AttentionType.ENCODER_DECODER
    )


def test_unified_attn_backend_validates_kv_cache_block_size():
    """The backend should reject KV-cache block sizes it cannot gather
    correctly."""
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


# Env contract tests ------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
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
    """The unified-attention helper gate should follow the exact env matrix."""
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

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


# Kernel path tests -------------------------------------------------------
@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.parametrize(("head_size", "dtype", "block_size"), DECODE_CASES)
def test_unified_attn_decode_matches_reference(head_size, dtype, block_size):
    """The ROCm unified-attention decode path should match the naive paged-KV
    reference for representative head sizes and dtypes."""
    from aiter.ops.triton.unified_attention import unified_attention

    _assert_aiter_supported()
    torch.manual_seed(0)

    case = _make_decode_case(
        head_size=head_size,
        dtype=dtype,
        block_size=block_size,
        seq_lens=[128, 256, 384, 512],
    )
    output_ref = ref_paged_attn(
        query=case["query"],
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        query_lens=[1] * len(case["seq_lens"]),
        kv_lens=case["seq_lens"],
        block_tables=case["block_tables"],
        scale=case["scale"],
    )

    output = torch.zeros_like(case["query"])
    unified_attention(
        q=case["query"],
        k=case["key_cache"],
        v=case["value_cache"],
        out=output,
        cu_seqlens_q=case["cu_seqlens_q"],
        max_seqlen_q=1,
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

    atol, rtol = 1.5e-2, 1e-2
    _print_close_stats(
        f"decode_ref head={head_size} dtype={dtype} block={block_size}",
        output,
        output_ref,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
def test_unified_attn_decode_determinism():
    """The ROCm unified-attention decode path should be deterministic across
    repeated runs."""
    from aiter.ops.triton.unified_attention import unified_attention

    _assert_aiter_supported()
    torch.manual_seed(3)

    case = _make_decode_case(
        head_size=128,
        dtype=torch.bfloat16,
        block_size=BLOCK_SIZE,
        seq_lens=[64, 128],
    )

    def run_decode():
        out = torch.zeros_like(case["query"])
        unified_attention(
            q=case["query"],
            k=case["key_cache"],
            v=case["value_cache"],
            out=out,
            cu_seqlens_q=case["cu_seqlens_q"],
            max_seqlen_q=1,
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
        return out

    reference = run_decode()
    for run in range(1, 4):
        output = run_decode()
        diff = output.float() - reference.float()
        assert torch.equal(reference, output), (
            f"Run {run}: unified-attention decode output differs from run 0 "
            f"(max diff = {(diff).abs().max().item():.2e})"
        )


@pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.parametrize("head_size", [64, 128])
def test_unified_attn_decode_fp8_kv_cache(head_size):
    """The ROCm unified-attention decode path should keep FP8 KV-cache error
    within the measured budget."""
    from aiter.ops.triton.unified_attention import unified_attention

    _assert_aiter_supported()
    _assert_fp8_supported()
    torch.manual_seed(5)

    dtype = torch.bfloat16
    num_q_heads = 8
    num_kv_heads = 8
    num_seqs = 2
    seq_lens = [128, 256]
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5
    num_blocks = 512

    torch.set_default_device("cuda")
    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype)
    key_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)
    value_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, head_size), -1.0, 1.0
    ).to(FP8_DTYPE)

    max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seqlens_q = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(num_seqs, num_kv_heads, dtype=torch.float32, device="cuda")

    output_fp8 = torch.zeros(num_seqs, num_q_heads, head_size, dtype=dtype)
    unified_attention(
        q=query,
        k=key_cache_fp8,
        v=value_cache_fp8,
        out=output_fp8,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_tensor,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=None,
        output_scale=None,
    )

    output_ref = ref_paged_attn(
        query=query,
        key_cache=key_cache_fp8.to(dtype),
        value_cache=value_cache_fp8.to(dtype),
        query_lens=[1] * num_seqs,
        kv_lens=seq_lens,
        block_tables=block_tables,
        scale=scale,
    )

    _assert_abs_error_budget(
        output_fp8,
        output_ref,
        label=f"fp8_kv_decode head={head_size}",
        tight_atol=0.01,
        max_atol=0.02,
        pass_rate=1.0,
        max_fail_rate=0.0,
    )
