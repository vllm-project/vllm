# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-specific tests for FP8 quantization and kernels."""

import importlib
import math
import warnings

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import (
    ref_dynamic_per_tensor_fp8_quant,
    ref_dynamic_per_token_quant,
)
from tests.kernels.utils import _assert_deterministic
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

DEVICE = "cuda"
FP8_DTYPE = current_platform.fp8_dtype()
fp8_only = pytest.mark.skipif(
    not current_platform.supports_fp8(), reason="ROCm FP8 only"
)

ENV_DEFAULTS = {
    "VLLM_ROCM_FP8_PADDING": True,
    "VLLM_ROCM_USE_AITER_FP8BMM": True,
    "VLLM_ROCM_FP8_MFMA_PAGE_ATTN": False,
}

SCALED_MM_CASES = [
    pytest.param((1, 128, 128), 0.05, 0.15, 0.95, id="1x128x128"),
    pytest.param((4, 256, 512), 0.10, 0.30, 0.96, id="4x256x512"),
    pytest.param((8, 512, 1024), 0.20, 0.50, 0.99, id="8x512x1024"),
]


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _quantile(flat: torch.Tensor, q: float) -> float:
    if flat.numel() == 1:
        return flat.item()
    return torch.quantile(flat, q).item()


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = math.floor(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _assert_abs_error_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    tight_atol: float,
    max_atol: float,
    pass_rate: float,
) -> None:
    diff = (actual.float() - expected.float()).abs().flatten()
    total = diff.numel()
    within_tight_count = int((diff <= tight_atol).sum().item())
    fail_count = total - within_tight_count
    above_max_count = int((diff > max_atol).sum().item())
    within_tight = within_tight_count / total
    allowed_fail_rate = 1.0 - pass_rate
    above_max = above_max_count / total
    max_fail_rate = 0.0
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    p99 = _quantile(diff, 0.99)
    p999 = _quantile(diff, 0.999)
    msg = (
        f"[rocm_fp8] {label}: abs<={tight_atol} pass={within_tight:.4%} "
        f"({within_tight_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"abs>{max_atol}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"max_diff={max_diff:.6g} mean_diff={mean_diff:.6g} "
        f"p99={p99:.6g} p999={p999:.6g}"
    )
    print(msg)
    if within_tight < 1.0:
        warnings.warn(msg, stacklevel=2)
    assert within_tight >= pass_rate, msg
    assert above_max <= max_fail_rate, msg


def _assert_rel_error_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    tight_rtol: float,
    max_rtol: float,
    pass_rate: float,
    max_fail_rate: float,
) -> None:
    rel = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-5)
    ).flatten()
    total = rel.numel()
    within_tight_count = int((rel <= tight_rtol).sum().item())
    fail_count = total - within_tight_count
    above_max_count = int((rel > max_rtol).sum().item())
    within_tight = within_tight_count / total
    allowed_fail_rate = 1.0 - pass_rate
    above_max = above_max_count / total
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()
    p99 = _quantile(rel, 0.99)
    p999 = _quantile(rel, 0.999)
    msg = (
        f"[rocm_fp8] {label}: rel<={tight_rtol} pass={within_tight:.4%} "
        f"({within_tight_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"rel>{max_rtol}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"max_rel={max_rel:.6g} mean_rel={mean_rel:.6g} "
        f"p99={p99:.6g} p999={p999:.6g}"
    )
    print(msg)
    if within_tight < 1.0:
        warnings.warn(msg, stacklevel=2)
    assert within_tight >= pass_rate, msg
    assert above_max <= max_fail_rate, msg


def _ref_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
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
            torch.ones(query_len, kv_len, device=q.device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def test_rocm_fp8_env_defaults(monkeypatch):
    with monkeypatch.context() as mp:
        for env_name in ENV_DEFAULTS:
            mp.delenv(env_name, raising=False)
        envs = _reload_envs()
        for env_name, expected in ENV_DEFAULTS.items():
            assert getattr(envs, env_name) is expected
    _reload_envs()


@pytest.mark.parametrize(
    ("env_name", "raw_value", "expected"),
    [
        ("VLLM_ROCM_FP8_PADDING", "0", False),
        ("VLLM_ROCM_FP8_PADDING", "1", True),
        ("VLLM_ROCM_USE_AITER_FP8BMM", "0", False),
        ("VLLM_ROCM_USE_AITER_FP8BMM", "1", True),
        ("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0", False),
        ("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "1", True),
    ],
)
def test_rocm_fp8_env_overrides(env_name, raw_value, expected, monkeypatch):
    with monkeypatch.context() as mp:
        mp.setenv(env_name, raw_value)
        envs = _reload_envs()
        assert getattr(envs, env_name) is expected
    _reload_envs()


@fp8_only
def test_maybe_pad_fp8_weight_respects_env(monkeypatch):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _maybe_pad_fp8_weight,
    )

    weight = torch.randn(64, 512, device=DEVICE).to(FP8_DTYPE)

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_FP8_PADDING", "1")
        _reload_envs()
        padded = _maybe_pad_fp8_weight(weight)
        assert padded.shape == weight.shape
        assert padded.dtype == weight.dtype
        assert padded.data_ptr() != weight.data_ptr()
        torch.testing.assert_close(padded.float(), weight.float())

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_FP8_PADDING", "0")
        _reload_envs()
        unpadded = _maybe_pad_fp8_weight(weight)
        assert unpadded.data_ptr() == weight.data_ptr()

    _reload_envs()


@fp8_only
@pytest.mark.parametrize(
    ("use_aiter", "fp8bmm_enabled", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_aiter_fp8bmm_enabled_api_respects_env(
    use_aiter, fp8bmm_enabled, expected, monkeypatch
):
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

    assert is_aiter_found_and_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_FP8BMM", "1" if fp8bmm_enabled else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()
        assert rocm_aiter_ops.is_fp8bmm_enabled() is expected

    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


@fp8_only
@pytest.mark.parametrize(
    ("shape", "tight_atol", "max_atol", "pass_rate"), SCALED_MM_CASES
)
def test_fp8_scaled_mm_matches_dequantized_reference(
    shape, tight_atol, max_atol, pass_rate
):
    torch.manual_seed(0)

    batch, m_dim, k_dim = shape
    n_dim = k_dim
    a = torch.randn(batch * m_dim, k_dim, device=DEVICE)
    b = torch.randn(n_dim, k_dim, device=DEVICE)

    a_fp8, scale_a = ref_dynamic_per_tensor_fp8_quant(a)
    b_fp8, scale_b = ref_dynamic_per_tensor_fp8_quant(b)
    out = torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        out_dtype=torch.bfloat16,
        scale_a=scale_a,
        scale_b=scale_b,
    )

    a_dq = a_fp8.float() * scale_a.float()
    b_dq = b_fp8.float() * scale_b.float()
    ref = torch.mm(a_dq, b_dq.t())

    assert out.shape == (batch * m_dim, n_dim)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()

    _assert_abs_error_budget(
        out.float(),
        ref,
        label=f"scaled_mm shape={shape}",
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
    )


@fp8_only
def test_fp8_scaled_mm_determinism():
    torch.manual_seed(5)

    m_dim, k_dim, n_dim = 64, 128, 128
    a = torch.randn(m_dim, k_dim, device=DEVICE)
    b = torch.randn(n_dim, k_dim, device=DEVICE)
    a_fp8, scale_a = ref_dynamic_per_tensor_fp8_quant(a)
    b_fp8, scale_b = ref_dynamic_per_tensor_fp8_quant(b)

    def run_gemm():
        return torch._scaled_mm(
            a_fp8,
            b_fp8.t(),
            out_dtype=torch.bfloat16,
            scale_a=scale_a,
            scale_b=scale_b,
        )

    _assert_deterministic(run_gemm, n_runs=4)


@fp8_only
def test_aiter_fp8_kv_attention_matches_bf16_reference():
    import aiter

    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

    assert is_aiter_found_and_supported()

    set_random_seed(10)

    num_q_heads, num_kv_heads = 8, 8
    head_size = 128
    query_len, kv_len = 4, 128
    block_size = 16
    num_blocks = 2048
    scale = head_size**-0.5

    query = torch.randn(
        query_len, num_q_heads, head_size, dtype=torch.bfloat16, device=DEVICE
    )
    key_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=DEVICE),
        min=-1.0,
        max=1.0,
    ).to(FP8_DTYPE)
    value_cache_fp8 = torch.clamp(
        torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=DEVICE),
        min=-1.0,
        max=1.0,
    ).to(FP8_DTYPE)

    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32, device=DEVICE)
    cu_seq_lens = torch.tensor([0, kv_len], dtype=torch.int32, device=DEVICE)
    block_tables = torch.randint(
        0,
        num_blocks,
        (1, (kv_len + block_size - 1) // block_size),
        dtype=torch.int32,
        device=DEVICE,
    )
    token_to_batch = torch.zeros(kv_len, dtype=torch.int32, device=DEVICE)
    seq_starts = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    gathered_key = torch.empty(
        kv_len, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEVICE
    )
    gathered_value = torch.empty_like(gathered_key)

    cp_mha_gather_cache(
        key_cache=key_cache_fp8,
        value_cache=value_cache_fp8,
        key=gathered_key,
        value=gathered_value,
        block_tables=block_tables,
        k_scales=torch.ones(1, dtype=torch.float32, device=DEVICE),
        v_scales=torch.ones(1, dtype=torch.float32, device=DEVICE),
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

    ref = _ref_paged_attention(
        query=query,
        key_cache=key_cache_fp8.to(torch.bfloat16),
        value_cache=value_cache_fp8.to(torch.bfloat16),
        query_lens=[query_len],
        kv_lens=[kv_len],
        block_tables=block_tables,
        scale=scale,
    )

    _assert_abs_error_budget(
        output.float(),
        ref.float(),
        label="aiter_fp8_kv_attention",
        tight_atol=0.15,
        max_atol=0.25,
        pass_rate=0.98,
    )


@fp8_only
@pytest.mark.parametrize("shape", [(128, 256), (512, 1024), (1024, 4096)])
def test_fp8_per_tensor_quant_matches_reference(shape):
    torch.manual_seed(0)

    x = torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)

    out, scale = ops.scaled_fp8_quant(x)
    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)

    assert out.shape == x.shape
    assert out.dtype == FP8_DTYPE
    torch.testing.assert_close(scale, ref_scale, atol=0.0, rtol=0.0)
    assert torch.equal(out, ref_out)

    dequant = out.float() * scale.float()
    _assert_rel_error_budget(
        dequant,
        x.float(),
        label=f"per_tensor_quant shape={shape}",
        tight_rtol=0.06,
        max_rtol=0.60,
        pass_rate=0.99,
        max_fail_rate=1e-4,
    )


@fp8_only
def test_aiter_group_fp8_quant_roundtrip():
    from aiter import dtypes

    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

    assert is_aiter_found_and_supported()

    torch.manual_seed(0)

    m_dim, n_dim = 64, 4096
    group_size = 128
    x = torch.randn(m_dim, n_dim, dtype=torch.bfloat16, device=DEVICE)

    out, scales = rocm_aiter_ops.group_fp8_quant(x, group_size)
    assert out.dtype == dtypes.fp8
    assert scales.dtype == torch.float32

    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :n_dim]
    dequant = out.float() * scales_expanded
    _assert_rel_error_budget(
        dequant,
        x.float(),
        label="aiter_group_fp8_quant",
        tight_rtol=0.06,
        max_rtol=0.30,
        pass_rate=0.99,
        max_fail_rate=1e-4,
    )


@fp8_only
def test_fp8_per_token_quant_matches_reference():
    torch.manual_seed(0)

    x = torch.randn(32, 4096, dtype=torch.bfloat16, device=DEVICE)

    out, scale = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    ref_out, ref_scale = ref_dynamic_per_token_quant(x, FP8_DTYPE)

    assert out.shape == x.shape
    assert out.dtype == FP8_DTYPE
    assert scale.shape == (x.shape[0], 1)
    torch.testing.assert_close(scale, ref_scale, atol=0.0, rtol=0.0)
    assert torch.equal(out, ref_out)

    dequant = out.float() * scale.float()
    _assert_rel_error_budget(
        dequant,
        x.float(),
        label="per_token_quant",
        tight_rtol=0.06,
        max_rtol=0.30,
        pass_rate=0.99,
        max_fail_rate=1e-4,
    )
