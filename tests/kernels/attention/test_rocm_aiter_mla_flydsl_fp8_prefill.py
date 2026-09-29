# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL FP8 MLA prefill in the ROCM_AITER_FA prefill backend.

Checks the fused per-tensor FP8 quantization against a torch reference and the
backend's FP8 new-token and context-chunk attention against its BF16 path,
including the fallback when AITER reports the configuration unsupported.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only test", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.v1.attention.ops.triton_per_tensor_fp8_quant import (  # noqa: E402
    fused_per_tensor_fp8_quant,
)

NUM_HEADS = 12
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
FP8_DTYPE = torch.float8_e4m3fn


def _reference_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, NUM_HEADS, QK_HEAD_DIM, device="cuda") * 4
    k = torch.randn(num_tokens, NUM_HEADS, QK_HEAD_DIM, device="cuda")
    kv = torch.randn(
        num_tokens, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, device="cuda"
    )
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., QK_NOPE_HEAD_DIM:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_zero_input() -> None:
    x = torch.zeros(4, NUM_HEADS, QK_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    (x_fp8,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert descale.item() > 0
    assert x_fp8.float().abs().max().item() == 0


def _make_backend(monkeypatch, enabled: bool, v_head_dim: int = V_HEAD_DIM):
    from vllm.v1.attention.backends.mla.prefill import aiter_flash_attn

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv(
        "VLLM_ROCM_USE_AITER_FLYDSL_FP8_PREFILL", "1" if enabled else "0"
    )
    rocm_aiter_ops.refresh_env_variables()
    aiter_flash_attn._get_flydsl_fp8_attn.cache_clear()
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    return aiter_flash_attn.AiterFlashAttnPrefillBackend(
        num_heads=NUM_HEADS,
        scale=1.0 / math.sqrt(QK_HEAD_DIM),
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )


@pytest.fixture(autouse=True)
def _restore_aiter_env():
    yield
    from vllm.v1.attention.backends.mla.prefill import aiter_flash_attn

    rocm_aiter_ops.refresh_env_variables()
    aiter_flash_attn._get_flydsl_fp8_attn.cache_clear()


def _cu_seqlens(lens: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).to(
        device="cuda", dtype=torch.int32
    )


def _assert_close_to_bf16(fp8_result, bf16_result) -> None:
    fp8_out, fp8_lse = fp8_result
    bf16_out, bf16_lse = bf16_result
    assert fp8_out.dtype == torch.bfloat16
    assert fp8_out.shape == bf16_out.shape and fp8_lse.shape == bf16_lse.shape
    cos = F.cosine_similarity(
        fp8_out.float().flatten(), bf16_out.float().flatten(), dim=0
    )
    assert cos > 0.998
    # FP8 rounding of Q.K dominates rows that attend to only a few keys.
    lse_err = (fp8_lse - bf16_lse).abs()
    assert lse_err.max() < 0.25 and lse_err.mean() < 0.01


@pytest.mark.parametrize(
    "query_lens,context_lens",
    [([2048], [16384]), ([128], [65536]), ([300, 1, 4000], [0, 5000, 64])],
)
def test_flydsl_fp8_prefill_matches_bf16(monkeypatch, query_lens, context_lens):
    fp8_backend = _make_backend(monkeypatch, enabled=True)
    if not fp8_backend.use_flydsl_fp8:
        pytest.skip("AITER FlyDSL FP8 attention unsupported on this device/build")
    bf16_backend = _make_backend(monkeypatch, enabled=False)
    assert not bf16_backend.use_flydsl_fp8

    torch.manual_seed(0)
    num_tokens, num_ctx = sum(query_lens), sum(context_lens)
    q = torch.randn(num_tokens, NUM_HEADS, QK_HEAD_DIM, device="cuda").bfloat16()
    k = torch.randn(num_tokens, NUM_HEADS, QK_HEAD_DIM, device="cuda").bfloat16()
    v = torch.randn(
        num_tokens, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, device="cuda"
    ).bfloat16()[..., QK_NOPE_HEAD_DIM:]
    ctx_k = torch.randn(num_ctx, NUM_HEADS, QK_HEAD_DIM, device="cuda").bfloat16()
    ctx_v = torch.randn(num_ctx, NUM_HEADS, V_HEAD_DIM, device="cuda").bfloat16()

    metadata = SimpleNamespace(
        query_start_loc=_cu_seqlens(query_lens), max_query_len=max(query_lens)
    )
    chunk = SimpleNamespace(
        token_slice=slice(0, num_tokens),
        query_start_loc=_cu_seqlens(query_lens),
        cu_seq_lens=_cu_seqlens(context_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(context_lens),
    )
    from vllm.v1.attention.backends.mla.prefill import aiter_flash_attn

    quantized_counts: list[int] = []

    def spy_quant(*tensors, **kwargs):
        quantized_counts.append(len(tensors))
        return fused_per_tensor_fp8_quant(*tensors, **kwargs)

    monkeypatch.setattr(aiter_flash_attn, "fused_per_tensor_fp8_quant", spy_quant)
    results = []
    for backend in (fp8_backend, bf16_backend):
        backend.prepare_metadata(metadata)
        new = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
        ctx = backend.run_prefill_context_chunk(
            chunk, q[chunk.token_slice], ctx_k, ctx_v
        )
        results.append((new, ctx))
    # Q is quantized with K/V for the new tokens and reused by the context chunk.
    assert quantized_counts == [3, 2]

    (fp8_new, fp8_ctx), (bf16_new, bf16_ctx) = results
    _assert_close_to_bf16(fp8_new, bf16_new)
    finite = torch.isfinite(bf16_ctx[1])
    assert torch.equal(finite, torch.isfinite(fp8_ctx[1]))
    _assert_close_to_bf16(
        (fp8_ctx[0], torch.where(finite, fp8_ctx[1], 0)),
        (bf16_ctx[0], torch.where(finite, bf16_ctx[1], 0)),
    )


def test_flydsl_fp8_prefill_falls_back_when_unsupported(monkeypatch):
    if not _make_backend(monkeypatch, enabled=True).use_flydsl_fp8:
        pytest.skip("AITER FlyDSL FP8 attention unsupported on this device/build")
    # AITER's capability check rejects this V head dim.
    backend = _make_backend(monkeypatch, enabled=True, v_head_dim=100)
    assert not backend.use_flydsl_fp8
