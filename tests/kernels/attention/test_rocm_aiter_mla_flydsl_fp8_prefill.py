# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy test for the FlyDSL FP8 path of the ROCM_AITER_FA MLA prefill
backend (``VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8=1``, gfx950 only)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
    AiterFlashAttnPrefillBackend,
    _quant_fp8_kv_per_tensor,
    _quant_fp8_per_tensor,
)

NOPE, ROPE, VD = 128, 64, 128
SCALE = (NOPE + ROPE) ** -0.5


def _flydsl_fp8_available() -> bool:
    if not (current_platform.is_rocm() and torch.cuda.is_available()):
        return False
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_supported
    except ImportError:
        return False
    return flydsl_flash_attn_fp8_supported(
        torch.device("cuda", 0), 16, 16, NOPE + ROPE, VD
    )


pytestmark = pytest.mark.skipif(
    not _flydsl_fp8_available(),
    reason="needs gfx950 and AITER FlyDSL FP8 attention",
)


def _mla_qkv(num_q: int, num_kv: int, num_heads: int):
    q = torch.randn(num_q, num_heads, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(num_kv, num_heads, NOPE + VD, device="cuda", dtype=torch.bfloat16)
    k_nope, v = kv.split([NOPE, VD], dim=-1)
    k_pe = torch.randn(num_kv, 1, ROPE, device="cuda", dtype=torch.bfloat16)
    k = torch.cat((k_nope, k_pe.expand(-1, num_heads, -1)), dim=-1)
    return q, k, v


def _cu(lens: list[int]) -> torch.Tensor:
    return torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)


def _ref_attn(q, k, v, q_lens, kv_lens, causal):
    """fp32 attention on the dequantized FP8 inputs the kernel sees."""
    q8, q_descale = _quant_fp8_per_tensor(q)
    k8, v8, k_descale, v_descale = _quant_fp8_kv_per_tensor(k, v)
    q, k, v = q8.float() * q_descale, k8.float() * k_descale, v8.float() * v_descale
    outs, lses = [], []
    for qi, ki, vi in zip(q.split(q_lens), k.split(kv_lens), v.split(kv_lens)):
        s = torch.einsum("qhd,khd->hqk", qi, ki) * SCALE
        if causal:
            mask = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device)
            s = s.masked_fill(~mask.tril(s.shape[-1] - s.shape[-2]), float("-inf"))
        lses.append(s.logsumexp(-1))
        outs.append(torch.einsum("hqk,khd->qhd", s.softmax(-1), vi))
    return torch.cat(outs), torch.cat(lses, dim=1)


def _make_backend(monkeypatch, num_heads: int) -> AiterFlashAttnPrefillBackend:
    monkeypatch.setenv("VLLM_ROCM_MLA_PREFILL_FLYDSL_FP8", "1")
    backend = AiterFlashAttnPrefillBackend(
        num_heads=num_heads,
        scale=SCALE,
        kv_lora_rank=512,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VD,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    assert backend.fp8_attn_func is not None
    return backend


def _check_attn(out, lse, ref_out, ref_lse):
    assert out.dtype == torch.bfloat16 and out.shape == ref_out.shape
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=0)
    # The kernel rounds softmax probabilities to FP8 for the PV product.
    assert (out.float() - ref_out).norm() <= 0.04 * ref_out.norm()


@pytest.mark.parametrize("num_heads", [12, 16])
def test_new_tokens_matches_reference(monkeypatch, num_heads):
    torch.manual_seed(0)
    lens = [1, 37, 515, 2048]
    q, k, v = _mla_qkv(sum(lens), sum(lens), num_heads)
    backend = _make_backend(monkeypatch, num_heads)
    backend.prepare_metadata(
        SimpleNamespace(query_start_loc=_cu(lens), max_query_len=max(lens))
    )
    out, lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    _check_attn(out, lse, *_ref_attn(q, k, v, lens, lens, causal=True))


@pytest.mark.parametrize("num_heads", [12, 16])
def test_context_chunk_matches_reference(monkeypatch, num_heads):
    torch.manual_seed(0)
    q_lens, kv_lens = [1, 64, 300], [4096, 513, 1000]
    q, k, v = _mla_qkv(sum(q_lens), sum(kv_lens), num_heads)
    backend = _make_backend(monkeypatch, num_heads)
    chunk = SimpleNamespace(
        query_start_loc=_cu(q_lens),
        cu_seq_lens=_cu(kv_lens),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
    )
    out, lse = backend.run_prefill_context_chunk(chunk, q, k, v)
    _check_attn(out, lse, *_ref_attn(q, k, v, q_lens, kv_lens, causal=False))


def _check_quant(x, x8, descale):
    x = x.float()
    amax = x.abs().max() if x.numel() else torch.tensor(0.0, device=x.device)
    expected = amax / 448 if amax > 0 else torch.tensor(1e-6, device=x.device)
    torch.testing.assert_close(descale.float().reshape(()), expected, rtol=1e-6, atol=0)
    assert x8.shape == x.shape
    ref = (x / expected).clamp(-448, 448)
    # Within one E4M3 step (3 mantissa bits) of the exact scaled value.
    assert ((x8.float() - ref).abs() <= ref.abs() / 8 + 2**-9).all()


# Token counts cover the single-pass and every two-pass size class.
@pytest.mark.parametrize("tokens", [0, 1, 7, 4096, 20000, 30000])
def test_kv_quant(tokens):
    torch.manual_seed(tokens)
    _, k, v = _mla_qkv(0, tokens, 12)
    k8, v8, k_descale, v_descale = _quant_fp8_kv_per_tensor(k, v)
    _check_quant(k, k8, k_descale)
    _check_quant(v, v8, v_descale)


def test_kv_quant_zero():
    k = torch.zeros(5000, 12, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(5000, 12, VD, device="cuda", dtype=torch.bfloat16)
    k8, v8, k_descale, v_descale = _quant_fp8_kv_per_tensor(k, v)
    assert k_descale.item() == pytest.approx(1e-6)
    assert v_descale.item() == pytest.approx(1e-6)
    assert not k8.float().any() and not v8.float().any()


@pytest.mark.parametrize("tokens", [1, 3, 8191])
def test_q_quant_strided(tokens):
    q = torch.randn(tokens, 13, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    q = q[:, 1:]
    q8, q_descale = _quant_fp8_per_tensor(q)
    _check_quant(q, q8, q_descale)
