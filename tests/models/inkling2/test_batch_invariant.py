# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance tests for Inkling2 under VLLM_BATCH_INVARIANT=1."""

import numpy as np
import pytest
import torch

import vllm.envs as envs
from tests.models.inkling2.test_attention import requires_sm90, shear_rel_logits
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingSconvBackend
from vllm.models.inkling2.nvidia.ops.attention import (
    plan_attention,
    plan_decode_attention,
)
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.ops.linear import linear
from vllm.models.inkling2.nvidia.ops.ll_gemm_sm120 import (
    use_qkv_bmm,
    use_sm12x_bf16_ll_gemm,
)


def test_sconv_backend_supports_batch_invariance():
    assert InklingSconvBackend.supports_batch_invariance() is True


def test_attention_planners_and_ll_gemm_disabled_under_batch_invariant(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert plan_attention(False, np.array([1]), np.array([2048]), 32, 4) == (1, False)
    assert plan_attention(True, np.array([64]), np.array([2048]), 32, 4) == (1, False)
    assert plan_decode_attention(False, 1, 1, 32, 4, 1024) == 0
    assert plan_decode_attention(True, 1, 1, 32, 4, 1024) == 0

    if torch.cuda.is_available():
        x = torch.empty((1, 2304), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((2304, 2304), device="cuda", dtype=torch.bfloat16)
        assert not use_sm12x_bf16_ll_gemm(x, w)
        assert not use_qkv_bmm(x, x, x, w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_linear_batch_invariant(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(0)
    x = torch.randn(64, 2304, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2304, 2304, device="cuda", dtype=torch.bfloat16)

    y1 = linear(x[:1], w)
    for m in (5, 64):
        torch.testing.assert_close(linear(x[:m], w)[:1], y1, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_attn_res_batch_invariant(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(0)
    num_stack, num_q, d = 4, 3, 2304
    stack = torch.randn(num_stack, 128, d, device="cuda", dtype=torch.bfloat16)
    last = torch.randn(128, d, device="cuda", dtype=torch.bfloat16)
    queries = torch.randn(num_q, d, device="cuda", dtype=torch.float32)
    norm_weight = torch.randn(num_q, d, device="cuda", dtype=torch.float32)

    out_small = attn_res(
        stack[:, :1].contiguous(),
        last[:1].contiguous(),
        None,
        queries,
        norm_weight,
        1e-6,
        False,
    )
    out_big = attn_res(stack, last, None, queries, norm_weight, 1e-6, False)
    torch.testing.assert_close(out_big[:, :1], out_small, atol=0, rtol=0)


@requires_sm90
@torch.inference_mode()
def test_rel_attention_batch_invariant_across_block_m(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(42)
    heads, kv_heads, page, dim = 16, 2, 16, 128
    kv = torch.randn(20, page, kv_heads, 2 * dim, device="cuda").bfloat16()
    k, v = kv.split(dim, dim=-1)
    table = torch.randperm(20, device="cuda").view(1, -1).int()
    q_full = torch.randn(8, heads, dim, device="cuda").bfloat16()
    rel_full = torch.randn(8, heads, 1024, device="cuda").bfloat16()

    def run(q_len: int, kv_len: int):
        q = q_full[:q_len]
        sheared = shear_rel_logits(rel_full[:q_len], [(q_len, kv_len)], 128, False)
        lengths = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(heads, q_len, device="cuda")
        rel_attention(
            q,
            k,
            v,
            out,
            lse,
            sheared,
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            max_query_len=q_len,
            softmax_scale=dim**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        return out, lse

    out_4, lse_4 = run(4, 192)
    out_8, lse_8 = run(8, 196)
    torch.testing.assert_close(out_8[:4], out_4, atol=0, rtol=0)
    torch.testing.assert_close(lse_8[:, :4], lse_4, atol=0, rtol=0)
