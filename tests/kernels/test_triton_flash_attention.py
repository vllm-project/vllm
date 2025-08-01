# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_flash_attention kernel

Run `pytest tests/kernels/test_triton_flash_attention.py`.
"""
import pytest
import torch

from vllm.attention.ops.triton_flash_attention import (SUPPORTED_LAYOUTS,
                                                       MetaData,
                                                       compute_alibi_tensor,
                                                       scale_fp8,
                                                       triton_attention_rocm)
from vllm.platforms import current_platform


class ReferenceAttention:

    def __init__(self, Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, use_alibi, dtype,
                 input_metadata):
        self.Z = Z
        self.HQ = HQ
        self.HK = HK
        self.N_CTX_Q = N_CTX_Q
        self.N_CTX_K = N_CTX_K
        self.D_HEAD = D_HEAD
        self.use_alibi = use_alibi
        self.dtype = dtype
        self.input_metadata = input_metadata

    def fwd(self, q, k, v):
        scores = torch.einsum('bhqd,bhkd->bhqk', q,
                              k).float() * self.input_metadata.sm_scale
        if self.input_metadata.causal:
            mask = torch.tril(torch.ones(self.N_CTX_Q,
                                         self.N_CTX_K,
                                         device="cuda"),
                              diagonal=self.N_CTX_K - self.N_CTX_Q)
            scores[:, :, mask == 0] = float("-inf")

        if self.input_metadata.bias is not None:
            scores += self.input_metadata.bias

        if self.use_alibi:
            scores += compute_alibi_tensor(self.input_metadata.alibi_slopes,
                                           self.N_CTX_Q, self.N_CTX_K)

        p = torch.softmax(scores, dim=-1)
        if self.input_metadata.causal:
            # If N_CTX_Q > N_CTX_K, there's at least one row of all -infs going
            # into softmax. This creates a row of NaNs as -inf - -inf == NaN.
            # So we fix this by converting the NaNs to 0s, which is what they
            # should be out of the softmax.
            nan_mask = torch.isnan(p)
            p[nan_mask == 1] = 0
        ref_out = torch.einsum('bhqk,bhkd->bhqd', p.to(self.dtype), v)
        # compare
        if self.input_metadata.layout == 'bshd':
            ref_out = ref_out.transpose(1, 2).clone()
        return ref_out

    def fwd_fp8(self, q_quantized, k_quantized, v_quantized):
        q = (q_quantized.to(torch.float16) * self.input_metadata.q_descale).to(
            self.dtype)
        k = (k_quantized.to(torch.float16) * self.input_metadata.k_descale).to(
            self.dtype)
        v = (v_quantized.to(torch.float16) * self.input_metadata.v_descale).to(
            self.dtype)
        result = self.fwd(q, k, v)
        if self.input_metadata.o_scale is not None:
            result, _ = scale_fp8(result, self.input_metadata.o_scale)
        return result

    def fwd_fp8_kv(self, q, k_quantized, v_quantized):
        k_descale, v_descale = (self.input_metadata.k_descale,
                                self.input_metadata.v_descale)
        k_dequantized = (k_quantized.to(torch.float32) *
                         k_descale.to(torch.float32)).to(self.dtype)
        v_dequantized = (v_quantized.to(torch.float32) *
                         v_descale.to(torch.float32)).to(self.dtype)
        return self.fwd(q, k_dequantized, v_dequantized)

    def varlen_fwd(self, q, k, v, is_mqa=False):
        ref_out = torch.empty_like(q)
        if is_mqa:
            # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so
            # the size aligns with Q.
            k_ref = k.view(k.shape[0], k.shape[1], 1,
                           k.shape[2]).expand(-1, -1, self.HQ // self.HK, -1)
            v_ref = v.view(v.shape[0], v.shape[1], 1,
                           v.shape[2]).expand(-1, -1, self.HQ // self.HK, -1)
        else:
            k_ref = k
            v_ref = v

        for i in range(0, self.input_metadata.num_contexts):
            start_q, start_k = self.input_metadata.cu_seqlens_q[
                i], self.input_metadata.cu_seqlens_k[i]
            end_q, end_k = self.input_metadata.cu_seqlens_q[
                i + 1], self.input_metadata.cu_seqlens_k[i + 1]
            k_curr = k_ref[start_k:end_k]
            v_curr = v_ref[start_k:end_k]
            if is_mqa:
                k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
                v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
            scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q],
                                  k_curr).float()
            p = torch.softmax(scores * self.input_metadata.sm_scale,
                              dim=-1).half()
            ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
        return ref_out


def quantize_input(q, k, v, fp8_kv=False, use_o_scale=False):
    q_descale = None
    if not fp8_kv:
        q, q_descale = scale_fp8(q)
    k, k_descale = scale_fp8(k)
    v, v_descale = scale_fp8(v)

    # In real world use case, the p scale would be a parameter trained by the
    # model.
    p_scale = None

    o_scale = torch.rand(1, device="cuda",
                         requires_grad=False) if use_o_scale else None

    return q, k, v, q_descale, k_descale, v_descale, p_scale, o_scale


def input_helper(
    Z,
    HQ,
    HK,
    N_CTX_Q,
    N_CTX_K,
    D_HEAD,
    dtype,
    layout=None,
    use_alibi=None,
    causal=None,
    is_fp8=False,
    fp8_kv=False,
    use_o_scale=False,
    use_bias=False,
):
    assert layout in SUPPORTED_LAYOUTS, "Got unsupported layout."

    current_platform.seed_everything(0)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts
        # 2^(-8/n)
        alibi_slopes = torch.tensor(
            [2**(-8 / HQ * i) for i in range(1, HQ + 1)],
            dtype=torch.float32,
            device="cuda").repeat(Z, 1)
    else:
        alibi_slopes = None

    if use_bias:
        bias = torch.randn((1, HQ, N_CTX_Q, N_CTX_K),
                           dtype=dtype,
                           device="cuda",
                           requires_grad=False)
    else:
        bias = None

    q = torch.randn(q_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)
    k = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)
    v = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=False)

    if is_fp8:
        (q, k, v, q_descale, k_descale, v_descale, p_scale,
         o_scale) = quantize_input(q,
                                   k,
                                   v,
                                   use_o_scale=use_o_scale,
                                   fp8_kv=fp8_kv)
    else:
        q_descale = k_descale = v_descale = p_scale = o_scale = None

    input_metadata = MetaData(sm_scale=D_HEAD**-0.5,
                              max_seqlens_q=N_CTX_Q,
                              max_seqlens_k=N_CTX_K,
                              layout=layout,
                              alibi_slopes=alibi_slopes,
                              alibi_batch=Z,
                              alibi_nheads=HQ,
                              q_descale=q_descale,
                              k_descale=k_descale,
                              v_descale=v_descale,
                              p_scale=p_scale,
                              o_scale=o_scale,
                              bias=bias,
                              seqlen_q=N_CTX_Q,
                              seqlen_k=N_CTX_K)
    return q, k, v, input_metadata


def varlen_input_helper(Z,
                        HQ,
                        HK,
                        N_CTX_Q,
                        N_CTX_K,
                        D_HEAD,
                        dtype,
                        equal_seqlens=False):
    current_platform.seed_everything(0)

    # Random sequence lengths. Using N_CTX as kind of max of sum of individual
    # seqs
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        seqlens_q = torch.randint(1,
                                  max_seqlens_q + 1, (Z, ),
                                  dtype=torch.int32)
        seqlens_k = torch.randint(1,
                                  max_seqlens_k + 1, (Z, ),
                                  dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z, ), N_CTX_Q // Z)
        seqlens_k = torch.full((Z, ), N_CTX_K // Z)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        seqlens_q.cumsum(dim=0, dtype=torch.int32)
    ])
    cu_seqlens_k = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        seqlens_k.cumsum(dim=0, dtype=torch.int32)
    ])
    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, HK, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, HK, D_HEAD), dtype=dtype,
                    device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 48, 12, 1, 1, 64),
    (4, 4, 4, 128, 128, 65),
    (16, 48, 48, 1, 1, 128),
    (64, 48, 24, 3, 3, 128),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd'])
def test_op_fwd(Z,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                causal,
                use_alibi,
                layout,
                dtype=torch.float16):
    current_platform.seed_everything(0)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout, use_alibi, causal)

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all
    # layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(k.shape[0], -1, k.shape[2],
                                                  k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1,
                                      -1).reshape(v.shape[0], -1, v.shape[2],
                                                  v.shape[3])

    ref_impl = ReferenceAttention(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                  use_alibi, dtype, input_metadata)
    ref_out = ref_impl.fwd(q, k, v)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
@pytest.mark.parametrize('use_o_scale', [True, False])
@pytest.mark.skipif(torch.cuda.get_device_capability() < (9, 0),
                    reason="Triton FP8 requires CUDA 9.0 or higher")
def test_op_fwd_fp8(Z,
                    H,
                    N_CTX_Q,
                    N_CTX_K,
                    D_HEAD,
                    causal,
                    layout,
                    use_o_scale,
                    dtype=torch.float32):
    current_platform.seed_everything(0)

    # Disable grad to save memory it won't run into OOM on CI machine.
    # q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD,
    # dtype, layout)

    q_quantized, k_quantized, v_quantized, input_metadata = input_helper(
        Z,
        H,
        H,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        dtype,
        causal=causal,
        layout=layout,
        is_fp8=True,
        use_o_scale=use_o_scale)

    o = torch.empty_like(q_quantized) if use_o_scale else None

    tri_out, _ = triton_attention_rocm(q_quantized, k_quantized, v_quantized,
                                       o, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd_fp8(q_quantized, k_quantized, v_quantized)

    # compare
    torch.testing.assert_close(ref_out.to(torch.float32),
                               tri_out.to(torch.float32),
                               atol=7e-2,
                               rtol=2e-1)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
    (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_fwd_fp8_kv(Z,
                       H,
                       N_CTX_Q,
                       N_CTX_K,
                       D_HEAD,
                       causal,
                       layout,
                       dtype=torch.float32):
    current_platform.seed_everything(0)

    q, k_quantized, v_quantized, input_metadata = input_helper(Z,
                                                               H,
                                                               H,
                                                               N_CTX_Q,
                                                               N_CTX_K,
                                                               D_HEAD,
                                                               dtype,
                                                               causal=causal,
                                                               layout=layout,
                                                               is_fp8=True,
                                                               fp8_kv=True)

    o = torch.empty_like(q)

    tri_out, _ = triton_attention_rocm(q, k_quantized, v_quantized, o,
                                       input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd_fp8_kv(q, k_quantized, v_quantized)

    torch.testing.assert_close(ref_out, tri_out, atol=3e-2, rtol=8e-1)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype):
    current_platform.seed_everything(0)
    q, k, v, input_metadata = input_helper(Z,
                                           H,
                                           H,
                                           N_CTX_Q,
                                           N_CTX_K,
                                           D_HEAD,
                                           dtype,
                                           layout='bhsd',
                                           causal=causal,
                                           use_bias=use_bias)
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.fwd(q, k, v)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


# NOTE: Uses thd layout, so also tests thd.
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(1, 48, 256, 64),
                                                 (4, 48, 512, 64),
                                                 (16, 48, 512, 64),
                                                 (64, 48, 128, 128)])
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX,
                                                  D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    triton_attention_rocm(q, k, v, tri_out, input_metadata)

    ref_impl = ReferenceAttention(Z, H, H, N_CTX, N_CTX, D_HEAD, False, dtype,
                                  input_metadata)
    ref_out = ref_impl.varlen_fwd(q, k, v, is_mqa=False)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


# NOTE: Uses thd layout, so also tests thd.
@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD', [(2, 48, 24, 128, 64),
                                                      (4, 48, 12, 256, 64),
                                                      (4, 48, 4, 512, 64),
                                                      (4, 64, 16, 128, 128)])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z,
                           HQ,
                           HK,
                           N_CTX,
                           D_HEAD,
                           causal,
                           dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX,
                                                  D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    triton_attention_rocm(q, k, v, tri_out, input_metadata)

    ref_impl = ReferenceAttention(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, False,
                                  dtype, input_metadata)
    ref_out = ref_impl.varlen_fwd(q, k, v, is_mqa=True)

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)
