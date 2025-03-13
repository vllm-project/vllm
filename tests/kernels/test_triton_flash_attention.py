# SPDX-License-Identifier: Apache-2.0
"""Tests for the triton_flash_attention kernel

Run `pytest tests/kernels/test_triton_flash_attention.py`.
"""
import pytest
import torch

from vllm.attention.ops.triton_flash_attention import (SUPPORTED_LAYOUTS,
                                                       MetaData,
                                                       compute_alibi_tensor,
                                                       triton_attention_rocm)
from vllm.platforms import current_platform


FP8_DTYPE_TORCH = torch.float8_e4m3fnuz

float8_info = torch.finfo(FP8_DTYPE_TORCH)
FP8_MIN = float8_info.min
FP8_MAX = float8_info.max

def get_shape_from_layout(q, k, metadata):
    assert metadata.layout in SUPPORTED_LAYOUTS, "Got unsupported layout."
    if metadata.layout == 'thd':
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    return batch, nheads_q, nheads_k, head_size


def quantize_fp8(tensor: torch.Tensor,
                  dim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_vals = tensor.abs().amax(
        dim=[i for i in range(tensor.dim()) if i != dim], keepdim=True)

    # Avoid division by zero
    max_vals[max_vals == 0] = 1e-8

    # Compute scale factors for each channel
    scale = (FP8_MAX / max_vals).clamp(1e-12)

    # Quantize the tensor
    tensor = tensor * scale
    tensor.clamp_(FP8_MIN, FP8_MAX)
    tensor_quantized = tensor.to(FP8_DTYPE_TORCH)

    return tensor_quantized, scale, 1 / scale


def quantize_input(q, k, v, input_metadata: MetaData, fp8_kv=False):
    is_supported_layout = input_metadata.layout in SUPPORTED_LAYOUTS
    assert is_supported_layout, "Got unsupported layout."
    if input_metadata.layout == 'bhsd':
        quantization_dim = 1
    elif input_metadata.layout == 'bshd':
        quantization_dim = 2

    q_descale = None
    if not fp8_kv:
        q, _, q_descale = quantize_fp8(q, dim=quantization_dim)
    k, _, k_descale = quantize_fp8(k, dim=quantization_dim)
    v, _, v_descale = quantize_fp8(v, dim=quantization_dim)

    # In real world use case, the p scale would be a parameter trained by the
    # model.
    p_scale = p_descale = None

    # We are not multiplying the scales together to get
    # qk_desale / o_descale e.g.
    # qk_desale = q_descale * k_descale
    # o_desale = p_descale * v_descale
    # it results in very small fp e.g. 0,0002, losing precision.
    # They are applied on the run.
    input_metadata.set_eight_bit_params(
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        # By default p_scaling is not enabled
        p_scale=p_scale,
        p_descale=p_descale,
        o_scale=None)

    return q, k, v


def input_helper(Z,
                 HQ,
                 HK,
                 N_CTX_Q,
                 N_CTX_K,
                 D_HEAD,
                 dtype,
                 layout,
                 requires_grad=True):
    assert layout in SUPPORTED_LAYOUTS, "Got unsupported layout."

    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    q = torch.randn(q_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=requires_grad)
    k = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=requires_grad)
    v = torch.randn(k_tensor_shape,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=requires_grad)

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z,
                        HQ,
                        HK,
                        N_CTX_Q,
                        N_CTX_K,
                        D_HEAD,
                        dtype,
                        equal_seqlens=False):
    torch.manual_seed(20)

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
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
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
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts
        # 2^(-8/n)
        alibi_slopes = torch.tensor(
            [2**(-8 / HQ * i) for i in range(1, HQ + 1)],
            dtype=torch.float32,
            device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

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

    scores = torch.einsum('bhqd,bhkd->bhqk', q,
                          k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"),
                          diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going
        # into the softmax. This produces a row of NaNs as -inf - -inf == NaN.
        # So we fix this by converting the NaNs to 0s, which is what they
        # should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
@pytest.mark.parametrize('persistent', ['fixed', 'dynamic'])
def test_op_persistent_fwd(Z,
                           HQ,
                           HK,
                           N_CTX_Q,
                           N_CTX_K,
                           D_HEAD,
                           causal,
                           use_alibi,
                           layout,
                           persistent,
                           dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts
        # 2^(-8/n)
        alibi_slopes = torch.tensor(
            [2**(-8 / HQ * i) for i in range(1, HQ + 1)],
            dtype=torch.float32,
            device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    input_metadata.set_persistent(persistent)

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

    scores = torch.einsum('bhqd,bhkd->bhqk', q,
                          k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"),
                          diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going
        # into the softmax. This produces a row of NaNs as -inf - -inf == NaN.
        # So we fix this by converting the NaNs to 0s, which is what they
        # should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 4, 128, 128, 65),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_fwd_fp8(Z,
                    H,
                    N_CTX_Q,
                    N_CTX_K,
                    D_HEAD,
                    causal,
                    layout,
                    dtype=torch.float16):
    torch.manual_seed(20)

    # Disable grad to save memory it won't run into OOM on CI machine.
    q, k, v, input_metadata = input_helper(Z,
                                           H,
                                           H,
                                           N_CTX_Q,
                                           N_CTX_K,
                                           D_HEAD,
                                           dtype,
                                           layout,
                                           requires_grad=False)
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)

    q_quantized, k_quantized, v_quantized = quantize_input(
        q, k, v, input_metadata)

    tri_out, _ = triton_attention_rocm(q_quantized, k_quantized, v_quantized,
                                       o, input_metadata)

    q_descale, k_descale, v_descale = (input_metadata.q_descale,
                                       input_metadata.k_descale,
                                       input_metadata.v_descale)

    q = q_quantized.to(torch.float16) * q_descale
    k = k_quantized.to(torch.float16) * k_descale
    v = v_quantized.to(torch.float16) * v_descale

    scores = torch.einsum('bhqd,bhkd->bhqk', q,
                          k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"),
                          diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going
        # into the softmax. This produces a row of NaNs as -inf - -inf == NaN.
        # So we fix this by converting the NaNs to 0s, which is what they
        # should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)

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
                        dtype=torch.float16):
    torch.manual_seed(20)

    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD,
                                           dtype, layout)
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)

    _, k_quantized, v_quantized = quantize_input(q,
                                                 k,
                                                 v,
                                                 input_metadata,
                                                 fp8_kv=True)
    k_descale, v_descale = input_metadata.k_descale, input_metadata.v_descale
    k_dequantized = (k_quantized.to(torch.float32)* k_descale.to(torch.float32)).half()
    v_dequantized = (v_quantized.to(torch.float32)* v_descale.to(torch.float32)).half()

    tri_out, _ = triton_attention_rocm(q, k_quantized, v_quantized, o,
                                       input_metadata)

    # Compute scores
    scores = torch.einsum('bhqd,bhkd->bhqk', q,
                          k_dequantized).float() * input_metadata.sm_scale

    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"),
                          diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")

    p = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(),
                           v_dequantized).to(torch.float16)

    if causal:
        nan_mask = torch.isnan(ref_out)
        ref_out[nan_mask] = 0

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    'Z, H, N_CTX_Q, N_CTX_K, D_HEAD',
    [
        (4, 48, 1, 1, 64),
        (4, 48, 1, 1, 128),
        (4, 48, 3, 3, 128),
        (4, 4, 128, 128, 65),
    ])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype):
    torch.manual_seed(20)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    q, k, v, input_metadata = input_helper(Z,
                                           H,
                                           H,
                                           N_CTX_Q,
                                           N_CTX_K,
                                           D_HEAD,
                                           dtype,
                                           layout='bhsd')
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    if causal:
        input_metadata.need_causal()
    if use_bias:
        bias = torch.randn((1, H, N_CTX_Q, N_CTX_K),
                           dtype=dtype,
                           device="cuda")
        input_metadata.need_bias(bias, Z, H, N_CTX_Q, N_CTX_K)
    else:
        bias = None
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _ = triton_attention_rocm(q, k, v, o, input_metadata)
    # reference implementation:171

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"),
                          diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_bias:
        scores += input_metadata.bias
    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going
        # into the softmax. This produces a row of NaNs as -inf - -inf == NaN.
        # So we fix this by converting the NaNs to 0s, which is what they
        # should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0

    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.to(dtype), v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [ (4, 48, 256, 64),
                                                 (4, 48, 512, 64),
                                                 (4, 48, 128, 128)])
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX,
                                                  D_HEAD, dtype)

    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = (input_metadata.cu_seqlens_q[i],
                            input_metadata.cu_seqlens_k[i])
        end_q, end_k = (input_metadata.cu_seqlens_q[i + 1],
                        input_metadata.cu_seqlens_k[i + 1])
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q],
                              k[start_k:end_k]).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p,
                                              v[start_k:end_k])
    triton_attention_rocm(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD',
                         [(2, 48, 24, 128, 64), (4, 48, 12, 256, 64),
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
    input_metadata.eight_bit_dtype_torch = FP8_DTYPE_TORCH
    ref_out = torch.empty_like(q)
    tri_out = torch.empty_like(q)
    # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so the
    # size aligns with Q.
    k_ref = k.view(k.shape[0], k.shape[1], 1,
                   k.shape[2]).expand(-1, -1, HQ // HK, -1)
    v_ref = v.view(v.shape[0], v.shape[1], 1,
                   v.shape[2]).expand(-1, -1, HQ // HK, -1)
    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[
            i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[
            i + 1], input_metadata.cu_seqlens_k[i + 1]
        k_curr = k_ref[start_k:end_k]
        k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
        v_curr = v_ref[start_k:end_k]
        v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k_curr).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
    triton_attention_rocm(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)

def main():
    test_op_fwd(4, 48, 12, 1, 1, 64, True, False, 'bhsd')
    test_op_fwd_fp8(4, 48, 1, 1, 64, True, 'bhsd')
if __name__ == "__main__":
    main()
