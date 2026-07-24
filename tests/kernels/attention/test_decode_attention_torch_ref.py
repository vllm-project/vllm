# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the Triton decode attention kernels.

Covers the three decode-attention kernels in
vllm/v1/attention/ops/triton_decode_attention.py:

  * _fwd_kernel_stage1          via decode_attention_fwd_normal  (MHA)
  * _fwd_grouped_kernel_stage1  via decode_attention_fwd_grouped (GQA)
  * _fwd_kernel_stage2          the softmax/reduce-V stage used by both

The pure-PyTorch implementation is the oracle, so the test is device-agnostic
and runs wherever current_platform points (CUDA, ROCm or XPU).

Note: decode_attention_fwd_normal / decode_attention_fwd_grouped do not
default k_scale / v_scale, and the kernels call tl.load(k_scale) /
tl.load(v_scale) unconditionally, so both must be passed as torch.Tensor
pointers (not Python floats / None).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)

# decode_attention_fwd_normal / decode_attention_fwd_grouped dispatch
# Triton kernels that require a GPU-class backend.
if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "decode_attention Triton kernels require a CUDA-alike "
        "or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type


def torch_decode_attention(q, k_buffer, v_buffer, req_to_tokens, b_seqlen,
                           sm_scale, page_size):
    """Naive float32 PyTorch decode-attention reference (the oracle).

    q:             (B, H_q, D)
    k_buffer:      (total_pages, page_size, H_kv, D)
    v_buffer:      (total_pages, page_size, H_kv, D)
    req_to_tokens: (B, max_pages)  -- page indices per request
    b_seqlen:      (B,)

    Returns (out, lse):
      out: (B, H_q, D)  attention output
      lse: (B, H_q)     natural-log sum-exp of the (scaled) scores
    """
    B, H_q, D = q.shape
    H_kv = k_buffer.shape[2]
    kv_group = H_q // H_kv

    out = torch.zeros(B, H_q, D, device=q.device, dtype=q.dtype)
    lse = torch.zeros(B, H_q, device=q.device, dtype=torch.float32)
    for b in range(B):
        seq_len = int(b_seqlen[b].item())
        num_pages = (seq_len + page_size - 1) // page_size

        page_ids = req_to_tokens[b, :num_pages]
        k_flat = k_buffer[page_ids].reshape(-1, H_kv, D)[:seq_len]
        v_flat = v_buffer[page_ids].reshape(-1, H_kv, D)[:seq_len]

        for h in range(H_q):
            kv_h = h // kv_group
            scores = (q[b, h].float() @ k_flat[:, kv_h].float().T) * sm_scale
            lse[b, h] = torch.logsumexp(scores, dim=-1)
            weights = torch.softmax(scores, dim=-1).to(v_flat.dtype)
            out[b, h] = weights @ v_flat[:, kv_h]
    return out, lse


def _make_inputs(B, H_q, H_kv, D_head, seq_len, num_kv_splits, page_size,
                 shuffle_pages=False):
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    total_pages = B * num_pages_per_req

    torch.manual_seed(42)
    q = torch.randn(B, H_q, D_head, device=DEVICE, dtype=torch.float16)
    k_buffer = torch.randn(total_pages, page_size, H_kv, D_head,
                           device=DEVICE, dtype=torch.float16) / 10
    v_buffer = torch.randn(total_pages, page_size, H_kv, D_head,
                           device=DEVICE, dtype=torch.float16) / 10
    att_out = torch.zeros(B, H_q, num_kv_splits, D_head + 1,
                          device=DEVICE, dtype=torch.float32)
    o = torch.zeros(B, H_q, D_head, device=DEVICE, dtype=torch.float16)
    lse = torch.zeros(B, H_q, device=DEVICE, dtype=torch.float32)

    # Page-indirection table. When shuffled, the kernel's paged gather is
    # genuinely exercised rather than mapping to a contiguous slice.
    page_order = torch.randperm(total_pages) if shuffle_pages \
        else torch.arange(total_pages)
    req_to_tokens = page_order.to(device=DEVICE, dtype=torch.int32).reshape(
        B, num_pages_per_req)
    b_seqlen = torch.full((B,), seq_len, device=DEVICE, dtype=torch.int32)

    # k_scale/v_scale MUST be tensor pointers: kernel does tl.load(k_scale).
    k_scale = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)
    v_scale = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)
    sm_scale = 1.0 / (D_head ** 0.5)
    return (q, k_buffer, v_buffer, o, lse, req_to_tokens, b_seqlen, att_out,
            k_scale, v_scale, sm_scale)


BATCH_SIZES = [1, 4]
D_HEADS = [64]
SEQ_LENS = [32, 64]
NUM_KV_SPLITS = [4]
PAGE_SIZE = 16


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("D_head", D_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_kv_splits", NUM_KV_SPLITS)
@pytest.mark.parametrize("shuffle_pages", [False, True])
@torch.inference_mode()
def test_decode_attention_normal(B, D_head, seq_len, num_kv_splits,
                                 shuffle_pages):
    """MHA path: _fwd_kernel_stage1 + _fwd_kernel_stage2."""
    H_q = H_kv = 8
    (q, k_buffer, v_buffer, o, lse, req_to_tokens, b_seqlen, att_out,
     k_scale, v_scale, sm_scale) = _make_inputs(
        B, H_q, H_kv, D_head, seq_len, num_kv_splits, PAGE_SIZE, shuffle_pages)

    decode_attention_fwd_normal(
        q, k_buffer, v_buffer, o, lse,
        req_to_tokens, b_seqlen, att_out,
        num_kv_splits, sm_scale, PAGE_SIZE,
        logit_cap=0.0, k_scale=k_scale, v_scale=v_scale,
    )

    ref_o, ref_lse = torch_decode_attention(
        q, k_buffer, v_buffer, req_to_tokens, b_seqlen, sm_scale, PAGE_SIZE)
    torch.testing.assert_close(o, ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("B", BATCH_SIZES)
@pytest.mark.parametrize("D_head", D_HEADS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_kv_splits", NUM_KV_SPLITS)
@pytest.mark.parametrize("shuffle_pages", [False, True])
@torch.inference_mode()
def test_decode_attention_grouped(B, D_head, seq_len, num_kv_splits,
                                  shuffle_pages):
    """GQA path: _fwd_grouped_kernel_stage1 + _fwd_kernel_stage2."""
    H_q, H_kv = 8, 2
    (q, k_buffer, v_buffer, o, lse, req_to_tokens, b_seqlen, att_out,
     k_scale, v_scale, sm_scale) = _make_inputs(
        B, H_q, H_kv, D_head, seq_len, num_kv_splits, PAGE_SIZE, shuffle_pages)

    decode_attention_fwd_grouped(
        q, k_buffer, v_buffer, o, lse,
        req_to_tokens, b_seqlen, att_out,
        num_kv_splits, sm_scale, PAGE_SIZE,
        logit_cap=0.0, k_scale=k_scale, v_scale=v_scale,
    )

    ref_o, ref_lse = torch_decode_attention(
        q, k_buffer, v_buffer, req_to_tokens, b_seqlen, sm_scale, PAGE_SIZE)
    torch.testing.assert_close(o, ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-2, rtol=2e-2)
