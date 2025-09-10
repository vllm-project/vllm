# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random
from typing import Optional

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def cal_diff(x: torch.Tensor,
             y: torch.Tensor,
             name: str,
             use_fp8: bool = False,
             diff_threshold: Optional[float] = None) -> None:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(
        (x * x + y * y).sum().item(), 1e-12)
    if diff_threshold is not None:
        # directly compare the cos_diff with the threshold
        assert cos_diff < diff_threshold
    else:
        # use the default threshold
        if (use_fp8):
            assert cos_diff < 1e-4
        else:
            assert cos_diff < 1e-5


CUTLASS_MLA_UNSUPPORTED_REASON = \
    "Cutlass MLA Requires compute capability of 10 or above." \
    if not current_platform.is_device_capability(100) \
    else "Cutlass MLA is supported"


@pytest.mark.skipif(not current_platform.has_device_capability(100),
                    reason=CUTLASS_MLA_UNSUPPORTED_REASON)
@pytest.mark.parametrize("b", [128])
@pytest.mark.parametrize("s_q", [1])
@pytest.mark.parametrize("mean_sk", [4096, 8192, 16384])
@pytest.mark.parametrize("h_q", [16, 32, 64, 128])
@pytest.mark.parametrize("h_kv", [1])
@pytest.mark.parametrize("d", [576])
@pytest.mark.parametrize("dv", [512])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.bfloat16,
        # fp8 can have occasional precision-related failures.
        pytest.param(torch.float8_e4m3fn, marks=pytest.mark.flaky(reruns=2))
    ])
@torch.inference_mode()
def test_cutlass_mla_decode(b, s_q, mean_sk, h_q, h_kv, d, dv, block_size,
                            causal, varlen, torch_dtype):
    device = torch.device("cuda:0")
    if torch_dtype == torch.float8_e4m3fn:
        init_dtype = torch.bfloat16
    else:
        init_dtype = torch_dtype
    torch.set_default_dtype(init_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    random.seed(42)

    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, "
          f"{d=}, {dv=}, {causal=}, {varlen=}, {torch_dtype=}")

    use_fp8 = torch_dtype == torch.float8_e4m3fn
    scale = math.sqrt(d)**(-1)
    cache_seqlens = torch.full((b, ), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2),
                                   s_q)
    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn(b, s_q, h_q, d)
    block_table = torch.arange(b * max_seqlen_pad // block_size,
                               dtype=torch.int32).view(
                                   b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = blocked_k[..., :dv]

    init_dtype = q.dtype
    if use_fp8:
        fp8_dtype = torch.float8_e4m3fn
        descale_q = torch.ones((1), dtype=torch.float32)
        descale_k = torch.ones((1), dtype=torch.float32)

        q = q.to(fp8_dtype)
        blocked_k = blocked_k.to(fp8_dtype)
        blocked_v = blocked_v.to(fp8_dtype)
    else:
        descale_q = None
        descale_k = None

    def cutlass_mla():
        MAX_HEADS = 128

        q_reshaped = q.squeeze(1)
        q_nope = q_reshaped[:, :, :dv].clone()
        q_pe = q_reshaped[:, :, dv:].clone()

        if h_q < MAX_HEADS:
            q_nope_padded = q_nope.new_empty((b, MAX_HEADS, dv))
            q_nope_padded[:, :h_q] = q_nope
            q_nope = q_nope_padded

            q_pe_padded = q_pe.new_empty((b, MAX_HEADS, d - dv))
            q_pe_padded[:, :h_q] = q_pe
            q_pe = q_pe_padded

        kv_cache_flat = blocked_k.squeeze(2)
        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda:0"))
        sm_count = device_properties.multi_processor_count
        workspace_size = ops.sm100_cutlass_mla_get_workspace_size(
            max_seqlen * block_size, b, sm_count, num_kv_splits=1)
        workspace = torch.empty(workspace_size,
                                device="cuda",
                                dtype=torch.uint8)

        out_ans = torch.empty(b, MAX_HEADS, dv, dtype=init_dtype)
        output_lse = torch.empty((b, MAX_HEADS),
                                 dtype=torch.float32,
                                 device=q_nope.device)
        ops.sm100_cutlass_mla_decode(out_ans, output_lse, q_nope, q_pe,
                                     kv_cache_flat, cache_seqlens, block_table,
                                     workspace, scale, 1)
        return out_ans[:, :h_q].contiguous(), output_lse[:, :h_q].contiguous()

    def scaled_dot_product_attention(query, key, value, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
            temp_mask = torch.ones(s_q, s_k,
                                   dtype=torch.bool).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla():
        q_ = (q.to(torch.float) * descale_q).to(init_dtype) if use_fp8 else q
        blocked_k_ = (blocked_k.to(torch.float) *
                      descale_k).to(init_dtype) if use_fp8 else blocked_k
        blocked_v_ = (blocked_v.to(torch.float) *
                      descale_k).to(init_dtype) if use_fp8 else blocked_v
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            out_i, lse_i = scaled_dot_product_attention(
                q_[i].transpose(0, 1),
                blocked_k_.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v_.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                is_causal=causal,
            )
            out[i] = out_i.transpose(0, 1)
            lse[i] = lse_i
        return out, lse

    out_cutlass, lse_cutlass = cutlass_mla()
    out_torch, lse_torch = ref_mla()
    # Extract the single token (s_q=1) slice to match cutlass output shape
    out_torch_slice = out_torch[:, 0, :, :]  # [b, h_q, dv]
    lse_torch_slice = lse_torch[:, 0, :]  # [b, h_q]
    cal_diff(out_cutlass, out_torch_slice, "out", use_fp8)
    # lse has larger numerical error, so use a larger threshold
    cal_diff(lse_cutlass, lse_torch_slice, "lse", use_fp8, diff_threshold=1e-3)

    t = triton.testing.do_bench(cutlass_mla)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d +
             b * s_q * h_q * d) * (torch.finfo(torch_dtype).bits // 8) + (
                 b * s_q * h_q * dv) * (torch.finfo(init_dtype).bits // 8)
    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS,",
          f"{bytes / 10 ** 6 / t:.0f} GB/s")
