# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def cal_diff(x: torch.Tensor,
             y: torch.Tensor,
             name: str,
             use_fp8: bool = False) -> None:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(
        (x * x + y * y).sum().item(), 1e-12)
    if (use_fp8):
        assert cos_diff < 1e-4
    else:
        assert cos_diff < 1e-5


CUTLASS_MLA_UNSUPPORTED_REASON = \
    "Cutlass MLA Requires compute capability of 10 or above." \
    if not current_platform.has_device_capability(100) \
    else "Cutlass MLA is supported"


@pytest.mark.skipif(not current_platform.has_device_capability(100),
                    reason=CUTLASS_MLA_UNSUPPORTED_REASON)
@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("mean_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("h_q", [128])
@pytest.mark.parametrize("d", [576])
@pytest.mark.parametrize("dv", [512])
@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("torch_dtype",
                         [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@torch.inference_mode()
def test_cutlass_mla_decode(b, mean_seq_len, h_q, d, dv, block_size, varlen,
                            torch_dtype):
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

    print(f"{b=}, {mean_seq_len=}, {h_q=}, {d=}, {dv=}, "
          f"{block_size=}, {varlen=}, {torch_dtype=}")

    use_fp8 = torch_dtype == torch.float8_e4m3fn

    q_nope_dim = 128
    q_pe_dim = 64
    scale = (q_nope_dim + q_pe_dim)**(-0.5)
    seq_lens = torch.full((b, ), mean_seq_len, dtype=torch.int32)
    if varlen:
        for i in range(b):
            seq_lens[i] = max(
                random.normalvariate(mean_seq_len, mean_seq_len / 2), 2)
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    # Pad block_num so that small blocks can be packed into full 128-sized
    # CUTLASS tiles. One 128-wide tile can hold (128 // block_size) small
    # blocks.
    pack_factor = 128 // block_size
    block_num = ((block_num + pack_factor - 1) // pack_factor) * pack_factor

    # Amplify input values to ensure test coverage of edge cases where CUTLASS
    # kernel errors occur with split_k settings.
    q = torch.randn(b, h_q, d) * 100
    block_table = torch.randint(0,
                                b * block_num, (b, block_num),
                                dtype=torch.int32)

    kv_cache = torch.randn(block_table.numel(), block_size, d)

    init_dtype = q.dtype
    if use_fp8:
        fp8_dtype = torch.float8_e4m3fn
        descale_q = torch.ones((1), dtype=torch.float32)
        descale_k = torch.ones((1), dtype=torch.float32)

        q = q.to(fp8_dtype)
        kv_cache = kv_cache.to(fp8_dtype)
    else:
        descale_q = None
        descale_k = None

    def cutlass_mla():
        out_ans = torch.zeros(b, h_q, dv, dtype=init_dtype)
        q_nope = q[:, :, :dv].clone()
        q_pe = q[:, :, dv:].clone()
        ops.cutlass_mla_decode(out_ans, q_nope, q_pe, kv_cache, seq_lens,
                               block_table, scale)
        return out_ans

    def scaled_dot_product_attention(query, key, value, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // 1, dim=0)  # h_kv = 1 for MLA
        value = value.repeat_interleave(h_q // 1, dim=0)
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
        kv_cache_ = (kv_cache.to(torch.float) *
                     descale_k).to(init_dtype) if use_fp8 else kv_cache
        out_ref = torch.zeros(b, h_q, dv, dtype=torch.float32)

        for i in range(b):
            # gather and flatten KV-cache
            kv = kv_cache_[
                block_table[i]]  # (max_num_blocks, block_size, head_dim)
            kv = kv.view(1, -1, d)[:, :seq_lens[i]]  # (1, seq_len, head_dim)
            v = kv[:, :, :dv]

            q_batch = q_[i].view(1, h_q, d)
            out_i, _ = scaled_dot_product_attention(
                q_batch.transpose(0, 1),  # (h_q, 1, d)
                kv.transpose(0, 1),  # (seq_len, 1, d) 
                v.transpose(0, 1),  # (seq_len, 1, dv)
                is_causal=False)
            out_ref[i] = out_i.transpose(0, 1).view(h_q, dv)

        return out_ref

    out_cutlass = cutlass_mla()
    out_torch = ref_mla()
    cal_diff(out_cutlass, out_torch, "out", use_fp8)

    t = triton.testing.do_bench(cutlass_mla)
    total_seq_len = seq_lens.sum().item()
    FLOPS = total_seq_len * h_q * (d + dv) * 2
    bytes = (total_seq_len * d +
             b * h_q * d) * (torch.finfo(torch_dtype).bits // 8) + (
                 b * h_q * dv) * (torch.finfo(init_dtype).bits // 8)
    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS,",
          f"{bytes / 10 ** 6 / t:.0f} GB/s")
