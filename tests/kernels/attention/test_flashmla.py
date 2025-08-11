# Adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random

import pytest
import torch

from vllm.attention.ops.flashmla import (flash_mla_with_kvcache,
                                         get_mla_metadata,
                                         is_flashmla_supported)
from vllm.triton_utils import triton


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max(
        (x * x + y * y).sum().item(), 1e-12)
    assert cos_diff < 1e-5

FLASH_MLA_UNSUPPORTED_REASON = is_flashmla_supported()[1] \
    if not is_flashmla_supported()[0] else "FlashMLA is supported"


@pytest.mark.skipif(not is_flashmla_supported()[0],
                    reason=FLASH_MLA_UNSUPPORTED_REASON)
@pytest.mark.parametrize("b", [128])
@pytest.mark.parametrize("s_q", [1, 2])
@pytest.mark.parametrize("mean_sk", [4096, 8192])
@pytest.mark.parametrize("h_q", [16, 32, 64, 128])
@pytest.mark.parametrize("h_kv", [1])
@pytest.mark.parametrize("d", [576])
@pytest.mark.parametrize("dv", [512])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("varlen", [False, True])
@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, block_size, causal,
                   varlen):
    # TODO: parametrize using pytest
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, "
          f"{d=}, {dv=}, {causal=}, {varlen=}")

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
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv,
                       d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
        )

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
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            ref_O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                is_causal=causal,
            )
            out[i] = ref_O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    cal_diff(out_flash, out_torch, "out")
    cal_diff(lse_flash, lse_torch, "lse")

    t = triton.testing.do_bench(flash_mla)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d +
             b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} "
          f"TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s")
