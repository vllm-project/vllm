# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.supports_fp8():
    pytest.skip(
        "MiniMax-M3 FP8 sparse attention scale tests require an FP8 GPU.",
        allow_module_level=True,
    )

if current_platform.is_rocm():
    from vllm.models.minimax_m3.amd.ops.sparse_attn import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )
else:
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_sparse_attn,
        minimax_m3_sparse_attn_decode,
    )


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 128
NUM_KV_HEADS = 1
NUM_HEADS = 2
K_SCALE = 0.25
V_SCALE = 0.5


def _scale_tensors(mode: str, num_blocks: int):
    if mode == "scalar":
        k_scale = torch.tensor(K_SCALE, dtype=torch.float32, device=DEVICE)
        v_scale = torch.tensor(V_SCALE, dtype=torch.float32, device=DEVICE)
    else:
        shape = (NUM_KV_HEADS, num_blocks * SPARSE_BLOCK_SIZE)
        k_scale = torch.full(shape, K_SCALE, dtype=torch.float32, device=DEVICE)
        v_scale = torch.full(shape, V_SCALE, dtype=torch.float32, device=DEVICE)
    return k_scale, v_scale


def _make_kv_cache(num_blocks: int, seed: int):
    torch.manual_seed(seed)
    fp8_dtype = current_platform.fp8_dtype()
    kv_ref = torch.randn(
        num_blocks,
        NUM_KV_HEADS,
        SPARSE_BLOCK_SIZE,
        2 * HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    kv_fp8 = torch.empty_like(kv_ref, dtype=fp8_dtype)
    kv_fp8[..., :HEAD_DIM] = (kv_ref[..., :HEAD_DIM].float() / K_SCALE).to(fp8_dtype)
    kv_fp8[..., HEAD_DIM:] = (kv_ref[..., HEAD_DIM:].float() / V_SCALE).to(fp8_dtype)

    kv_dequant = torch.empty_like(kv_ref)
    kv_dequant[..., :HEAD_DIM] = (kv_fp8[..., :HEAD_DIM].float() * K_SCALE).to(DTYPE)
    kv_dequant[..., HEAD_DIM:] = (kv_fp8[..., HEAD_DIM:].float() * V_SCALE).to(DTYPE)
    return kv_fp8, kv_dequant


@pytest.mark.parametrize("scale_mode", ["scalar", "per_token_head"])
@torch.inference_mode()
def test_minimax_m3_sparse_prefill_fp8_kv_scales(scale_mode: str):
    total_q = 17
    num_blocks = 1
    torch.manual_seed(0)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    kv_fp8, kv_dequant = _make_kv_cache(num_blocks, seed=1)
    k_scale, v_scale = _scale_tensors(scale_mode, num_blocks)

    topk = torch.zeros(NUM_KV_HEADS, total_q, 1, dtype=torch.int32, device=DEVICE)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([total_q], dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    got = torch.empty_like(q)
    ref = torch.empty_like(q)
    unscaled = torch.empty_like(q)

    minimax_m3_sparse_attn(
        q,
        kv_fp8,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        got,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    minimax_m3_sparse_attn(
        q,
        kv_dequant,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        ref,
    )
    minimax_m3_sparse_attn(
        q,
        kv_fp8,
        topk,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        total_q,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        unscaled,
    )

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(unscaled, ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("scale_mode", ["scalar", "per_token_head"])
@torch.inference_mode()
def test_minimax_m3_sparse_decode_fp8_kv_scales(scale_mode: str):
    total_q = 2
    num_blocks = 1
    torch.manual_seed(2)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    kv_fp8, kv_dequant = _make_kv_cache(num_blocks, seed=3)
    k_scale, v_scale = _scale_tensors(scale_mode, num_blocks)

    topk = torch.zeros(NUM_KV_HEADS, total_q, 1, dtype=torch.int32, device=DEVICE)
    block_table = torch.zeros(total_q, 1, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor([64, 128], dtype=torch.int32, device=DEVICE)
    got = torch.empty_like(q)
    ref = torch.empty_like(q)
    unscaled = torch.empty_like(q)

    minimax_m3_sparse_attn_decode(
        q,
        kv_fp8,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        got,
        decode_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    minimax_m3_sparse_attn_decode(
        q,
        kv_dequant,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        ref,
        decode_query_len=1,
    )
    minimax_m3_sparse_attn_decode(
        q,
        kv_fp8,
        topk,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        unscaled,
        decode_query_len=1,
    )

    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(unscaled, ref, rtol=1e-1, atol=1e-1)
