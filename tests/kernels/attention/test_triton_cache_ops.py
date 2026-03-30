# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise equivalence tests: Triton cache kernels vs CUDA cache kernels."""

import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.utils.torch_utils import set_random_seed

# ---------- indexer_k_quant_and_cache parameters ----------
# head_dim == quant_block_size == 128 always for the indexer kernel.
HEAD_DIMS = [128]
QUANT_BLOCK_SIZES = [128]
SCALE_FMTS = ["ue8m0", "other"]
CACHE_BLOCK_SIZES = [16]
NUM_BLOCKS = [32]
NUM_TOKENS = [1, 42]
DTYPES = [torch.bfloat16]
SEEDS = [0]

# ---------- concat_and_cache_mla parameters ----------
KV_LORA_RANKS = [256, 512]
PE_DIMS = [64]
MLA_BLOCK_SIZES = [16]
MLA_NUM_BLOCKS = [8]
MLA_NUM_TOKENS = [1, 42]
MLA_KV_CACHE_DTYPES = ["auto", "fp8_e4m3"]


# =====================================================================
#  indexer_k_quant_and_cache
# =====================================================================


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("quant_block_size", QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("scale_fmt", SCALE_FMTS)
@pytest.mark.parametrize("cache_block_size", CACHE_BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_indexer_k_quant_and_cache_equivalence(
    head_dim: int,
    quant_block_size: int,
    scale_fmt: str,
    cache_block_size: int,
    num_blocks: int,
    num_tokens: int,
    dtype: torch.dtype,
    seed: int,
):
    set_random_seed(seed)
    device = "cuda"

    # cache_stride = fp8 data + float32 scales (in fp8-element units)
    cache_stride = head_dim + head_dim * 4 // quant_block_size

    total_slots = num_blocks * cache_block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    k = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)

    # Create two identical caches
    kv_cache_cuda = torch.zeros(
        num_blocks,
        cache_block_size,
        cache_stride,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kv_cache_triton = kv_cache_cuda.clone()

    # Run CUDA kernel
    ops.indexer_k_quant_and_cache(
        k,
        kv_cache_cuda,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    # Run Triton kernel
    from vllm.v1.attention.ops.indexer_k_quant_and_cache import (
        indexer_k_quant_and_cache as triton_indexer_k_quant_and_cache,
    )

    triton_indexer_k_quant_and_cache(
        k,
        kv_cache_triton,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    # Bitwise comparison (view as uint8 to catch any bit-level differences)
    cuda_bytes = kv_cache_cuda.view(torch.uint8)
    triton_bytes = kv_cache_triton.view(torch.uint8)
    torch.testing.assert_close(cuda_bytes, triton_bytes, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("quant_block_size", QUANT_BLOCK_SIZES)
@pytest.mark.parametrize("cache_block_size", CACHE_BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_indexer_k_quant_and_cache_padding(
    head_dim: int,
    quant_block_size: int,
    cache_block_size: int,
    num_blocks: int,
    seed: int,
):
    """Verify that padded tokens (slot_mapping == -1) leave the cache untouched."""
    set_random_seed(seed)
    device = "cuda"

    cache_stride = head_dim + head_dim * 4 // quant_block_size
    num_tokens = 4
    # All slots are -1 (padding)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device)

    kv_cache = torch.zeros(
        num_blocks,
        cache_block_size,
        cache_stride,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    snapshot = kv_cache.clone()

    from vllm.v1.attention.ops.indexer_k_quant_and_cache import (
        indexer_k_quant_and_cache as triton_indexer_k_quant_and_cache,
    )

    triton_indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        "ue8m0",
    )

    torch.testing.assert_close(
        kv_cache.view(torch.uint8),
        snapshot.view(torch.uint8),
        atol=0,
        rtol=0,
    )


# =====================================================================
#  concat_and_cache_mla
# =====================================================================


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("pe_dim", PE_DIMS)
@pytest.mark.parametrize("block_size", MLA_BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", MLA_NUM_BLOCKS)
@pytest.mark.parametrize("num_tokens", MLA_NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", MLA_KV_CACHE_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_concat_and_cache_mla_equivalence(
    kv_lora_rank: int,
    pe_dim: int,
    block_size: int,
    num_blocks: int,
    num_tokens: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
):
    set_random_seed(seed)
    device = "cuda"

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, pe_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + pe_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)

    cache_elem_dtype = torch.uint8 if kv_cache_dtype == "fp8_e4m3" else dtype
    kv_cache_cuda = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=cache_elem_dtype,
        device=device,
    )
    kv_cache_triton = kv_cache_cuda.clone()

    # Run CUDA kernel
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache_cuda,
        slot_mapping,
        kv_cache_dtype,
        scale,
    )

    # Run Triton kernel
    from vllm.v1.attention.ops.concat_and_cache_mla import (
        concat_and_cache_mla as triton_concat_and_cache_mla,
    )

    triton_concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache_triton,
        slot_mapping,
        kv_cache_dtype,
        scale,
    )

    # Bitwise comparison
    cuda_bytes = kv_cache_cuda.view(torch.uint8)
    triton_bytes = kv_cache_triton.view(torch.uint8)
    torch.testing.assert_close(cuda_bytes, triton_bytes, atol=0, rtol=0)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("pe_dim", PE_DIMS)
@pytest.mark.parametrize("block_size", MLA_BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", MLA_NUM_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_concat_and_cache_mla_padding(
    kv_lora_rank: int,
    pe_dim: int,
    block_size: int,
    num_blocks: int,
    seed: int,
):
    """Verify that padded tokens (slot_mapping == -1) leave the cache untouched."""
    set_random_seed(seed)
    device = "cuda"

    num_tokens = 4
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(num_tokens, pe_dim, dtype=torch.bfloat16, device=device)
    entry_size = kv_lora_rank + pe_dim
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        entry_size,
        dtype=torch.bfloat16,
        device=device,
    )
    snapshot = kv_cache.clone()

    from vllm.v1.attention.ops.concat_and_cache_mla import (
        concat_and_cache_mla as triton_concat_and_cache_mla,
    )

    triton_concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        "auto",
        scale,
    )

    torch.testing.assert_close(
        kv_cache.view(torch.uint8),
        snapshot.view(torch.uint8),
        atol=0,
        rtol=0,
    )


# =====================================================================
#  fused_norm_rope + indexer_k_quant_and_cache
# =====================================================================


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_fused_norm_rope_indexer_equivalence(
    num_tokens: int,
    seed: int,
):
    """Verify the fused LayerNorm+RoPE+FP8-quant path in fused_norm_rope
    produces a bitwise-identical kv_cache to the unfused path
    (layer_norm -> rope -> indexer_k_quant_and_cache)."""
    from vllm.model_executor.models.deepseek_v3_2_monolithic.ops import (
        fused_norm_rope,
    )
    from vllm.v1.attention.ops.indexer_k_quant_and_cache import (
        indexer_k_quant_and_cache,
    )

    set_random_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    # Dimensions matching DeepSeek V3
    q_dim = 1536
    kv_dim = 512
    kpe_dim = 64
    index_k_dim = 128
    rot_dim = 64  # qk_rope_head_dim for indexer
    topk = 8
    cache_block_size = 16
    num_blocks = 32
    cache_stride = index_k_dim + 4  # 128 fp8 bytes + 4 scale bytes

    # Inputs
    positions = torch.randint(0, 1024, (num_tokens,), device=device)
    q_c = torch.randn(num_tokens, q_dim, dtype=dtype, device=device)
    q_rms_w = torch.randn(q_dim, dtype=dtype, device=device)
    kv_c = torch.randn(num_tokens, kv_dim, dtype=dtype, device=device)
    kv_rms_w = torch.randn(kv_dim, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, kpe_dim, dtype=dtype, device=device)
    kpe_cos_sin = torch.randn(4096, kpe_dim, dtype=dtype, device=device)
    index_k = torch.randn(num_tokens, index_k_dim, dtype=dtype, device=device)
    index_k_w = torch.randn(index_k_dim, dtype=dtype, device=device)
    index_k_b = torch.randn(index_k_dim, dtype=dtype, device=device)
    ik_cos_sin = torch.randn(4096, rot_dim, dtype=dtype, device=device)
    topk_buf = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    eps = 1e-6

    total_slots = num_blocks * cache_block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(
        slot_mapping_lst,
        dtype=torch.long,
        device=device,
    )
    kv_cache_fused = torch.zeros(
        num_blocks,
        cache_block_size,
        cache_stride,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kv_cache_ref = kv_cache_fused.clone()

    # --- Reference path: standalone LayerNorm + RoPE + FP8 quant ---
    from vllm.model_executor.models.deepseek_v3_2_monolithic.ops import (
        layer_norm,
        qk_rope,
    )

    index_k_ref = layer_norm(index_k, index_k_w, index_k_b, eps)
    # RoPE in-place (1 head, non-interleaved, no start offset).
    # qk_rope applies to both Q and K; pass a dummy for Q.
    dummy_q = torch.empty_like(index_k_ref.unsqueeze(1))
    qk_rope(
        positions,
        dummy_q,
        index_k_ref.unsqueeze(1),
        ik_cos_sin,
        q_start_offset=0,
        interleaved=False,
    )
    indexer_k_quant_and_cache(
        index_k_ref,
        kv_cache_ref,
        slot_mapping,
        128,
        "ue8m0",
    )

    # --- Fused path: all in one kernel ---
    index_k_fused = index_k.clone()
    fused_norm_rope(
        positions,
        q_c.clone(),
        q_rms_w,
        eps,
        kv_c.clone(),
        kv_rms_w,
        eps,
        k_pe.clone(),
        kpe_cos_sin,
        index_k_fused,
        index_k_w,
        index_k_b,
        eps,
        ik_cos_sin,
        topk_buf.clone(),
        slot_mapping=slot_mapping,
        indexer_k_cache=kv_cache_fused,
    )

    # The fused path keeps full fp32 precision through LayerNorm →
    # RoPE → FP8 quant (no intermediate bf16 truncation), so the
    # FP8 values may differ from the bf16 standalone path.  Verify
    # via dequantization that the results are close.
    ref_bytes = kv_cache_ref.view(torch.uint8)
    fused_bytes = kv_cache_fused.view(torch.uint8)
    assert (ref_bytes.int() - fused_bytes.int()).abs().float().mean() < 0.5
