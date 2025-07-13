# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from vllm.tests.kernels.test_cutlass_mla_decode import ref_mla

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="FlashInfer MLA Requires compute capability of 10 or above.",
        allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mean_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
def test_cutlass_mla_decode(dtype: torch.dtype, mean_seq_len: int, bs: int,
                            block_size: int):
    torch.set_default_dtype(dtype)
    torch.set_default_device('cuda')
    torch.manual_seed(42)

    # Deepseek R1 config
    num_heads = 128
    kv_lora_rank = 512
    v_head_dim = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    scale = qk_head_dim**(-0.5)

    seq_lens = torch.empty(bs).normal_(mean_seq_len, mean_seq_len / 2)
    seq_lens = seq_lens.clip(2).to(torch.int32)
    max_seq_len = seq_lens.max().item()

    # Generate block tables with random but unique block IDs
    # From https://github.com/flashinfer-ai/flashinfer/pull/1222
    blocks_per_seq = (seq_lens + block_size - 1) // block_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()
    total_blocks_needed = sum(blocks_per_seq)
    # Get random unique IDs for all blocks
    all_block_ids = torch.randperm(total_blocks_needed)

    block_id = 0
    block_tables = torch.zeros(
        (bs, max_num_blocks_per_seq),
        dtype=torch.int32,
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(bs):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    kv_cache = torch.randn(block_tables.numel(), block_size, qk_head_dim)
    q = torch.randn(bs, num_heads, qk_head_dim)

    out_ref = q.new_zeros(bs, num_heads, v_head_dim)
    ref_mla(out_ref, q, kv_cache, scale, block_tables, seq_lens)
    out_ans = torch.zeros_like(out_ref)

    workspace_buffer = torch.empty(
        FLASHINFER_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device=q.device,
    )

    trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        max_seq_len=max_seq_len,
        scale=scale,
        out=out_ans,
    )

    torch.testing.assert_close(out_ans, out_ref, atol=1e-2, rtol=1e-2)
