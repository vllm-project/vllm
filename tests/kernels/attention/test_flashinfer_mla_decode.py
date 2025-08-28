# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from torch import Tensor

from vllm.platforms import current_platform

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="FlashInfer MLA Requires compute capability of 10 or above.",
        allow_module_level=True)


def ref_mla(
        out: Tensor,  # (bs, num_heads, v_head_dim)
        query: Tensor,  # (bs, num_heads, head_dim)
        kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
        scale: float,
        block_tables: Tensor,  # (bs, max_num_blocks)
        seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[
            block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1,
                     head_dim)[:, :seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q,
                                           kv,
                                           v,
                                           scale=scale,
                                           enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
def test_flashinfer_mla_decode(dtype: torch.dtype, bs: int, block_size: int):
    torch.set_default_device('cuda')
    torch.manual_seed(42)

    # Deepseek R1 config
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    scale = (qk_nope_head_dim + qk_rope_head_dim)**-0.5

    MAX_SEQ_LEN = 1024

    seq_lens = [torch.randint(2, MAX_SEQ_LEN, (1, )).item() for _ in range(bs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Generate block tables with random but unique block IDs
    # From https://github.com/flashinfer-ai/flashinfer/pull/1222
    blocks_per_seq = (seq_lens_tensor + block_size - 1) // block_size
    max_num_blocks_per_seq = max(blocks_per_seq.max().item(), 4)
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
        block_tables[i, :num_blocks_needed] = all_block_ids[block_id:block_id +
                                                            num_blocks_needed]
        block_id += num_blocks_needed

    kv_cache = torch.randn(block_tables.numel(), block_size,
                           qk_head_dim).to(dtype)
    q = torch.randn(bs, num_heads, qk_head_dim).to(dtype)

    out_ref = q.new_zeros(bs, num_heads, kv_lora_rank)
    ref_mla(out_ref, q, kv_cache, scale, block_tables, seq_lens_tensor)

    workspace_buffer = torch.zeros(
        FLASHINFER_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device=q.device,
    )
    # Flashinfer MLA expects the query to be of shape
    # (bs, q_len_per_request, num_heads, qk_head_dim),
    # where q_len_per_request is the MTP query length (=1 without MTP)
    q = q.unsqueeze(1)

    out_ans = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale,
    )
    out_ans = out_ans.squeeze(1)
    torch.testing.assert_close(out_ans, out_ref, atol=1e-2, rtol=1e-2)
