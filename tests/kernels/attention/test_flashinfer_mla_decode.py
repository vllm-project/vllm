# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.platforms import current_platform

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="FlashInfer MLA Requires compute capability of 10 or above.",
        allow_module_level=True,
    )
else:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla


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
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
def test_flashinfer_mla_decode(dtype: torch.dtype, bs: int, block_size: int):
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Deepseek R1 config
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    MAX_SEQ_LEN = 1024

    seq_lens = [torch.randint(2, MAX_SEQ_LEN, (1,)).item() for _ in range(bs)]
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
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    kv_cache = torch.randn(block_tables.numel(), block_size, qk_head_dim).to(dtype)
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


def test_flashinfer_mla_decode_dcp_combine_matches_no_dcp():
    """Simulate DCP=N at kernel level and check it reproduces no-DCP.

    Mirrors what ``MLAAttention.forward`` does in production under
    DCP > 1:
    - Each rank computes per-shard attention output + LSE on its slice
      of the KV cache (via the trtllm-gen MLA decode kernel).
    - Allgathered LSEs are passed to ``correct_attn_out`` (the inner
      kernel of ``cp_lse_ag_out_rs``) along with ``is_lse_base_on_e``
      sourced from the impl's ``lse_base_on_e`` ClassVar -- ``False``
      for FlashInfer because its LSE is in log2.
    - Each rank's reweighted output is summed for the final result.

    This test simulates that DCP path on a single device (no NCCL, no
    multi-rank spawn) by manually splitting the per-batch block table
    across N "ranks" and stacking the per-rank LSEs before calling
    ``correct_attn_out``. The combined output is asserted close to the
    reference produced by a single full-KV kernel call.

    If the DCP combine math regresses (broken ``IS_BASE_E`` branch,
    wrong ``lse_base_on_e`` ClassVar, miswired call site, etc.), the
    simulated DCP output will diverge from the reference and this test
    fails.
    """
    from vllm.v1.attention.backends.mla.flashinfer_mla import FlashInferMLAImpl
    from vllm.v1.attention.ops.common import correct_attn_out

    # Anchor on the production ClassVar so the test exercises the same
    # kernel branch MLAAttention picks at runtime.
    is_base_e = FlashInferMLAImpl.lse_base_on_e

    torch.set_default_device("cuda")
    torch.manual_seed(7)

    bs = 2
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    block_size = 64
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
    dtype = torch.bfloat16

    # Simulate DCP=4. seq_len must split cleanly: each rank gets
    # pages_per_rank * block_size tokens.
    n_ranks = 4
    pages_per_rank = 4
    n_pages = n_ranks * pages_per_rank  # 16
    seq_len = n_pages * block_size  # 1024
    seq_lens_full = torch.full((bs,), seq_len, dtype=torch.int32)
    seq_lens_per_rank = torch.full((bs,), seq_len // n_ranks, dtype=torch.int32)

    # Unique block IDs per batch, random permutation so rank-i's pages
    # aren't trivially the first N entries of the global block table.
    all_block_ids = torch.randperm(bs * n_pages, dtype=torch.int32)
    block_tables_full = all_block_ids.view(bs, n_pages).contiguous()

    kv_cache = torch.randn(bs * n_pages, block_size, qk_head_dim, dtype=dtype)
    q = torch.randn(bs, num_heads, qk_head_dim, dtype=dtype)

    workspace_buffer = torch.zeros(
        FLASHINFER_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device="cuda",
    )

    def _decode(block_tables, seq_lens, max_seq_len):
        out, lse = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
            return_lse=True,
        )
        # out: (bs, 1, num_heads, kv_lora_rank), lse: (bs, num_heads)
        return out.squeeze(1), lse

    # Reference: full KV, no DCP.
    ref_out, _ = _decode(block_tables_full, seq_lens_full, seq_len)

    # Per-rank: each rank sees ``pages_per_rank`` consecutive pages of
    # the global block table (the simplest DCP layout -- interleave_size
    # equals block_size). Same Q on every rank.
    per_rank_outs = []
    per_rank_lses = []
    for rank in range(n_ranks):
        bt_r = block_tables_full[
            :, rank * pages_per_rank : (rank + 1) * pages_per_rank
        ].contiguous()
        out_r, lse_r = _decode(bt_r, seq_lens_per_rank, seq_len // n_ranks)
        per_rank_outs.append(out_r)
        per_rank_lses.append(lse_r)

    # Stack LSEs to (N, B, H) as ``correct_attn_out`` expects.
    all_lses = torch.stack(per_rank_lses, dim=0)

    # For each rank, apply the kernel's per-rank reweighting and
    # accumulate. Clone the per-rank output because ``correct_attn_out``
    # writes back in-place.
    combined = torch.zeros_like(ref_out)
    for rank in range(n_ranks):
        out_r = per_rank_outs[rank].clone()
        corrected_r, _ = correct_attn_out(
            out_r,
            all_lses,
            rank,
            ctx=None,
            is_lse_base_on_e=is_base_e,
        )
        combined = combined + corrected_r

    # Compare combined DCP=4 output to the no-DCP reference. The
    # tolerance is looser than test_flashinfer_mla_decode (1e-2) because
    # the DCP path does extra arithmetic (per-rank reweight + sum) which
    # accumulates more rounding error.
    torch.testing.assert_close(combined, ref_out, atol=2e-2, rtol=2e-2)
