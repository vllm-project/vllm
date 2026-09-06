# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for gdn_scatter_block_checkpoints (the all-mode checkpoint scatter).

CPU-only, deterministic: inter_states[c] is filled with the value c, final_states[s]
with 100+s, so we can assert exactly which chunk/final state lands in which block.
Validates the FLA start-of-chunk index mapping (h[c]=state before chunk c) and the
final-block-from-final-state rule, for fresh / varlen / chunked-continuation cases.
"""

import torch

from vllm.model_executor.layers.mamba.gdn.all_mode_utils import (
    gdn_scatter_block_checkpoints,
)

D = 2  # tiny per-state dim
CHUNK = 4
BLK = 8  # chunk_stride = 2


def _state(num_blocks):
    return torch.full((num_blocks, D), -1.0)


def _inter(nt):
    return torch.stack([torch.full((D,), float(c)) for c in range(nt)])


def _final(n):
    return torch.stack([torch.full((D,), 100.0 + s) for s in range(n)])


def test_scatter_single_fresh():
    # 20-token seq, num_computed=0: blocks 0,1 full + block 2 partial(final).
    ssm = _state(20)
    inter = _inter(5)  # chunks 0..4
    final = _final(1)
    block_table = torch.tensor([[10, 11, 12, 0, 0]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([0]),
        block_idx_last_scheduled_token_p=torch.tensor([2]),
        num_computed_tokens_p=torch.tensor([0]),
        first_chunk_p=torch.tensor([0]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    assert (ssm[10] == 2).all(), ssm[10]  # end of block 0 == h[2]
    assert (ssm[11] == 4).all(), ssm[11]  # end of block 1 == h[4]
    assert (ssm[12] == 100).all(), ssm[12]  # final (partial) block == final_states[0]
    print("single_fresh OK")


def test_scatter_varlen_two_seqs():
    ssm = _state(40)
    inter = _inter(8)  # seq0 chunks 0-4, seq1 chunks 5-7
    final = _final(2)
    block_table = torch.tensor([[10, 11, 12, 0, 0], [20, 21, 0, 0, 0]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([0, 0]),
        block_idx_last_scheduled_token_p=torch.tensor([2, 1]),
        num_computed_tokens_p=torch.tensor([0, 0]),
        first_chunk_p=torch.tensor([0, 5]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    assert (ssm[10] == 2).all() and (ssm[11] == 4).all() and (ssm[12] == 100).all()
    assert (ssm[20] == 7).all(), ssm[20]  # seq1 block0 end == h[5+2]=h[7]
    assert (ssm[21] == 101).all(), ssm[21]  # seq1 final block == final_states[1]
    print("varlen_two_seqs OK")


def test_scatter_continuation():
    # num_computed=8 (block 0 already cached), schedule tokens 8..20 (3 chunks).
    ssm = _state(40)
    inter = _inter(3)  # scheduled chunks 0..2 (global)
    final = _final(1)
    block_table = torch.tensor([[30, 31, 32]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([1]),
        block_idx_last_scheduled_token_p=torch.tensor([2]),
        num_computed_tokens_p=torch.tensor([8]),
        first_chunk_p=torch.tensor([0]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    assert (ssm[30] == -1).all(), ssm[30]  # already-cached block untouched
    assert (ssm[31] == 2).all(), ssm[31]  # block 1 end == h[2]
    assert (ssm[32] == 100).all(), ssm[32]  # final block == final_states[0]
    print("continuation OK")


def test_scatter_multiblock_continuation():
    # num_computed=16 (blocks 0,1 already cached), schedule blocks 2,3 full +
    # block 4 partial(final). Exercises num_computed spanning >1 cached block
    # + >1 interior block.
    ssm = _state(60)
    inter = _inter(5)  # scheduled chunks 0..4 (global)
    final = _final(1)
    block_table = torch.tensor([[40, 41, 42, 43, 44]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([2]),
        block_idx_last_scheduled_token_p=torch.tensor([4]),
        num_computed_tokens_p=torch.tensor([16]),
        first_chunk_p=torch.tensor([0]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    assert (ssm[40] == -1).all() and (ssm[41] == -1).all()  # cached blocks untouched
    assert (ssm[42] == 2).all(), ssm[42]  # block 2 end: k=((3)*8-16)/4=2 -> h[2]
    assert (ssm[43] == 4).all(), ssm[43]  # block 3 end: k=((4)*8-16)/4=4 -> h[4]
    assert (ssm[44] == 100).all(), ssm[44]  # final (partial) block == final_states[0]
    print("multiblock_continuation OK")


def test_scatter_unaligned_num_computed_skips_interior():
    # A non-chunk-aligned num_computed_tokens must NOT write an
    # approximate ("nearest chunk") interior SSM checkpoint into the content-hash-
    # addressed prefix cache. Instead the interior scatter is skipped for that
    # sequence (no crash) so it never poisons APC. The final block is still
    # written from final_states (exact regardless of alignment). This must not
    # crash because the startup CUDA-graph dummy run feeds synthetic, unaligned
    # num_computed to throwaway state.
    ssm = _state(60)
    inter = _inter(2)  # chunks 0,1 only
    final = _final(1)
    block_table = torch.tensor([[50, 51]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([0]),
        block_idx_last_scheduled_token_p=torch.tensor([1]),
        num_computed_tokens_p=torch.tensor([2]),  # 2 % CHUNK(4) != 0 -> skip interior
        first_chunk_p=torch.tensor([0]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    assert (ssm[50] == -1).all(), ssm[50]  # interior block skipped (untouched)
    assert (ssm[51] == 100).all(), ssm[51]  # final block still written (exact)
    print("unaligned_num_computed_skips_interior OK")


def test_scatter_mixed_alignment_skips_only_unaligned():
    # In a mixed batch, only the unaligned sequence's interior
    # blocks are skipped; the aligned sequence is scattered normally, and both
    # sequences' final blocks are written.
    ssm = _state(60)
    inter = _inter(6)
    final = _final(2)
    block_table = torch.tensor([[10, 11, 12, 0], [20, 21, 22, 0]])
    gdn_scatter_block_checkpoints(
        ssm,
        inter,
        final,
        block_table,
        block_idx_first_scheduled_token_p=torch.tensor([0, 0]),
        block_idx_last_scheduled_token_p=torch.tensor([2, 2]),
        num_computed_tokens_p=torch.tensor([0, 5]),  # seq1: 5 % 4 != 0 -> skip
        first_chunk_p=torch.tensor([0, 3]),
        mamba_block_size=BLK,
        chunk_size=CHUNK,
    )
    # seq0 (aligned): interior blocks written.
    assert (ssm[10] != -1).all() and (ssm[11] != -1).all()
    assert (ssm[12] == 100).all()  # seq0 final
    # seq1 (unaligned): interior blocks skipped, final still written.
    assert (ssm[20] == -1).all() and (ssm[21] == -1).all()
    assert (ssm[22] == 101).all()  # seq1 final
    print("mixed_alignment_skips_only_unaligned OK")


if __name__ == "__main__":
    test_scatter_single_fresh()
    test_scatter_varlen_two_seqs()
    test_scatter_continuation()
    test_scatter_multiblock_continuation()
    test_scatter_unaligned_num_computed_skips_interior()
    test_scatter_mixed_alignment_skips_only_unaligned()
    print("SCATTER ALL PASS")
