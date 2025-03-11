# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from vllm.attention.ops.nki_blocksparse_flash_attn import (
    BlockSparsePlan, FlashAttentionPlanner)


def ceil_div(a, b):
    return (a + b - 1) // b


def validate_plan(plan, truth):
    assert np.all(plan.tile_q_indices == truth.tile_q_indices)
    assert np.all(
        plan.tile_block_table_offsets == truth.tile_block_table_offsets)
    assert np.all(plan.tile_q_seq_ids == truth.tile_q_seq_ids)
    assert np.all(plan.tile_kv_seq_ids == truth.tile_kv_seq_ids)


class TestFlashAttentionPlanner:

    def _compute_truth(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        prompt_lens,
        context_lens,
    ):
        prompt_ends = np.cumsum(prompt_lens)
        prompt_starts = np.concatenate([[0], prompt_ends[:-1]])

        padded_context_lens = ceil_div(context_lens, block_size) * block_size
        context_starts = np.concatenate([[0],
                                         np.cumsum(padded_context_lens)[:-1]])
        context_ends = context_starts + context_lens

        total_prompt = prompt_ends[-1]
        total_context = context_ends[-1]
        num_q_tile = ceil_div(total_prompt, tile_size_q)
        num_kv_tile = ceil_div(total_context, tile_size_kv)

        def _is_tile_needed(q_tile_idx, kv_tile_idx):
            q_start = q_tile_idx * tile_size_q
            q_end = q_start + tile_size_q
            kv_start = kv_tile_idx * tile_size_kv
            kv_end = kv_start + tile_size_kv
            for seq_id in range(num_seqs):
                seq_q_start = prompt_starts[seq_id]
                seq_q_end = prompt_ends[seq_id]
                seq_kv_start = context_starts[seq_id]
                seq_kv_end = context_ends[seq_id]
                # check if seq has overlap with tile
                if q_start >= seq_q_end or q_end <= seq_q_start:
                    continue
                if kv_start >= seq_kv_end or kv_end <= seq_kv_start:
                    continue
                # has overlap, this tile is needed
                return True
            return False

        # prepare seq ids
        def _gen_seq_ids(size, default_value, seq_starts, seq_ends):
            seq_ids = np.full((size, ), default_value, dtype=np.int32)
            for seq_id in range(num_seqs):
                seq_ids[seq_starts[seq_id]:seq_ends[seq_id]] = seq_id
            return seq_ids

        q_seq_ids = _gen_seq_ids(num_q_tile * tile_size_q, num_seqs,
                                 prompt_starts, prompt_ends)
        kv_seq_ids = _gen_seq_ids(
            num_kv_tile * tile_size_kv,
            num_seqs + 1,
            context_starts,
            context_ends,
        )

        tile_q_indices = []
        tile_block_table_offsets = []
        tile_q_seq_ids = []
        tile_kv_seq_ids = []
        for q_tile_idx in range(num_q_tile):
            for kv_tile_idx in range(num_kv_tile):
                if _is_tile_needed(q_tile_idx, kv_tile_idx):
                    tile_q_indices.append(q_tile_idx)
                    tile_block_table_offsets.append(kv_tile_idx *
                                                    tile_size_kv // block_size)
                    tile_q_seq_ids.append(
                        q_seq_ids[q_tile_idx * tile_size_q:(q_tile_idx + 1) *
                                  tile_size_q])
                    tile_kv_seq_ids.append(
                        kv_seq_ids[kv_tile_idx *
                                   tile_size_kv:(kv_tile_idx + 1) *
                                   tile_size_kv])
        return BlockSparsePlan(
            tile_q_indices=np.array(tile_q_indices, dtype=np.int32),
            tile_block_table_offsets=np.array(tile_block_table_offsets,
                                              dtype=np.int32),
            tile_q_seq_ids=np.array(tile_q_seq_ids, dtype=np.int32),
            tile_kv_seq_ids=np.array(tile_kv_seq_ids, dtype=np.int32),
            block_size=block_size,
        )

    def _run_test(self, prompt_lens, context_lens, block_size, tile_size_q,
                  tile_size_kv):
        assert len(prompt_lens) == len(context_lens)
        num_seqs = len(prompt_lens)
        blocksparse_planner = FlashAttentionPlanner(
            prompt_lens=prompt_lens,
            context_lens=context_lens,
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            block_size=block_size,
        )
        plan = blocksparse_planner.plan()
        truth = self._compute_truth(
            num_seqs=num_seqs,
            block_size=block_size,
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            prompt_lens=prompt_lens,
            context_lens=context_lens,
        )
        validate_plan(plan, truth)

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    def test_random_size(self, num_seqs, block_size, tile_size_q,
                         tile_size_kv):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = 4096
        prompt_lens = np.random.randint(1,
                                        MAX_QUERY_LEN_PER_SEQ,
                                        size=(num_seqs, ))
        context_lens = np.random.randint(1,
                                         MAX_KV_LEN_PER_SEQ,
                                         size=(num_seqs, ))
        self._run_test(prompt_lens, context_lens, block_size, tile_size_q,
                       tile_size_kv)

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_aligned_size_plus_offset(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        offset,
    ):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 2 * tile_size_q
        MAX_KV_LEN_PER_SEQ = 2 * tile_size_kv
        prompt_lens = np.random.randint(1,
                                        MAX_QUERY_LEN_PER_SEQ,
                                        size=(num_seqs, ))
        prompt_lens = ceil_div(prompt_lens, tile_size_q) * tile_size_q
        context_lens = np.random.randint(1,
                                         MAX_KV_LEN_PER_SEQ,
                                         size=(num_seqs, ))
        context_lens = ceil_div(context_lens, block_size) * block_size
        # shift the first element by offset so that all elements are aligned to
        # tile boundary + offset
        prompt_lens[0] += offset
        context_lens[0] += offset
        self._run_test(prompt_lens, context_lens, block_size, tile_size_q,
                       tile_size_kv)

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    def test_size1_q_and_empty_kv(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
    ):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = 4096
        prompt_lens = np.random.randint(1,
                                        MAX_QUERY_LEN_PER_SEQ,
                                        size=(num_seqs, ))
        context_lens = np.random.randint(1,
                                         MAX_KV_LEN_PER_SEQ,
                                         size=(num_seqs, ))
        # randomly pick 50% of sequences to have zero context tokens
        zero_len_seq = np.random.choice(num_seqs, size=(num_seqs // 2, ))
        context_lens[zero_len_seq] = 0
        self._run_test(prompt_lens, context_lens, block_size, tile_size_q,
                       tile_size_kv)

    # test the case with a large amount of padding and empty tiles
    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 512, 8, 16],
            [16, 512, 8, 16],
            [32, 1024, 16, 16],
        ],
    )
    def test_small_tile_large_padding(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
    ):
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = block_size // 2
        prompt_lens = np.random.randint(1,
                                        MAX_QUERY_LEN_PER_SEQ,
                                        size=(num_seqs, ))
        context_lens = np.random.randint(1,
                                         MAX_KV_LEN_PER_SEQ,
                                         size=(num_seqs, ))
        self._run_test(prompt_lens, context_lens, block_size, tile_size_q,
                       tile_size_kv)
