# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 ReplaySSM decode write-position derivation in
BaseMambaAttentionMetadataBuilder: write_pos and is_flush computed from the
per-request ring origin (replayssm_decode_base) and num_computed.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class ReplaySSMBuildCase:
    """A decode batch and its expected per-row write_pos / is_flush.

    num_computed = seq_len - query_len; write_pos =
    (num_computed - decode_base) % buffer_len; is_flush = write_pos ==
    buffer_len - 1 (or a forced one-token flush when num_computed < decode_base).
    """

    seq_lens: list[int]
    query_lens: list[int]
    is_prefilling: list[bool]
    num_prompt_tokens: list[int]
    decode_base: list[int]
    buffer_len: int
    expected_write_pos: list[int]
    expected_is_flush: list[int]
    mamba_cache_mode: str = "none"


REPLAYSSM_BUILD_CASES = {
    # decode_base == num_prompt (fresh request).
    "fresh_decode": ReplaySSMBuildCase(
        seq_lens=[106],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[5],
        expected_is_flush=[0],
    ),
    # decode_base > num_prompt anchors write_pos at the resume point.
    "resumed_reanchors_to_zero": ReplaySSMBuildCase(
        seq_lens=[106],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[105],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[0],
    ),
    # write_pos == buffer_len - 1 flushes.
    "flush_boundary": ReplaySSMBuildCase(
        seq_lens=[116],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
    ),
    # Resumed request landing on a flush boundary.
    "resumed_flush_boundary": ReplaySSMBuildCase(
        seq_lens=[121],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[105],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
    ),
    # Per-row write_pos / is_flush are independent.
    "mixed_rows": ReplaySSMBuildCase(
        seq_lens=[104, 106, 216],
        query_lens=[1, 1, 1],
        is_prefilling=[False, False, False],
        num_prompt_tokens=[100, 100, 200],
        decode_base=[100, 105, 200],
        buffer_len=16,
        expected_write_pos=[3, 0, 15],
        expected_is_flush=[0, 0, 1],
    ),
    # write_pos wraps within the buffer (6 % 4 == 2).
    "small_buffer_wrap": ReplaySSMBuildCase(
        seq_lens=[112],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[105],
        buffer_len=4,
        expected_write_pos=[2],
        expected_is_flush=[0],
    ),
    # Single-token prefill-as-decode still in the prompt (num_computed <
    # decode_base): forced one-token flush.
    "leftover_prompt_one_token_flush": ReplaySSMBuildCase(
        seq_lens=[100],
        query_lens=[1],
        is_prefilling=[True],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[1],
    ),
    # Align mode (block_size 16). Past the first boundary the ring re-anchors at
    # the block start: num_computed 117 -> block_start 112, write_pos 5 (vs 1 in
    # none mode).
    "align_reanchor_past_boundary": ReplaySSMBuildCase(
        seq_lens=[118],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[5],
        expected_is_flush=[0],
        mamba_cache_mode="align",
    ),
    # First-block boundary (num_computed+1 == 112) forces a flush even though
    # write_pos (11) != buffer_len - 1.
    "align_first_block_boundary_flush": ReplaySSMBuildCase(
        seq_lens=[112],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[11],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # First step of a new block re-anchors write_pos to 0.
    "align_new_block_start_zero": ReplaySSMBuildCase(
        seq_lens=[113],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[0],
        expected_is_flush=[0],
        mamba_cache_mode="align",
    ),
    # block_size % buffer_len == 0: a later boundary lands on write_pos ==
    # buffer_len - 1, so the boundary flush coincides with the natural flush.
    "align_boundary_coincides_natural_flush": ReplaySSMBuildCase(
        seq_lens=[128],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=16,
        expected_write_pos=[15],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # block_size % buffer_len != 0 (buffer_len 6): the boundary step still flushes
    # although write_pos (3) != buffer_len - 1.
    "align_unaligned_buffer_forces_flush": ReplaySSMBuildCase(
        seq_lens=[128],
        query_lens=[1],
        is_prefilling=[False],
        num_prompt_tokens=[100],
        decode_base=[100],
        buffer_len=6,
        expected_write_pos=[3],
        expected_is_flush=[1],
        mamba_cache_mode="align",
    ),
    # Per-row independence in align mode: partial-block / new-block / boundary.
    "align_mixed_rows": ReplaySSMBuildCase(
        seq_lens=[105, 113, 112],
        query_lens=[1, 1, 1],
        is_prefilling=[False, False, False],
        num_prompt_tokens=[100, 100, 100],
        decode_base=[100, 100, 100],
        buffer_len=16,
        expected_write_pos=[4, 0, 11],
        expected_is_flush=[0, 0, 1],
        mamba_cache_mode="align",
    ),
}


def _make_mamba_spec(buffer_len: int) -> MambaSpec:
    # Five-tensor ReplaySSM page; the builder only reads shapes[4][0] (bc groups).
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            (1, 1),
            (1, 1, 1),
            (1, buffer_len, 1),
            (1, buffer_len),
            (1, buffer_len, 1),
        ),
        dtypes=(torch.float32,),
    )


def _create_replayssm_builder(
    buffer_len: int, mamba_cache_mode: str = "none"
) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    # Set the flags after construction to skip validate_mamba_cached_kernel
    # (it requires a Triton backend) on the mock model.
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    return MockMambaBuilder(
        _make_mamba_spec(buffer_len), ["layer0"], vllm_config, DEVICE
    )


def _build(builder: MockMambaBuilder, case: ReplaySSMBuildCase):
    batch = BatchSpec(seq_lens=case.seq_lens, query_lens=case.query_lens)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor(case.is_prefilling, dtype=torch.bool),
        num_prompt_tokens_cpu=torch.tensor(case.num_prompt_tokens, dtype=torch.int32),
        replayssm_decode_base_cpu=torch.tensor(case.decode_base, dtype=torch.int32),
    )
    return builder.build(0, common)


@pytest.mark.parametrize(
    "case", REPLAYSSM_BUILD_CASES.values(), ids=REPLAYSSM_BUILD_CASES.keys()
)
def test_replayssm_write_pos(case: ReplaySSMBuildCase):
    builder = _create_replayssm_builder(case.buffer_len, case.mamba_cache_mode)
    meta = _build(builder, case)

    assert meta.write_pos_d is not None
    assert meta.is_flush_d is not None
    n = len(case.expected_write_pos)
    assert meta.write_pos_d[:n].tolist() == case.expected_write_pos
    assert meta.is_flush_d[:n].tolist() == case.expected_is_flush


def test_resumed_request_differs_from_fresh():
    """Same token count, different decode_base: fresh (base 100) -> write_pos 5,
    resumed (base 105) -> write_pos 0."""
    builder = _create_replayssm_builder(16)
    batch = BatchSpec(seq_lens=[106, 106], query_lens=[1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, False]),
        num_prompt_tokens_cpu=torch.tensor([100, 100], dtype=torch.int32),
        replayssm_decode_base_cpu=torch.tensor([100, 105], dtype=torch.int32),
    )
    meta = builder.build(0, common)

    assert meta.write_pos_d.tolist()[:2] == [5, 0]
    assert meta.is_flush_d.tolist()[:2] == [0, 0]
