# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM metadata from BaseMambaAttentionMetadataBuilder.

Triton ReplaySSM plans write_pos/is_flush on the GPU (see
tests/kernels/mamba/test_replayssm_plan.py); the builder only supplies the
persistent buffers that planning fills, plus align-mode sequence lengths.
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
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.mamba import MambaBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class ReplaySSMBuildCase:
    seq_lens: list[int]
    query_lens: list[int]
    is_prefilling: list[bool]


def _make_mamba_spec(
    buffer_len: int,
    mamba_backend: MambaBackendEnum,
    num_speculative_tokens: int = 0,
) -> MambaSpec:
    ring_buffer_len = buffer_len + (
        1 + num_speculative_tokens
        if mamba_backend == MambaBackendEnum.FLASHINFER
        else 0
    )
    shapes = (
        (1, 1),
        (1, 1, 1),
        (1, ring_buffer_len, 1),
        (1, ring_buffer_len),
        (1, ring_buffer_len, 1),
    )
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=(torch.float32,),
    )


def _create_replayssm_builder(
    buffer_len: int,
    mamba_cache_mode: str = "none",
    *,
    mamba_backend: MambaBackendEnum = MambaBackendEnum.TRITON,
    num_speculative_tokens: int = 0,
) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    # Set the flags after construction to skip validate_mamba_cached_kernel
    # (it requires a real SupportsReplaySSM model) on the mock model.
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.mamba_config.backend = mamba_backend
    if num_speculative_tokens > 0:
        vllm_config.scheduler_config.async_scheduling = False
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    return MockMambaBuilder(
        _make_mamba_spec(buffer_len, mamba_backend, num_speculative_tokens),
        ["layer0"],
        vllm_config,
        DEVICE,
    )


def _build(
    builder: MockMambaBuilder,
    case: ReplaySSMBuildCase,
    num_accepted_tokens: torch.Tensor | None = None,
):
    batch = BatchSpec(seq_lens=case.seq_lens, query_lens=case.query_lens)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor(case.is_prefilling, dtype=torch.bool),
    )
    return builder.build(0, common, num_accepted_tokens=num_accepted_tokens)


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_triton_replayssm_planning_buffers(mamba_cache_mode):
    builder = _create_replayssm_builder(16, mamba_cache_mode)
    case = ReplaySSMBuildCase(
        seq_lens=[106, 113, 140],
        query_lens=[1, 1, 34],
        is_prefilling=[False, False, True],
    )
    first = _build(builder, case)
    second = _build(builder, case)

    assert first.num_decodes == 2
    for meta in (first, second):
        assert meta.write_pos_d is not None and meta.is_flush_d is not None
        assert meta.write_pos_d.shape == (2,) and meta.is_flush_d.shape == (2,)
        assert meta.bc_pre_scratch is not None
    # Persistent buffers: the planning kernel writes them in place.
    assert second.write_pos_d.data_ptr() == first.write_pos_d.data_ptr()
    assert second.is_flush_d.data_ptr() == first.is_flush_d.data_ptr()

    if mamba_cache_mode == "align":
        assert first.replayssm_align_block_size == BLOCK_SIZE
        assert first.replayssm_seq_lens_d is not None
        assert first.replayssm_seq_lens_d.tolist() == case.seq_lens
    else:
        assert first.replayssm_align_block_size == 0
        assert first.replayssm_seq_lens_d is None


def test_spec_decode_single_token_chunk_synthesizes_acceptance_metadata():
    builder = _create_replayssm_builder(16, num_speculative_tokens=3)
    # Single-token prefill chunk with prior state, run as a decode row.
    case = ReplaySSMBuildCase(seq_lens=[100], query_lens=[1], is_prefilling=[True])

    meta = _build(builder, case)

    assert meta.query_start_loc_d is not None
    assert meta.query_start_loc_d.tolist() == [0, 1]
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.tolist() == [1]


def test_flashinfer_replayssm_state_indices_are_stable_for_full_cudagraph():
    checkpointing_ssu = pytest.importorskip("flashinfer.mamba.checkpointing_ssu")
    if not hasattr(checkpointing_ssu, "allocate_checkpointing_ssu_scratch"):
        pytest.skip("requires FlashInfer ReplaySSM autotuning support")

    builder = _create_replayssm_builder(
        16,
        mamba_backend=MambaBackendEnum.FLASHINFER,
        num_speculative_tokens=3,
    )
    builder.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

    first = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[106, 106],
            query_lens=[1, 1],
            is_prefilling=[False, False],
        ),
    )
    first_indices = first.replayssm_state_indices_d
    assert first_indices is not None
    assert first_indices.is_contiguous()
    first_ptr = first_indices.data_ptr()

    second = _build(
        builder,
        ReplaySSMBuildCase(
            seq_lens=[122, 122],
            query_lens=[1, 1],
            is_prefilling=[False, False],
        ),
    )
    second_indices = second.replayssm_state_indices_d
    assert second_indices is not None
    assert second_indices.data_ptr() == first_ptr
    assert torch.equal(second_indices, second.state_indices_tensor_d[:, 0])
