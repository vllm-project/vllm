# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata coverage for fold-every-commit ReplaySSM speculative decode."""

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
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
SPEC_QUERY_LEN = 1 + NUM_SPEC
BLOCK_SPEC = 16
BUFFER_LEN = 8
NHEADS, HEAD_DIM, DSTATE, NGROUPS = 2, 4, 8, 1
CONV_DIM = NHEADS * HEAD_DIM + 2 * NGROUPS * DSTATE
CONV_STATE_LEN = 3 + NUM_SPEC
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReplaySSM spec metadata uses CUDA buffers"
)


def _make_spec_mamba_spec(*, dim_first: bool = False) -> MambaSpec:
    conv_shape = (CONV_DIM, CONV_STATE_LEN) if dim_first else (CONV_STATE_LEN, CONV_DIM)
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            conv_shape,
            (NHEADS, HEAD_DIM, DSTATE),
            (NHEADS, SPEC_QUERY_LEN, HEAD_DIM),
            (NHEADS, SPEC_QUERY_LEN),
            (NGROUPS, SPEC_QUERY_LEN, DSTATE),
        ),
        dtypes=(torch.float32,) * 5,
    )


def _create_spec_builder(
    *, buffer_len: int = BUFFER_LEN, full_cuda_graph: bool = False
) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        num_gpu_blocks=1024,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.use_replayssm_spec = True
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    return MockMambaBuilder(_make_spec_mamba_spec(), ["layer0"], vllm_config, DEVICE)


def _make_common(seq_lens: list[int], query_lens: list[int] | None = None):
    n = len(seq_lens)
    query_lens = query_lens if query_lens is not None else [SPEC_QUERY_LEN] * n
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    )
    return common.replace(is_prefilling=torch.tensor([False] * n, dtype=torch.bool))


def _build(builder: MockMambaBuilder, common):
    return builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(
            common.num_reqs, dtype=torch.int32, device=DEVICE
        ),
    )


def test_metadata_keeps_only_current_window_scratch():
    builder = _create_spec_builder()
    metadata = _build(builder, _make_common([120, 124]))

    assert metadata.spec_bc_pre_scratch is not None
    assert metadata.spec_bc_pre_scratch.shape == (
        2,
        NGROUPS,
        SPEC_QUERY_LEN,
        BLOCK_SPEC,
    )
    assert not hasattr(metadata, "spec_write_pos_d")
    assert not hasattr(builder, "spec_write_pos")


def test_replayssm_buffer_len_does_not_change_spec_scratch():
    short = _create_spec_builder(buffer_len=1)
    long = _create_spec_builder(buffer_len=128)

    assert short.decode_spec_bc_pre.shape[2:] == (SPEC_QUERY_LEN, BLOCK_SPEC)
    assert long.decode_spec_bc_pre.shape[2:] == (SPEC_QUERY_LEN, BLOCK_SPEC)


@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
def test_conv_state_layouts_are_supported(dim_first):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.use_replayssm_spec = True

    builder = MockMambaBuilder(
        _make_spec_mamba_spec(dim_first=dim_first),
        ["layer0"],
        vllm_config,
        DEVICE,
    )

    assert builder.decode_spec_bc_pre.shape[1] == NGROUPS


def test_cudagraph_scratch_uses_persistent_padded_buffer():
    builder = _create_spec_builder(full_cuda_graph=True)
    metadata = _build(builder, _make_common([120, 120]))

    assert metadata.spec_bc_pre_scratch is not None
    assert metadata.spec_bc_pre_scratch.shape[0] == metadata.num_reqs
    assert metadata.spec_bc_pre_scratch.data_ptr() == (
        builder.decode_spec_bc_pre.data_ptr()
    )


def test_single_prompt_tail_is_the_only_forced_commit():
    builder = _create_spec_builder()
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100], query_lens=[1]), BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor([True], dtype=torch.bool))

    metadata = _build(builder, common)

    assert metadata.num_decodes == 1
    assert metadata.spec_force_commit_d is not None
    assert metadata.spec_force_commit_d.tolist() == [True]

    regular = _build(builder, _make_common([120], query_lens=[SPEC_QUERY_LEN]))
    assert regular.spec_force_commit_d is not None
    assert regular.spec_force_commit_d.tolist() == [False]


def test_conv_state_no_longer_depends_on_previous_acceptance():
    builder = _create_spec_builder()
    common = _make_common([120, 124])

    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor([3, 2], dtype=torch.int32, device=DEVICE),
    )

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [1, 1]


def test_decode_window_larger_than_activation_capacity_is_rejected():
    builder = _create_spec_builder()
    builder.reorder_batch_threshold = SPEC_QUERY_LEN + 1
    common = _make_common([120], query_lens=[SPEC_QUERY_LEN + 1])

    with pytest.raises(ValueError, match="activation capacity"):
        _build(builder, common)


def test_spec_page_requires_all_activation_buffers():
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.use_replayssm_spec = True
    bad_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=_make_spec_mamba_spec().shapes[:2],
        dtypes=(torch.float32,) * 2,
    )

    with pytest.raises(ValueError, match="requires the 5-tensor Mamba2 page"):
        MockMambaBuilder(bad_spec, ["layer0"], vllm_config, DEVICE)
