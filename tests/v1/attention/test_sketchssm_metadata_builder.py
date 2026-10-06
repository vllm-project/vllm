# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM metadata in BaseMambaAttentionMetadataBuilder: the sketch rows
next to the shared window rings, and the flush plan the layer completes."""

from types import SimpleNamespace

import numpy as np
import torch

from tests.v1.attention.test_replayssm_metadata_builder import _make_mamba_spec
from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.sketchssm.mamba2 import Mamba2SketchSSM

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _create_sketchssm_builder(buffer_len: int) -> MockMambaBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    # Set the flag after construction to skip the SketchSSM config validation
    # (it requires a real SupportsSketchSSM model) on the mock model.
    vllm_config.cache_config.sketchssm = "unused.pt"
    vllm_config.cache_config.replayssm_buffer_len = buffer_len
    vllm_config.cache_config.mamba_cache_mode = "none"
    return MockMambaBuilder(
        _make_mamba_spec(buffer_len, MambaBackendEnum.TRITON, 0),
        ["layer0"],
        vllm_config,
        DEVICE,
    )


def test_sketchssm_metadata():
    """--sketchssm alone builds the ring and sketch metadata."""
    builder = _create_sketchssm_builder(16)
    assert builder.use_window_rings and not builder.use_replayssm
    # A decode row, a prompt's one-token tail and two prompt chunks.
    batch = BatchSpec(seq_lens=[116, 41, 100, 60], query_lens=[1, 1, 50, 20])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, True, True, True]),
        req_idx=np.array([7, 2, 3, 5]),
    )
    meta = builder.build(0, common)

    assert meta.num_decodes == 2 and meta.num_prefills == 2
    assert meta.write_pos_d is not None and meta.is_flush_d is not None
    assert meta.sketch_indices_d is not None and meta.sketch_indices_p is not None
    assert meta.sketch_missing_d is not None
    assert meta.sketch_flush_rows_d is not None
    assert meta.write_pos_d.shape == (2,) and meta.is_flush_d.shape == (2,)
    assert meta.sketch_indices_d.tolist() == [7, 2]
    # The tail runs as a decode row before any sketch of its request exists.
    assert meta.sketch_missing_d.tolist() == [0, 1]
    assert meta.sketch_flush_rows_d.shape == (2,)
    assert meta.sketch_indices_p.tolist() == [3, 5]


def test_sketchssm_decode_rows_pad_past_requests():
    """Padded decode rows past the real requests read sketch row 0."""
    builder = _create_sketchssm_builder(16)
    batch = BatchSpec(seq_lens=[116, 117, 118], query_lens=[1, 1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, False, False]),
        req_idx=np.array([7, 3]),
    )
    meta = builder.build(0, common)
    assert meta.sketch_indices_d is not None and meta.sketch_missing_d is not None
    assert meta.sketch_indices_d.tolist() == [7, 3, 0]
    assert meta.sketch_missing_d.tolist() == [0, 0, 0]


def test_sketchssm_plan_flush():
    """Rows without a sketch flush too; the flush rows are listed in order."""
    ring_pos = torch.tensor([0, 5, 0, 1, 9], dtype=torch.int32)
    meta = SimpleNamespace(
        is_flush_d=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
        sketch_missing_d=torch.tensor([0, 0, 1, 0], dtype=torch.int8),
        sketch_flush_rows_d=torch.full((4,), -1, dtype=torch.int32),
    )
    state_indices = torch.tensor([[1], [2], [3], [4]], dtype=torch.int32)

    Mamba2SketchSSM.plan_flush(ring_pos, state_indices, meta)

    assert meta.is_flush_d.tolist() == [0, 1, 1, 0]
    assert meta.sketch_flush_rows_d.tolist() == [1, 2, -1, -1]
    # The forced flush leaves the ring of its slot empty.
    assert ring_pos.tolist() == [0, 5, 0, 0, 9]
