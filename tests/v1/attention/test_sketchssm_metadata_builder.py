# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM metadata in BaseMambaAttentionMetadataBuilder: sketch rows,
flush rows and prompt-completion flags next to the shared window rings."""

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
from vllm.v1.attention.backends.mamba_attn import _sketch_decode_rows

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
    # A flushing decode row, a completing prompt and a chunked prompt.
    batch = BatchSpec(seq_lens=[116, 100, 60], query_lens=[1, 50, 20])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        is_prefilling=torch.tensor([False, True, True]),
        replayssm_decode_base_cpu=torch.tensor([100, 100, 120], dtype=torch.int32),
        req_idx=np.array([7, 3, 5]),
    )
    meta = builder.build(0, common)

    assert meta.num_decodes == 1 and meta.num_prefills == 2
    assert meta.write_pos_d.tolist()[:1] == [15]
    assert meta.is_flush_d.tolist()[:1] == [1]
    assert meta.sketch_indices_d.tolist()[:1] == [7]
    assert meta.sketch_flush_rows_d.tolist()[:1] == [0]
    assert meta.sketch_indices_p.tolist() == [3, 5]
    assert meta.sketch_build_p.tolist() == [1, 0]


def test_sketchssm_decode_rows_pad_past_requests():
    """Padded decode rows past the real requests read sketch row 0."""
    batch = BatchSpec(seq_lens=[116, 117, 118], query_lens=[1, 1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE).replace(
        req_idx=np.array([7, 3])
    )
    rows, flush_rows = _sketch_decode_rows(common, 3, torch.tensor([0, 1, 0]))
    assert rows.tolist() == [7, 3, 0]
    assert flush_rows.tolist() == [1, -1, -1]
