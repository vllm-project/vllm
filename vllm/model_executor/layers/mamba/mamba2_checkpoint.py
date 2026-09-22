# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata


class Mamba2PrefillCheckpointBuilder(MambaPrefillCheckpointBuilder):
    """Compact the checkpoint rows rather than masking them on device.

    The base builder keeps every prefill row and marks declined ones with
    ``NULL_BLOCK_ID``, which suits KDA's Triton exporter because it masks per
    program. Mamba2 writes with plain torch indexing, where selecting the
    valid rows on device is boolean indexing and costs a GPU<->CPU sync per.
    ``offsets`` is already on the host, so compacting there is free.
    """

    def build(
        self,
        m: CommonAttentionMetadata,
        request_rows: list[int],
    ) -> MambaPrefillCheckpointMetadata | None:
        meta = super().build(m, request_rows)
        if meta is None:
            return None
        keep = [i for i, offset in enumerate(meta.offsets) if offset]
        idx = async_tensor_h2d(keep, m.query_start_loc.device, torch.int64)
        return MambaPrefillCheckpointMetadata(
            meta.checkpoint_offsets[idx],
            meta.state_indices[idx],
            meta.offsets,
        )
