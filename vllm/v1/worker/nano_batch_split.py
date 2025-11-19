# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from vllm.compilation.nanoflow import manager as nano_manager
from vllm.compilation.nanoflow.split_utils import NanoOpInfo
from vllm.forward_context import get_forward_context
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices


def nano_ubatch_split(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
) -> tuple[Optional[UBatchSlices], Optional[torch.Tensor]]:
    """
    Prepare two UBatch-compatible nano-batch slices.

    - Uses nano_manager.prepare_nano_split to decide if splitting is beneficial
      (i.e., num_nano_batches > 1).
    - Computes a single token split point using custom logic to remain
      compatible with UBatch execution.
    """
    assert num_tokens_unpadded == num_tokens_padded
    batch_size = int(len(num_scheduled_tokens_per_request))
    total_tokens = int(np.sum(num_scheduled_tokens_per_request))
    if batch_size <= 1 or total_tokens <= 1:
        return (None, None)

    tokens_list = num_scheduled_tokens_per_request.tolist()
    split_config = nano_manager.prepare_nano_split(batch_size, tokens_list)
    if getattr(split_config, "num_nano_batches", 1) <= 1:
        return (None, None)
    assert split_config.num_nano_batches == 2

    first_slice = UBatchSlice(
        slice(0, split_config.batch_indices[1]), slice(0, split_config.split_indices[1])
    )
    second_slice = UBatchSlice(
        slice(split_config.batch_indices[1], batch_size),
        slice(split_config.split_indices[1], split_config.split_indices[2]),
    )

    @contextmanager
    def op_hook(op_info: NanoOpInfo):
        ctx = get_forward_context()
        attn_metadata_list = ctx.attn_metadata
        assert isinstance(attn_metadata_list, list)
        ctx.attn_metadata = attn_metadata_list[op_info.idx]
        try:
            yield
        finally:
            ctx.attn_metadata = attn_metadata_list
            pass

    nano_manager.set_op_hook(op_hook)

    return (
        [first_slice, second_slice],
        torch.tensor([num_tokens_padded], device="cpu", dtype=torch.int32),
    )
