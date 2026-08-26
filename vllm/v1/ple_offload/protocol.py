# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC message definitions for PLE CPU offload."""

from dataclasses import dataclass

import msgspec
import torch

# ---------------------------------------------------------------------------
# IPC message dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PleOffloadRegistration:
    """Sent once from each GPU worker during offload setup."""

    worker_id: int
    tp_rank: int
    dp_rank: int
    # CUDA tensors are serialized through PyTorch CUDA IPC.
    gpu_output_buffers: dict[str, torch.Tensor]
    sem_flag_tensors: dict[str, torch.Tensor]
    # CPU tensors are allocated in shared memory and registered once.
    input_ids_buf: torch.Tensor
    query_start_loc_buf: torch.Tensor
    ngram_context_buf: torch.Tensor | None


@dataclass
class PleOffloadRequest:
    """Sent by each DP rank's TP rank zero at every inference step."""

    dp_rank: int
    num_tokens: int
    num_reqs: int


_PLE_OFFLOAD_REQUEST_DECODER = msgspec.msgpack.Decoder(PleOffloadRequest)
