# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config

# Canonical defaults for packed-tensor transfer.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training.

    Base class; concrete backends use the subclasses below, which also carry
    the static "must-agree" wire params (`packed`, buffer sizes). The same
    config object is constructed at both the trainer and inference sides, so
    those params cannot drift between the two ends of a transfer.
    """

    backend: Literal["nccl", "ipc", "sparse_nccl"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """


@config
class IPCWeightTransferConfig(WeightTransferConfig):
    """Weight transfer config for the CUDA IPC backend."""

    backend: str = "ipc"
    """The backend to use for weight transfer. Fixed to `"ipc"` for this
    config."""
    packed: bool = False
    """Whether to use packed tensor transfer for bounded-memory chunking."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer when packed=True."""
