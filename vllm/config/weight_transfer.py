# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config

# Canonical defaults for packed-tensor transfer. These live here (the lowest
# config layer) so both the trainer and inference sides read the same values
# from `WeightTransferConfig`; `packed_tensor` re-imports them for its function
# signatures. Keeping them on the config avoids the old duplication between the
# trainer args and the per-round worker update-info.
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
class NCCLWeightTransferConfig(WeightTransferConfig):
    """Weight transfer config for the dense NCCL backend."""

    backend: str = "nccl"
    """The backend to use for weight transfer. Fixed to `"nccl"` for this
    config."""
    packed: bool = True
    """Whether to use packed tensor broadcasting for efficiency. When True,
    multiple tensors are batched together before broadcasting to reduce NCCL
    communication overhead."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer. Both producer (trainer) and
    consumer (worker) must use the same value."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Both producer and consumer must use the same value."""


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
