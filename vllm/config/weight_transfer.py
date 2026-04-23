# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "wpi"] = "nccl"
    """The backend to use for weight transfer.

    Available backends:
    - "nccl": Direct NCCL broadcast between trainer and workers
    - "ipc": CUDA IPC handles for same-node weight sharing
    - "wpi": Weight Propagation Interface — driver-managed NCCL broadcast
             with persistent VRAM buffers and zero-copy FD sharing.
             Requires `wpi_client` package.
    """
