# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "sparse_nccl"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """

    weight_format: Literal["checkpoint", "runtime"] = "checkpoint"
    """Format sent by dense weight-transfer backends.

    ``checkpoint`` restores the model's checkpoint schema and runs post-load
    processing at commit. ``runtime`` requires tensors already converted to
    the receiving rank's serving/kernel layout and commits them without any
    post-load processing.
    """
