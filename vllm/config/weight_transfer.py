# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """

    frozen_weight_modules: list[str] = Field(default_factory=list)
    """Runtime module-name glob patterns excluded from weight reloads. Their
    parameters are immutable after initial loading and must be omitted by the
    sender. L2 sleep keeps a persistent CPU copy; CPU-resident weights are reused.
    """
