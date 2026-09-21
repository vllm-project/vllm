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
    """Runtime module-name glob patterns retained across level-2 sleep. Their
    parameters must remain immutable and be omitted by the sender. This only
    controls sleep/wake, not weight loading. GPU parameters are backed up to CPU
    at sleep and restored at wake; CPU-resident parameters need no copy.
    """
