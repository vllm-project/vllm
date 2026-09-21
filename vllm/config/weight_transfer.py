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

    frozen_weight_names: list[str] = Field(default_factory=list)
    """Parameter-name globs to preserve across level-2 sleep.
    The sender must omit these parameters; loaders must preserve their storage.
    """
