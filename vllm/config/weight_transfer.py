# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl"] = "nccl"
    """The backend to use for weight transfer."""
