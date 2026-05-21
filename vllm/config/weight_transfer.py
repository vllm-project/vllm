# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """
