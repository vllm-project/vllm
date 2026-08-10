# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config

WeightTransferMethods = Literal["nccl", "ipc", "sparse_nccl"]


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: WeightTransferMethods | str = "nccl"
    """The backend to use for weight transfer. Built-in options are ``"nccl"``,
    ``"ipc"``, and ``"sparse_nccl"``. Custom backends registered by external
    plugins are also accepted. Validated against the
    ``WeightTransferEngineFactory`` registry at engine creation time.
    """
