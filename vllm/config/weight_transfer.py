# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training.

    Selects the transport backend. Backend-specific *wire* params (whether the
    transfer is packed, buffer sizes) that must agree between the trainer and
    inference sides are not set here — the trainer supplies them via its
    backend-specific `TrainerInitInfo` and propagates them to the worker at the
    init handshake, so they cannot drift.
    """

    backend: Literal["nccl", "ipc", "sparse_nccl"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """
