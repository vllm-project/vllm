# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """

    reload_mode: Literal["layerwise", "trace"] = "layerwise"
    """Checkpoint reload implementation. Trace is opt-in for NCCL and IPC,
    and requires a supported policy for every transformed layer."""

    preserve_checkpoint: bool = False
    """Keep trace checkpoint staging until the next round. By default each
    layer releases staging immediately after conversion."""

    def __post_init__(self) -> None:
        if self.reload_mode == "trace" and self.backend not in ("nccl", "ipc"):
            raise ValueError("Trace reload supports only NCCL and IPC")
        if self.preserve_checkpoint and self.reload_mode != "trace":
            raise ValueError("preserve_checkpoint requires reload_mode='trace'")
