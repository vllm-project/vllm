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

    reload_mode: Literal["layerwise", "direct"] = "layerwise"
    """How the `nccl` and `ipc` backends write received weights into the model.
    `"layerwise"` loads each layer into a temporary copy, post-processes it and
    copies it back: correct for every model. `"direct"` has each `weight_loader`
    write into the live parameters with no post-processing: only for models
    whose post-load step leaves the parameters as the checkpoint has them, and
    nothing checks this. A failed direct update leaves the weights undefined
    and the engine must be restarted. `sparse_nccl` and `sharded_rdt` ignore
    this setting.
    """
