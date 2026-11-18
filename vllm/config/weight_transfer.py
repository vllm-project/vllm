# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt", "nccl_m2n"] | str = (
        "nccl"
    )
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """


def uses_direct_reload(weight_transfer_config: WeightTransferConfig | None) -> bool:
    """Whether weight updates reload in place: `VLLM_RELOAD_MODE=direct`, or
    the default `auto` with runtime weight updates enabled. sharded_rdt always
    reloads layerwise."""
    import vllm.envs as envs

    transfer = weight_transfer_config
    if transfer is not None and transfer.backend == "sharded_rdt":
        return False
    mode = envs.VLLM_RELOAD_MODE
    return mode == "direct" or (mode == "auto" and transfer is not None)
