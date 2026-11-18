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
    """Whether weight updates reload in place: with runtime weight updates
    enabled, unless `VLLM_RELOAD_MODE=layerwise` or the backend is sharded_rdt."""
    import vllm.envs as envs

    transfer = weight_transfer_config
    return (
        transfer is not None
        and transfer.backend != "sharded_rdt"
        and envs.VLLM_RELOAD_MODE == "direct"
    )
