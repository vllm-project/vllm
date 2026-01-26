# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Weight transfer engines for syncing model weights from trainers
to inference workers.
"""

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

WEIGHT_TRANSFER_ENGINE_REGISTRY = {
    "nccl": NCCLWeightTransferEngine,
}


def register_weight_transfer_engine(
    name: str, engine: type[WeightTransferEngine]
) -> None:
    if name in WEIGHT_TRANSFER_ENGINE_REGISTRY:
        raise ValueError(f"Weight transfer engine {name} already registered")
    WEIGHT_TRANSFER_ENGINE_REGISTRY[name] = engine


def init_transfer_engine(config: WeightTransferConfig, parallel_config: ParallelConfig):
    if config.backend not in WEIGHT_TRANSFER_ENGINE_REGISTRY:
        raise ValueError(f"Invalid weight transfer backend: {config.backend}")

    engine_cls = WEIGHT_TRANSFER_ENGINE_REGISTRY[config.backend]
    return engine_cls(config, parallel_config)


__all__ = [
    "WeightTransferEngine",
    "NCCLWeightTransferEngine",
    "register_weight_transfer_engine",
    "WEIGHT_TRANSFER_ENGINE_REGISTRY",
    "WeightUpdateRequest",
]
