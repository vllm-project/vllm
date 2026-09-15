# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Scheduler and Worker must agree on when a remote reservation becomes stale.
_RESERVATION_TTL_SECONDS = 300


def _positive_int(name: str, value: object) -> int:
    message = f"ECMooncakeConnector requires {name} to be a positive integer."
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError(message) from error
    else:
        raise ValueError(message)
    if result <= 0:
        raise ValueError(f"ECMooncakeConnector requires {name} > 0.")
    return result


def _positive_float(name: str, value: object) -> float:
    message = f"ECMooncakeConnector requires {name} > 0."
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(message)
    return result


@dataclass(frozen=True)
class MooncakeECConfig:
    """Validated settings shared by the Scheduler and Worker roles.

    Runtime control addresses are registered after worker sockets bind.
    """

    is_producer: bool
    is_consumer: bool
    protocol: str
    buffer_device: str
    control_timeout_ms: int
    push_wait_timeout_s: float
    pool_size: int
    registry_addr: str
    dp_rank: int

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> MooncakeECConfig:
        parallel_config = vllm_config.parallel_config
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        if ec_config.is_ec_producer:
            if parallel_config.tensor_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers require tensor_parallel_size=1."
                )
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers do not support pipeline parallelism."
                )
            if parallel_config.data_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers require data_parallel_size=1."
                )

        registered_buffer_size = _positive_int(
            "ec_buffer_size", ec_config.ec_buffer_size
        )
        get = ec_config.get_from_extra_config
        registry_addr = get("proxy_registry_addr", None)
        if not registry_addr:
            raise ValueError("ECMooncakeConnector requires proxy_registry_addr.")

        return cls(
            is_producer=ec_config.is_ec_producer,
            is_consumer=ec_config.is_ec_consumer,
            protocol=str(get("mooncake_protocol", "rdma")),
            buffer_device=str(ec_config.ec_buffer_device or "cuda").lower(),
            control_timeout_ms=max(
                1,
                math.ceil(
                    _positive_float("control_timeout_s", get("control_timeout_s", 30))
                    * 1000
                ),
            ),
            push_wait_timeout_s=_positive_float(
                "push_wait_timeout_s", get("push_wait_timeout_s", 60)
            ),
            pool_size=registered_buffer_size,
            registry_addr=str(registry_addr),
            dp_rank=parallel_config.data_parallel_index,
        )
