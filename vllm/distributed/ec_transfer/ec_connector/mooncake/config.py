# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse and validate configuration shared by Mooncake connector roles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _integer(name: str, value: object) -> int:
    message = f"ECMooncakeConnector requires {name} to be an integer."
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(message) from error
    raise ValueError(message)


def _positive_integer(name: str, value: object) -> int:
    parsed = _integer(name, value)
    if parsed <= 0:
        raise ValueError(f"ECMooncakeConnector requires {name} > 0.")
    return parsed


def _finite_float(name: str, value: object, allow_zero: bool) -> float:
    requirement = ">= 0" if allow_zero else "> 0"
    message = f"ECMooncakeConnector requires {name} {requirement}."
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(message) from error
    if not math.isfinite(parsed) or parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(message)
    return parsed


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ECMooncakeConnector requires non-empty {name}.")
    return value.strip()


@dataclass(frozen=True)
class MooncakeECConfig:
    """Validated runtime settings for one Scheduler or Worker instance.

    Attributes:
        is_producer: Whether this instance can originate encoder-cache pushes.
        is_consumer: Whether this instance can receive encoder-cache pushes.
        protocol: Mooncake transport protocol passed to ``TransferEngine``.
        buffer_device: Device used for registered staging and receive buffers.
        reservation_port: Rank-adjusted Worker control-plane base port.
        reservation_addr: Scheduler-visible Consumer control address.
        control_timeout_s: Timeout for one ZMQ request/response exchange.
        push_wait_timeout_s: Maximum Scheduler wait for a ready notification.
        transfer_workers: Maximum concurrent data-plane transfer batches.
        control_workers: Maximum concurrent control-plane operations.
        producer_pool_size: Bytes reserved for the Producer staging pool.
        consumer_pool_size: Bytes reserved for the Consumer receive pool.
        transfer_metrics_log_interval: Producer transfer log interval in seconds.
        consumer_metrics_log_interval: Consumer metrics log interval in seconds.
    """

    is_producer: bool
    is_consumer: bool
    protocol: str
    buffer_device: str
    reservation_port: int | None
    reservation_addr: str | None
    control_timeout_s: float
    push_wait_timeout_s: float
    transfer_workers: int
    control_workers: int
    producer_pool_size: int
    consumer_pool_size: int
    transfer_metrics_log_interval: float
    consumer_metrics_log_interval: float

    @property
    def control_timeout_ms(self) -> int:
        return max(1, math.ceil(self.control_timeout_s * 1000))

    @classmethod
    def from_vllm_config(
        cls, vllm_config: VllmConfig, role: ECConnectorRole
    ) -> MooncakeECConfig:
        """Build role-specific settings from the top-level vLLM config.

        Args:
            vllm_config: Source vLLM configuration.
            role: Connector process role being configured.

        Returns:
            Validated, normalized Mooncake connector settings.

        Raises:
            ValueError: If an option is invalid or the requested parallel
                topology is unsupported for a Producer.
        """
        parallel_config = vllm_config.parallel_config
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        is_producer = ec_config.is_ec_producer
        if is_producer:
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

        registered_buffer_size = _positive_integer(
            "ec_buffer_size", ec_config.ec_buffer_size
        )

        extra = ec_config.ec_connector_extra_config
        raw_port = extra.get("reservation_zmq_port")
        reservation_port = (
            _integer("reservation_zmq_port", raw_port) if raw_port is not None else None
        )
        if reservation_port is not None and not 1 <= reservation_port <= 65535:
            raise ValueError(
                "ECMooncakeConnector requires reservation_zmq_port in 1..65535."
            )

        if reservation_port is not None:
            reservation_port += (
                parallel_config.data_parallel_index
                * parallel_config.tensor_parallel_size
            )
            highest_port = reservation_port + parallel_config.tensor_parallel_size - 1
            if not 1 <= reservation_port <= highest_port <= 65535:
                raise ValueError(
                    "ECMooncakeConnector reservation ports must be in 1..65535."
                )

        reservation_addr = (
            _nonempty_string("reservation_zmq_addr", extra["reservation_zmq_addr"])
            if "reservation_zmq_addr" in extra
            else None
        )
        if reservation_addr is None and reservation_port is not None:
            reservation_addr = f"tcp://127.0.0.1:{reservation_port}"

        is_consumer = ec_config.is_ec_consumer
        if is_consumer and role == ECConnectorRole.SCHEDULER and not reservation_addr:
            raise ValueError(
                "ec_consumer with ECMooncakeConnector requires "
                "reservation_zmq_port or reservation_zmq_addr."
            )
        if is_consumer and role == ECConnectorRole.WORKER and reservation_port is None:
            raise ValueError(
                "ec_consumer with ECMooncakeConnector workers require "
                "reservation_zmq_port."
            )

        control_timeout_s = _finite_float(
            "control_timeout_s", extra.get("control_timeout_s", 30), False
        )
        if control_timeout_s > (2**31 - 1) / 1000:
            raise ValueError("ECMooncakeConnector control_timeout_s is too large.")
        push_wait_timeout_s = _finite_float(
            "push_wait_timeout_s", extra.get("push_wait_timeout_s", 60), False
        )
        transfer_workers = _positive_integer(
            "transfer_max_workers", extra.get("transfer_max_workers", 4)
        )
        control_workers = _positive_integer(
            "control_max_workers", extra.get("control_max_workers", 8)
        )
        producer_pool_size = _positive_integer(
            "producer_buffer_pool_size",
            extra.get("producer_buffer_pool_size", registered_buffer_size),
        )
        consumer_pool_size = _positive_integer(
            "consumer_buffer_pool_size",
            extra.get("consumer_buffer_pool_size", registered_buffer_size),
        )

        protocol = _nonempty_string(
            "mooncake_protocol", extra.get("mooncake_protocol", "rdma")
        )
        raw_buffer_device = ec_config.ec_buffer_device
        if raw_buffer_device is not None and not isinstance(raw_buffer_device, str):
            raise ValueError("ECMooncakeConnector ec_buffer_device must be a string.")

        return cls(
            is_producer=is_producer,
            is_consumer=is_consumer,
            protocol=protocol,
            buffer_device=(raw_buffer_device or "cuda").strip() or "cuda",
            reservation_port=reservation_port,
            reservation_addr=reservation_addr,
            control_timeout_s=control_timeout_s,
            push_wait_timeout_s=push_wait_timeout_s,
            transfer_workers=transfer_workers,
            control_workers=control_workers,
            producer_pool_size=producer_pool_size,
            consumer_pool_size=consumer_pool_size,
            transfer_metrics_log_interval=_finite_float(
                "transfer_metrics_log_interval",
                extra.get("transfer_metrics_log_interval", 10),
                True,
            ),
            consumer_metrics_log_interval=_finite_float(
                "consumer_metrics_log_interval",
                extra.get("consumer_metrics_log_interval", 10),
                True,
            ),
        )
