# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.distributed.io_transfer.io_connector.base import (
    IOConnectorBase,
    IOConnectorRole,
)
from vllm.distributed.io_transfer.io_connector.factory import IOConnectorFactory

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_IO_CONNECTOR_AGENT: IOConnectorBase | None = None
_IO_EXECUTOR: ThreadPoolExecutor | None = None


def get_io_transfer() -> IOConnectorBase:
    assert _IO_CONNECTOR_AGENT is not None, "IO transfer connector is not initialized"
    return _IO_CONNECTOR_AGENT


def has_io_transfer() -> bool:
    return _IO_CONNECTOR_AGENT is not None


def get_io_executor() -> Executor | None:
    """Return the dedicated IO thread pool, or None if it was not created."""
    return _IO_EXECUTOR


def ensure_io_transfer_initialized(
    vllm_config: "VllmConfig", create_io_executor: bool = False
) -> None:
    """Initialize the process-wide IO transfer connector if configured.

    When *create_io_executor* is set, also create a dedicated thread pool for
    blocking connector calls, sized by ``max_io_threads`` in the connector's
    extra config, so they do not occupy renderer workers.
    """
    global _IO_CONNECTOR_AGENT, _IO_EXECUTOR

    io_transfer_config = vllm_config.io_transfer_config
    if io_transfer_config is None or not io_transfer_config.is_io_transfer_instance:
        return

    if _IO_CONNECTOR_AGENT is None:
        _IO_CONNECTOR_AGENT = IOConnectorFactory.create_connector(
            config=vllm_config, role=IOConnectorRole.from_config(io_transfer_config)
        )

    if create_io_executor and _IO_EXECUTOR is None:
        _IO_EXECUTOR = ThreadPoolExecutor(
            max_workers=io_transfer_config.get_from_extra_config(
                "max_io_threads", None
            ),
            thread_name_prefix="io-transfer",
        )


def ensure_io_transfer_shutdown() -> None:
    """Shut down the IO transfer connector and its thread pool."""
    global _IO_CONNECTOR_AGENT, _IO_EXECUTOR
    if _IO_EXECUTOR is not None:
        _IO_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _IO_EXECUTOR = None
    if _IO_CONNECTOR_AGENT is not None:
        _IO_CONNECTOR_AGENT.shutdown()
        _IO_CONNECTOR_AGENT = None
