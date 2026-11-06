# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake memory operations shared by PD and PP ownership handoff."""

from collections.abc import Callable
from typing import Any


def create_transfer_engine(
    hostname: str,
    protocol: str,
    device_name: str,
    *,
    engine_factory: Callable[[], Any] | None = None,
) -> Any:
    if engine_factory is None:
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise RuntimeError(
                "Mooncake transfer requires the platform-compatible "
                "mooncake-transfer-engine package"
            ) from exc
        engine_factory = TransferEngine
    engine = engine_factory()
    if engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name) != 0:
        raise RuntimeError("Mooncake Transfer Engine initialization failed.")
    return engine


def register_memory(engine: Any, addresses: list[int], sizes: list[int]) -> None:
    if engine.batch_register_memory(addresses, sizes) != 0:
        raise RuntimeError("Mooncake batch memory registration failed.")


def unregister_memory(engine: Any, addresses: list[int]) -> None:
    if engine.batch_unregister_memory(addresses) != 0:
        raise RuntimeError("Mooncake batch memory unregistration failed.")


def batch_write(
    engine: Any,
    peer: str,
    sources: list[int],
    targets: list[int],
    sizes: list[int],
) -> int:
    return engine.batch_transfer_sync_write(peer, sources, targets, sizes)
