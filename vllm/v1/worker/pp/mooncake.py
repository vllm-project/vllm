# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered PP cache writes using MooncakeConnector's memory operations."""

from collections.abc import Callable, Mapping
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer_engine import (
    batch_write,
    create_transfer_engine,
    register_memory,
    unregister_memory,
)
from vllm.utils.network_utils import get_ip


class PPMooncakeTransport:
    def __init__(
        self,
        state: Mapping,
        device_module: Any,
        protocol: str,
        device_name: str,
        engine_factory: Callable[[], Any] | None = None,
    ):
        self.state = state
        self.regions = [
            (layer, key, raw.data_ptr(), raw.nbytes)
            for layer in sorted(state)
            for key, raw in sorted(state[layer].items())
        ]
        self.addresses = [region[2] for region in self.regions]
        sizes = [region[3] for region in self.regions]
        self.registered_bytes = sum(sizes)
        self._closed = False
        self._peer_regions: dict[tuple[int, str], int] = {}
        self._peer_session: str | None = None
        hostname = get_ip()
        device = next(iter(next(iter(state.values())).values())).device
        try:
            self.engine = create_transfer_engine(
                hostname, protocol, device_name, engine_factory=engine_factory
            )
        finally:
            # NVLink engine initialization may change the calling thread's device.
            if device.type != "cpu":
                device_module.set_device(device)
        port = self.engine.get_rpc_port()
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError("Mooncake did not provide a valid RPC port")
        self.session = f"{hostname}:{port}"
        register_memory(self.engine, self.addresses, sizes)

    def bind_peer(self, session: str, regions: list) -> None:
        expected = {(layer, key): size for layer, key, _, size in self.regions}
        actual = {}
        addresses = {}
        for layer, key, address, size in regions:
            if type(address) is not int or address <= 0 or (layer, key) in actual:
                raise ValueError("Invalid Mooncake peer cache region")
            actual[layer, key] = size
            addresses[layer, key] = address
        if actual != expected or not isinstance(session, str) or not session:
            raise ValueError("Mooncake peer cache regions differ")
        self._peer_regions, self._peer_session = addresses, session

    def write(self, layers: list[int], max_chunk_bytes: int) -> None:
        if self._closed or self._peer_session is None:
            raise RuntimeError("Mooncake PP transport is closed or has no peer")
        sources, targets, sizes = [], [], []
        for layer in layers:
            for key, raw in sorted(self.state[layer].items()):
                target = self._peer_regions[layer, key]
                for start in range(0, raw.nbytes, max_chunk_bytes):
                    sources.append(raw.data_ptr() + start)
                    targets.append(target + start)
                    sizes.append(min(raw.nbytes - start, max_chunk_bytes))
        ret = batch_write(self.engine, self._peer_session, sources, targets, sizes)
        if ret != 0:
            raise RuntimeError(f"Mooncake PP cache write failed (ret={ret})")

    def close(self) -> None:
        if self._closed:
            return
        unregister_memory(self.engine, self.addresses)
        self._closed = True
        # Keep tensors alive until after the engine releases its registrations.
        self.engine = None
