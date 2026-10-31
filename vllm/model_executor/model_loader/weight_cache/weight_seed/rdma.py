# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``rdma`` weight mover: pull storages through the Mooncake TransferEngine.

The seed it publishes is plain JSON (a session string and integer regions),
so it is the only mover usable on the remote control plane.
"""

from collections.abc import Mapping

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    RDMA_BACKEND,
    WeightMover,
)
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.network_utils import get_ip

try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = PlaceholderModule("mooncake.engine").placeholder_attr(
        "TransferEngine"
    )

logger = init_logger(__name__)


class RdmaMover(WeightMover):
    """Pull storages through the Mooncake TransferEngine."""

    name = RDMA_BACKEND
    is_node_local = False

    def __init__(self, protocol: str = "rdma"):
        self.protocol = protocol
        self._engine = None
        self._source_session: str | None = None
        # Mooncake refuses to register memory twice, so each storage is
        # registered once and serves every later mirror.
        self._registered: dict[int, int] = {}

    def _initialize_engine(self):
        engine = TransferEngine()
        if engine.initialize(get_ip(), "P2PHANDSHAKE", self.protocol, "") != 0:
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        return engine, f"{get_ip()}:{engine.get_rpc_port()}"

    def _get_engine(self):
        if self._engine is None:
            self._engine, self._source_session = self._initialize_engine()
        return self._engine

    def prepare_seed(self, storages: list[torch.UntypedStorage]) -> dict:
        engine = self._get_engine()
        regions = [[storage.data_ptr(), storage.nbytes()] for storage in storages]
        new = [
            (pointer, nbytes)
            for pointer, nbytes in regions
            if nbytes and pointer not in self._registered
        ]
        if new:
            pointers, lengths = (list(column) for column in zip(*new))
            if engine.batch_register_memory(pointers, lengths) != 0:
                raise RuntimeError("Mooncake failed to register weight-cache storages")
            self._registered.update(new)
        return {
            "backend": self.name,
            "session": self._source_session,
            "regions": regions,
        }

    def copy_into(
        self,
        storages: list[Mapping],
        seed: Mapping,
        buffers: list[torch.Tensor],
    ) -> None:
        session = seed.get("session")
        if not session:
            raise RuntimeError("Source daemon did not publish a Mooncake session")
        regions = seed.get("regions")
        if not isinstance(regions, list) or len(regions) != len(storages):
            raise RuntimeError("Source daemon published no region per storage")
        transfers = [
            (buffer.data_ptr(), remote_pointer, buffer.nbytes)
            for buffer, (remote_pointer, _) in zip(buffers, regions)
            if buffer.nbytes
        ]
        if not transfers:
            return
        local_pointers, remote_pointers, lengths = (
            list(column) for column in zip(*transfers)
        )
        engine, _ = self._initialize_engine()
        if engine.batch_register_memory(local_pointers, lengths) != 0:
            raise RuntimeError("Mooncake failed to register mirror storages")
        try:
            ret = engine.batch_transfer_sync_read(
                session, local_pointers, remote_pointers, lengths
            )
        finally:
            engine.batch_unregister_memory(local_pointers)
        if ret != 0:
            raise RuntimeError(f"Mooncake weight-cache read failed with status {ret}")

    def release(self) -> None:
        if self._engine is not None and self._registered:
            self._engine.batch_unregister_memory(list(self._registered))
            self._registered.clear()
