# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test for the UCCL P2P connector using a fake wrapper."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.uccl_p2p import (
    UcclP2pConnector,
    UcclP2pConnectorScheduler,
    UcclP2pConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.uccl_p2p.metadata import (
    UcclP2pHandshakePayload,
)

from .utils import create_vllm_config, make_kv_cache_config


@dataclass
class FakeXferDesc:
    """Minimal fake XferDesc matching the uccl.p2p interface."""

    addr: int = 0
    size: int = 0
    mr_id: int = 0
    lkeys: list[int] | None = None
    rkeys: list[int] | None = None


class FakeUcclP2pWrapper:
    """Stub that mimics the UcclP2pWrapper interface for smoke testing."""

    def __init__(self, local_gpu_idx: int):
        self._md = b"fake_md"
        self._conn_ids: dict[str, int] = {}
        self._counter = 0

    def get_agent_metadata(self) -> bytes:
        return self._md

    def register_memory(self, tensor_list: list[torch.Tensor]) -> list[Any]:
        descs = [
            FakeXferDesc(addr=t.data_ptr(), size=t.numel() * t.element_size())
            for t in tensor_list
        ]
        return descs

    def deregister_memory(self, desc_list: list[Any]) -> None:
        pass

    def get_serialized_descs(self, desc_list: list[Any]) -> bytes:
        return b"fake_serialized"

    def deserialize_descs(self, serialized: bytes) -> list[Any]:
        return [FakeXferDesc()]

    def add_remote_agent(
        self,
        agent_name: str,
        endpoint_metadata: bytes,
    ) -> str:
        self._counter += 1
        conn_id = self._counter
        self._conn_ids[agent_name] = conn_id
        return agent_name

    def remove_remote_agent(self, agent_name: str) -> None:
        self._conn_ids.pop(agent_name, None)

    def make_prepped_xfer(
        self,
        op_name: str,
        local_descs: list[Any],
        remote_descs: list[Any],
        notif_msg: bytes | None = None,
        agent_name: str | None = None,
    ) -> int:
        self._counter += 1
        return self._counter

    def transfer(self, transfer_id: int) -> None:
        pass

    def check_xfer_state(self, transfer_id: int) -> str:
        return "DONE"

    def release_xfer_handle(self, transfer_id: int) -> None:
        pass

    def release_dlist_handle(self, handle: Any) -> None:
        pass

    def get_xfer_telemetry(self, transfer_id: int) -> dict[str, Any]:
        return {}

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    def shutdown(self) -> None:
        pass


def _monkeypatch_worker_init(monkeypatch):
    """Replace the real UcclP2pWrapper with FakeUcclP2pWrapper."""
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.uccl_p2p.worker.UcclP2pWrapper",
        FakeUcclP2pWrapper,
    )
    # Mock TP group functions for standalone testing.
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.uccl_p2p.worker"
        ".get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.uccl_p2p.worker"
        ".get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def test_uccl_p2p_connector_worker_smoke(monkeypatch):
    """Smoke test: UcclP2pConnector in WORKER role with fake wrapper."""
    _monkeypatch_worker_init(monkeypatch)

    vllm_config = create_vllm_config(
        kv_connector="UcclP2pConnector",
        kv_role="kv_consumer",
        disable_hybrid_kv_cache_manager=True,
    )
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size,
    )

    with set_current_vllm_config(vllm_config):
        conn = UcclP2pConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config=kv_cache_config
        )
        assert conn.connector_worker is not None
        assert isinstance(conn.connector_worker, UcclP2pConnectorWorker)

        # Run register_kv_caches with a trivial tensor map.
        fake_kv = {"layer0": torch.randn(100, 16, 4, 16, dtype=torch.float32)}
        conn.register_kv_caches(fake_kv)

        # Verify handshake metadata was produced.
        md = conn.get_handshake_metadata()
        assert md is not None
        assert isinstance(md, UcclP2pHandshakePayload)
        assert md.compatibility_hash
        assert md.agent_metadata_bytes

        # get_finished should return empty sets.
        finished_sending, finished_recving = conn.get_finished(set())
        assert finished_sending == set()
        assert finished_recving == set()

        conn.shutdown()


def test_uccl_p2p_connector_scheduler_smoke(monkeypatch):
    """Smoke test: UcclP2pConnector in SCHEDULER role."""
    _monkeypatch_worker_init(monkeypatch)

    vllm_config = create_vllm_config(
        kv_connector="UcclP2pConnector",
        kv_role="kv_consumer",
        disable_hybrid_kv_cache_manager=True,
    )
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size,
    )

    with set_current_vllm_config(vllm_config):
        conn = UcclP2pConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=kv_cache_config
        )
        assert conn.connector_scheduler is not None
        assert isinstance(conn.connector_scheduler, UcclP2pConnectorScheduler)
        conn.shutdown()
