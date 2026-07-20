# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.logger import init_logger

WorkerAddr = str

logger = init_logger(__name__)


def get_mooncake_dp_engine_index(parallel_config: ParallelConfig) -> int:
    """Return the per-engine DP index used for Mooncake side channels."""
    if parallel_config.local_engines_only:
        assert parallel_config.data_parallel_rank_local is not None
        return parallel_config.data_parallel_rank_local

    return parallel_config.data_parallel_index


class RegisterWorkerPayload(BaseModel):
    engine_id: EngineId
    dp_rank: int
    tp_rank: int
    pp_rank: int
    addr: WorkerAddr


@dataclass
class EngineEntry:
    engine_id: EngineId
    # {tp_rank: {pp_rank: worker_addr}}
    worker_addr: dict[int, dict[int, WorkerAddr]]


class MooncakeBootstrapServer:
    """
    A centralized server running on the global rank 0 prefiller worker.
    Prefiller workers register their connection info (IP, port, ranks) here.
    """

    def __init__(
        self,
        host: str,
        port: int,
        expected_tp_size: int = 1,
        expected_pp_size: int = 1,
        wait_for_complete_topology: bool = False,
    ):
        if expected_tp_size < 1 or expected_pp_size < 1:
            raise ValueError("Expected TP and PP sizes must be positive.")

        self.workers: dict[int, EngineEntry] = {}
        self.expected_tp_size = expected_tp_size
        self.expected_pp_size = expected_pp_size
        self.wait_for_complete_topology = wait_for_complete_topology

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        # All methods are async. No need to use lock to protect data.
        self.app.post("/register")(self.register_worker)
        self.app.get("/query", response_model=dict[int, EngineEntry])(self.query)

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="mooncake_bootstrap_server", daemon=True
        )
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)  # Wait for the server to start
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("Mooncake Bootstrap Server stopped.")

    async def register_worker(self, payload: RegisterWorkerPayload):
        """Handles registration of a prefiller worker."""
        if payload.tp_rank < 0 or payload.tp_rank >= self.expected_tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TP rank {payload.tp_rank} must be less than TP size "
                    f"{self.expected_tp_size}."
                ),
            )
        if payload.pp_rank < 0 or payload.pp_rank >= self.expected_pp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PP rank {payload.pp_rank} must be less than PP size "
                    f"{self.expected_pp_size}."
                ),
            )
        if payload.dp_rank not in self.workers:
            self.workers[payload.dp_rank] = EngineEntry(
                engine_id=payload.engine_id,
                worker_addr={},
            )

        dp_entry = self.workers[payload.dp_rank]
        if dp_entry.engine_id != payload.engine_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine ID mismatch for dp_rank={payload.dp_rank}: "
                    f"expected {dp_entry.engine_id}, got {payload.engine_id}"
                ),
            )
        if payload.tp_rank not in dp_entry.worker_addr:
            dp_entry.worker_addr[payload.tp_rank] = {}

        tp_entry = dp_entry.worker_addr[payload.tp_rank]
        if payload.pp_rank in tp_entry:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Worker with dp_rank={payload.dp_rank}, "
                    f"tp_rank={payload.tp_rank}, pp_rank={payload.pp_rank} "
                    f"is already registered at "
                    f"{tp_entry[payload.pp_rank]}, "
                    f"but still want to register at {payload.addr}"
                ),
            )

        tp_entry[payload.pp_rank] = payload.addr
        logger.debug(
            "Registered worker: engine_id=%s, dp_rank=%d, tp_rank=%d, pp_rank=%d at %s",
            payload.engine_id,
            payload.dp_rank,
            payload.tp_rank,
            payload.pp_rank,
            payload.addr,
        )

        return {"status": "ok"}

    async def query(self) -> dict[int, EngineEntry]:
        if self.wait_for_complete_topology and not self._is_topology_complete():
            raise HTTPException(
                status_code=503,
                detail="Mooncake prefiller worker topology is not ready.",
            )
        return self.workers

    def _is_topology_complete(self) -> bool:
        if not self.workers:
            return False

        expected_tp_ranks = set(range(self.expected_tp_size))
        expected_pp_ranks = set(range(self.expected_pp_size))
        return all(
            set(entry.worker_addr) == expected_tp_ranks
            and all(
                set(tp_entry) == expected_pp_ranks
                for tp_entry in entry.worker_addr.values()
            )
            for entry in self.workers.values()
        )
