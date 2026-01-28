# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.logger import init_logger

WorkerAddr = str

logger = init_logger(__name__)


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

    def __init__(self, vllm_config: VllmConfig, host: str, port: int):
        self.workers: dict[int, EngineEntry] = {}

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
        return self.workers


# Workaround for #27987
# Drop the last "-{random_uuid():.8}"
# After #32630 or other solution is merged, we can remove this workaround.
class TruncatingDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._store = dict()
        self.update(dict(*args, **kwargs))

    @staticmethod
    def _truncate_key(key):
        if not isinstance(key, str) or len(key) < 10:
            raise TypeError("Keys must be strings with at least 10 characters")
        return key[:-9]

    def __setitem__(self, key, value):
        truncated_key = self._truncate_key(key)
        self._store[truncated_key] = value

    def __getitem__(self, key):
        truncated_key = self._truncate_key(key)
        return self._store[truncated_key]

    def __delitem__(self, key):
        truncated_key = self._truncate_key(key)
        del self._store[truncated_key]

    def __contains__(self, key):
        truncated_key = self._truncate_key(key)
        return truncated_key in self._store

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    def __repr__(self):
        return f"{type(self).__name__}({self._store})"
