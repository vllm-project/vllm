# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import threading
import time
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    """A centralized registry for prefiller connection info (IP, port, ranks)."""

    def __init__(self, host: str, port: int):
        self.workers: dict[int, EngineEntry] = {}

        self.host = host
        self.port = port
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None
        self._socket: socket.socket | None = None
        self._startup_error: BaseException | None = None

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        # All methods are async. No need to use lock to protect data.
        self.app.post("/register")(self.register_worker)
        self.app.get("/query", response_model=dict[int, EngineEntry])(self.query)

    def start(self, *, timeout: float = 30.0):
        if self.server_thread:
            if (
                self.server_thread.is_alive()
                and self.server is not None
                and self.server.started
                and not self.server.should_exit
                and self._socket is not None
                and self._socket.fileno() >= 0
            ):
                return
            raise RuntimeError(
                "Mooncake bootstrap server is still starting or stopping"
            )

        try:
            # Bind in the caller so errors propagate, and retain the listener
            # when handing it to Uvicorn, including for automatically chosen ports.
            family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            self._socket = socket.create_server((self.host, self.port), family=family)
            self.port = self._socket.getsockname()[1]
            config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
            server = self.server = uvicorn.Server(config=config)
            listener = self._socket
            self._startup_error = None

            def run():
                try:
                    server.run(sockets=[listener])
                except BaseException as exc:
                    self._startup_error = exc

            self.server_thread = threading.Thread(
                target=run, name="mooncake_bootstrap_server", daemon=True
            )
            self.server_thread.start()
            deadline = time.monotonic() + timeout
            while not server.started:
                if not self.server_thread.is_alive():
                    raise RuntimeError(
                        "Mooncake bootstrap server exited during startup"
                    ) from self._startup_error
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Mooncake bootstrap server did not start in time"
                    )
                time.sleep(0.01)
        except BaseException:
            self.shutdown()
            raise
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        was_started = self.server is not None and self.server.started
        if self.server is not None:
            self.server.should_exit = True
        if self.server_thread is not None and self.server_thread.ident is not None:
            self.server_thread.join(timeout=5)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self.server_thread is not None and self.server_thread.is_alive():
            logger.warning("Mooncake bootstrap server did not stop in time")
        else:
            self.server_thread = None
            self.server = None
            if was_started:
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
        existing = tp_entry.get(payload.pp_rank)
        if existing is not None:
            # A client timeout can fire after the server recorded the
            # registration so an identical retry must not be an error.
            if existing == payload.addr:
                return {"status": "ok"}
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Worker with dp_rank={payload.dp_rank}, "
                    f"tp_rank={payload.tp_rank}, pp_rank={payload.pp_rank} "
                    f"is already registered at "
                    f"{existing}, "
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
