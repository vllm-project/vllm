# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Encoder-cache (EC) connector backed by Mooncake TransferEngine.

Used in disaggregated setups where an encoder / prefill instance produces
multimodal encoder outputs and a decode instance loads them over RDMA-capable
Mooncake transport instead of shared filesystem.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import torch
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import is_local_first_rank
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    TransferEngine = None  # type: ignore[misc, assignment]
    _MOONCAKE_IMPORT_ERROR = e
else:
    _MOONCAKE_IMPORT_ERROR = None


@dataclass
class ECMooncakeLoadSpec:
    """Per-item metadata shipped from scheduler to worker (pickle-friendly)."""

    mm_hash: str
    num_token: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    producer_zmq: str


@dataclass
class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Worker-side metadata for one scheduler step."""

    loads: list[ECMooncakeLoadSpec] = field(default_factory=list)

    def add_load(self, spec: ECMooncakeLoadSpec) -> None:
        self.loads.append(spec)


class ECMooncakeRegistryServer:
    """Lightweight HTTP registry on the producer for remote has_cache_item / info."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

    def _register_routes(self) -> None:
        @self.app.get("/ec/info/{mm_hash}")
        async def ec_info(mm_hash: str) -> dict[str, Any]:
            with self._lock:
                data = self._entries.get(mm_hash)
            if data is None:
                raise HTTPException(status_code=404, detail="unknown mm_hash")
            return data

    def start(self) -> None:
        if self.server_thread is not None:
            return
        config = uvicorn.Config(app=self.app, host=self.host, port=self.port)
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="ec_mooncake_registry", daemon=True
        )
        self.server_thread.start()
        while self.server is not None and not self.server.started:
            time.sleep(0.05)
        logger.info(
            "EC Mooncake registry listening on http://%s:%d", self.host, self.port
        )

    def shutdown(self) -> None:
        if self.server is None or self.server_thread is None or not self.server.started:
            return
        self.server.should_exit = True
        self.server_thread.join()
        logger.info("EC Mooncake registry stopped.")

    def publish(self, mm_hash: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._entries[mm_hash] = payload

    def unpublish(self, mm_hash: str) -> None:
        with self._lock:
            self._entries.pop(mm_hash, None)


class ECMooncakeConnector(ECConnectorBase):
    """
    EC connector using Mooncake TransferEngine for GPU tensor transport.

    Extra config (``ec_connector_extra_config``):

    - ``remote_registry_url`` (consumer, required): Base URL of the producer
      registry, e.g. ``http://192.168.0.2:9018``.
    - ``registry_http_port`` (producer, optional): Port for the in-process HTTP
      registry (default ``9018``).
    - ``mooncake_protocol`` (optional): Passed to ``TransferEngine.initialize``
      (default ``"rdma"``).

    Limitations: ``tensor_parallel_size`` and ``pipeline_parallel_size`` must
    be ``1`` (same assumption as Mooncake KV connector for P2P handshake).
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        if _MOONCAKE_IMPORT_ERROR is not None or TransferEngine is None:
            raise ImportError(
                "Install mooncake-transfer-engine (see "
                "https://github.com/kvcache-ai/Mooncake ) to use ECMooncakeConnector."
            ) from _MOONCAKE_IMPORT_ERROR

        if vllm_config.parallel_config.tensor_parallel_size > 1:
            raise ValueError("ECMooncakeConnector requires tensor_parallel_size=1.")
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "ECMooncakeConnector does not support pipeline parallelism yet."
            )

        self._role = role
        self._ec_cfg = vllm_config.ec_transfer_config
        assert self._ec_cfg is not None
        self._extra = self._ec_cfg.ec_connector_extra_config
        self._protocol: str = self._extra.get("mooncake_protocol", "rdma")
        self._remote_registry_url: str | None = self._extra.get("remote_registry_url")
        self._registry_http_port: int = int(self._extra.get("registry_http_port", 9018))

        # Scheduler (consumer): mm_hash -> pending tensor layout from registry
        self._pending_specs: dict[str, ECMooncakeLoadSpec] = {}
        self._mm_datas_need_loads: dict[str, int] = {}

        # Worker producer
        self._engine: TransferEngine | None = None
        self._hostname = get_ip()
        self._registry: ECMooncakeRegistryServer | None = None
        self._zmq_listen_addr: str | None = None
        self._zmq_thread: threading.Thread | None = None
        self._zmq_ctx: zmq.Context | None = None
        self._tensor_by_hash: dict[str, torch.Tensor] = {}
        self._tensor_lock = threading.Lock()
        self._producer_services_started = False

        if role == ECConnectorRole.SCHEDULER and self.is_consumer:
            if not self._remote_registry_url:
                raise ValueError(
                    "ec_consumer with ECMooncakeConnector requires "
                    "ec_connector_extra_config['remote_registry_url']."
                )

    def _ensure_engine(self) -> TransferEngine:
        if self._engine is None:
            eng = TransferEngine()
            ret = eng.initialize(self._hostname, "P2PHANDSHAKE", self._protocol, "")
            if ret != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed.")
            self._engine = eng
            logger.info(
                "ECMooncakeConnector TransferEngine ready at %s:%d",
                self._hostname,
                eng.get_rpc_port(),
            )
        return self._engine

    def _start_producer_zmq_listener(self) -> None:
        if self._zmq_thread is not None:
            return

        def loop() -> None:
            assert self._zmq_ctx is not None
            sock = self._zmq_ctx.socket(zmq.REP)
            port = sock.bind_to_random_port(f"tcp://{self._hostname}")
            self._zmq_listen_addr = f"tcp://{self._hostname}:{port}"
            logger.info("EC Mooncake pull listener at %s", self._zmq_listen_addr)
            eng = self._ensure_engine()
            while True:
                try:
                    raw = sock.recv()
                except zmq.ContextTerminated:
                    break
                try:
                    req = json.loads(raw.decode("utf-8"))
                    if req.get("op") != "pull":
                        sock.send_json({"ok": False, "err": "unknown op"})
                        continue
                    mm_hash = req["mm_hash"]
                    dst_session = req["dst_session"]
                    dst_ptr = int(req["dst_ptr"])
                    nbytes = int(req["nbytes"])
                    with self._tensor_lock:
                        tensor = self._tensor_by_hash.get(mm_hash)
                    if tensor is None:
                        sock.send_json({"ok": False, "err": "unknown mm_hash"})
                        continue
                    src_ptr = tensor.data_ptr()
                    if tensor.nbytes != nbytes:
                        sock.send_json({"ok": False, "err": "size mismatch"})
                        continue
                    ret = eng.batch_transfer_sync_write(
                        dst_session, [src_ptr], [dst_ptr], [nbytes]
                    )
                    sock.send_json({"ok": ret == 0, "mooncake_ret": int(ret)})
                except Exception as e:
                    logger.exception("EC Mooncake pull handler error: %s", e)
                    try:
                        sock.send_json({"ok": False, "err": str(e)})
                    except zmq.ZMQError:
                        break

        self._zmq_ctx = zmq.Context()
        self._zmq_thread = threading.Thread(target=loop, name="ec-mooncake-zmq", daemon=True)
        self._zmq_thread.start()
        while self._zmq_listen_addr is None:
            time.sleep(0.01)

    def _ensure_producer_services(self) -> None:
        if self._producer_services_started:
            return
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
            return
        self._ensure_engine()
        self._start_producer_zmq_listener()
        if is_local_first_rank():
            self._registry = ECMooncakeRegistryServer("0.0.0.0", self._registry_http_port)
            self._registry.start()
        self._producer_services_started = True

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        eng = self._ensure_engine()
        raw_buf = self._ec_cfg.ec_buffer_device
        buf = (
            raw_buf.lower()
            if isinstance(raw_buf, str) and raw_buf
            else "cuda"
        )
        if buf == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ECMooncakeConnector requires CUDA for ec_buffer_device=cuda")
        device = torch.device(buf)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                continue
            torch_dtype = getattr(torch, spec.dtype, None)
            if torch_dtype is None:
                raise ValueError(f"Unsupported torch dtype string: {spec.dtype!r}")
            t = torch.empty(spec.shape, dtype=torch_dtype, device=device)
            ret = eng.batch_register_memory([t.data_ptr()], [t.nbytes])
            if ret != 0:
                raise RuntimeError(
                    "Mooncake EC batch_register_memory failed on consumer."
                )
            pull = {
                "op": "pull",
                "mm_hash": spec.mm_hash,
                "dst_session": f"{self._hostname}:{eng.get_rpc_port()}",
                "dst_ptr": t.data_ptr(),
                "nbytes": t.nbytes,
            }
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 120_000)
            sock.connect(spec.producer_zmq)
            try:
                sock.send_json(pull)
                resp = sock.recv_json()
            finally:
                sock.close(linger=0)
                ctx.term()
            if not resp.get("ok"):
                raise RuntimeError(f"EC Mooncake pull failed: {resp}")
            encoder_cache[spec.mm_hash] = t
            logger.debug("Loaded EC tensor for mm_hash=%s via Mooncake", spec.mm_hash)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
            return
        self._ensure_producer_services()
        tensor = encoder_cache[mm_hash]
        eng = self._ensure_engine()
        ret = eng.batch_register_memory([tensor.data_ptr()], [tensor.nbytes])
        if ret != 0:
            raise RuntimeError("Mooncake EC batch_register_memory failed on producer.")
        with self._tensor_lock:
            self._tensor_by_hash[mm_hash] = tensor

        dtype_str = str(tensor.dtype).split(".")[-1]
        payload = {
            "nbytes": tensor.nbytes,
            "shape": list(tensor.shape),
            "dtype": dtype_str,
            "producer_zmq": self._zmq_listen_addr,
        }
        if self._registry is not None:
            self._registry.publish(mm_hash, payload)
        logger.debug("Published EC tensor mm_hash=%s to registry", mm_hash)

    def has_cache_item(self, identifier: str) -> bool:
        if not self.is_consumer or self._role != ECConnectorRole.SCHEDULER:
            return False
        assert self._remote_registry_url is not None
        url = self._remote_registry_url.rstrip("/") + f"/ec/info/{identifier}"
        try:
            r = httpx.get(url, timeout=5.0)
        except httpx.HTTPError as e:
            logger.warning("EC Mooncake registry query failed for %s: %s", identifier, e)
            return False
        if r.status_code != 200:
            return False
        data = r.json()
        zmq_addr = data.get("producer_zmq")
        if not zmq_addr:
            return False
        self._pending_specs[identifier] = ECMooncakeLoadSpec(
            mm_hash=identifier,
            num_token=0,
            nbytes=int(data["nbytes"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
            producer_zmq=str(zmq_addr),
        )
        return True

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = ECMooncakeConnectorMetadata()
        for mm_hash, num_token in self._mm_datas_need_loads.items():
            spec = self._pending_specs.get(mm_hash)
            if spec is None:
                logger.warning("Missing EC Mooncake spec for mm_hash=%s", mm_hash)
                continue
            meta.add_load(
                ECMooncakeLoadSpec(
                    mm_hash=spec.mm_hash,
                    num_token=num_token,
                    nbytes=spec.nbytes,
                    shape=spec.shape,
                    dtype=spec.dtype,
                    producer_zmq=spec.producer_zmq,
                )
            )
            self._pending_specs.pop(mm_hash, None)
        self._mm_datas_need_loads.clear()
        return meta

    def __del__(self) -> None:
        try:
            if self._registry is not None:
                self._registry.shutdown()
            if self._zmq_ctx is not None:
                self._zmq_ctx.term()
        except Exception:
            pass
