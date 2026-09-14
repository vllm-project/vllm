# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instance registration for the EPD proxy."""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import zmq

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

DEFAULT_ANNOUNCE_INTERVAL = 30.0
_REQUEST_TIMEOUT_MS = 5000


class ProxyRegistrar:
    """Periodically announce one instance to the proxy registry."""

    def __init__(
        self,
        registry_addr: str,
        payload: dict[str, Any],
        interval: float = DEFAULT_ANNOUNCE_INTERVAL,
    ) -> None:
        self.registry_addr = registry_addr
        self.payload = payload
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="epd-register", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        self._wait_for_service()
        if self._stop.is_set():
            return
        context = zmq.Context()
        try:
            while not self._stop.is_set():
                self._send(context, "register")
                self._stop.wait(self.interval)
            self._send(context, "unregister")
        finally:
            context.term()

    def _wait_for_service(self) -> None:
        parsed = urlsplit(self.payload["url"])
        assert parsed.hostname is not None and parsed.port is not None
        while not self._stop.is_set():
            try:
                endpoint = (parsed.hostname, parsed.port)
                with socket.create_connection(endpoint, timeout=1):
                    return
            except OSError:
                self._stop.wait(1)

    def _send(self, context: zmq.Context, operation: str) -> None:
        try:
            with context.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVTIMEO, _REQUEST_TIMEOUT_MS)
                sock.setsockopt(zmq.SNDTIMEO, _REQUEST_TIMEOUT_MS)
                sock.connect(self.registry_addr)
                sock.send_json({**self.payload, "operation": operation})
                response = sock.recv_json()
                if not response.get("ok"):
                    raise RuntimeError(response.get("error"))
        except Exception as error:
            logger.warning("EPD registration failed: %s", error)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def infer_role(vllm_config: VllmConfig) -> str:
    ec_config = getattr(vllm_config, "ec_transfer_config", None)
    if ec_config is not None and ec_config.is_encode_only:
        return "encode"
    kv_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_config is not None and kv_config.is_kv_producer:
        return "prefill"
    return "decode"


def registrar_from_vllm_config(
    vllm_config: VllmConfig,
    *,
    ec_zmq_addrs: list[str] | None = None,
    dp_rank: int | None = None,
) -> ProxyRegistrar | None:
    ec_config = getattr(vllm_config, "ec_transfer_config", None)
    if ec_config is None or not ec_config.is_ec_transfer_instance:
        return None
    get = ec_config.get_from_extra_config
    registry_addr = get("proxy_registry_addr", None)
    url = get("_http_address", None)
    if not registry_addr or not url:
        return None
    return ProxyRegistrar(
        registry_addr,
        {
            "role": infer_role(vllm_config),
            "url": url,
            "engine_id": ec_config.engine_id,
            "dp_rank": dp_rank,
            "dp_size": vllm_config.parallel_config.data_parallel_size,
            "ec_zmq_addrs": ec_zmq_addrs or [],
        },
        float(get("proxy_announce_interval", DEFAULT_ANNOUNCE_INTERVAL)),
    )


def start_worker_registration(vllm_config: VllmConfig) -> ProxyRegistrar | None:
    """Register one worker per replica; Mooncake registers its runtime endpoints."""
    ec_config = getattr(vllm_config, "ec_transfer_config", None)
    if ec_config is not None and ec_config.ec_connector == "ECMooncakeConnector":
        return None
    registrar = registrar_from_vllm_config(
        vllm_config, dp_rank=vllm_config.parallel_config.data_parallel_index
    )
    if registrar is not None:
        from vllm.distributed.parallel_state import (
            get_pcp_group,
            get_pp_group,
            get_tp_group,
        )

        if any(
            group.rank_in_group != 0
            for group in (get_tp_group(), get_pp_group(), get_pcp_group())
        ):
            return None
        registrar.start()
    return registrar
