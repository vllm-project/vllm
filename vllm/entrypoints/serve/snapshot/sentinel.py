# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import requests

from vllm.logger import init_logger
from vllm.snapshot.utils import (
    RETRY_INTERVAL,
    RETRY_LOG_FREQUENCY,
    get_local_ip,
    is_restore,
    load_snapshot_metadata,
)

logger = init_logger(__name__)

SUSPEND_TIMEOUT = 3600.0
RESUME_TIMEOUT = 3600.0
DEVICE_UNLOCK_TIMEOUT = 10.0
HEALTH_TIMEOUT = 5.0


class SnapshotSentinel(threading.Thread):
    """Drive the snapshot suspend/checkpoint/resume lifecycle."""

    def __init__(
        self,
        snapshot_metadata: str,
        port: int,
        use_tls: bool,
        ca_file: str | None,
    ) -> None:
        super().__init__(name="snapshot-sentinel", daemon=True)
        self._port = port
        self._scheme = "https" if use_tls else "http"
        self._verify = ca_file if use_tls and ca_file else True
        self._snapshot_metadata = snapshot_metadata
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._wait_until_infer_healthy()
        if self._stop_event.is_set():
            return

        logger.info("[snapshot] Infer is healthy, starting to suspend")
        self._call_suspend()
        if self._stop_event.is_set():
            return

        self._reach_checkpoint()
        if self._stop_event.is_set():
            return

        logger.info("[snapshot] Restored from host-side snapshot, starting to resume")
        self._call_resume()

    def _request(
        self,
        method: str,
        path: str,
        timeout: float,
        host: str,
        params: dict[str, str] | None = None,
    ) -> None:
        formatted_host = f"[{host}]" if ":" in host else host
        response = requests.request(
            method,
            f"{self._scheme}://{formatted_host}:{self._port}/{path.lstrip('/')}",
            params=params,
            timeout=timeout,
            verify=self._verify,
        )
        response.raise_for_status()

    def _wait_until_infer_healthy(self) -> None:
        retries = 0
        host = get_local_ip()
        while not self._stop_event.is_set():
            try:
                self._request("GET", "/health", HEALTH_TIMEOUT, host)
                return
            except Exception as exc:
                if retries % RETRY_LOG_FREQUENCY == 0:
                    logger.warning(
                        "[snapshot] Infer health check failed, will retry: %s",
                        exc,
                    )
                retries += 1
                self._stop_event.wait(RETRY_INTERVAL)

    def _call_suspend(self) -> None:
        retries = 0
        host = get_local_ip()
        while not self._stop_event.is_set():
            model_save_path = None
            try:
                model_save_path = load_snapshot_metadata(
                    self._snapshot_metadata, "model_save_path"
                )
                self._request(
                    "POST",
                    "/suspend",
                    SUSPEND_TIMEOUT,
                    host,
                    {"model_save_path": model_save_path},
                )
                logger.info(
                    "[snapshot] Suspend completed, model_save_path=%r",
                    model_save_path,
                )
                return
            except Exception as exc:
                if retries % RETRY_LOG_FREQUENCY == 0:
                    logger.warning(
                        "[snapshot] Suspend request failed %s times, will retry, "
                        "model_save_path=%r: %s",
                        retries,
                        model_save_path,
                        exc,
                    )
                retries += 1
                self._stop_event.wait(RETRY_INTERVAL)

    def _reach_checkpoint(self) -> None:
        retries = 0
        host = get_local_ip()
        while not self._stop_event.is_set() and not is_restore():
            try:
                checkpoint = load_snapshot_metadata(
                    self._snapshot_metadata, "checkpoint"
                )
                if checkpoint != "done":
                    raise ValueError("Container checkpoint is not done")

                self._request(
                    "POST", "/device_unlock", DEVICE_UNLOCK_TIMEOUT, host
                )
                logger.info(
                    "[snapshot] Checkpoint completed, device unlocked; stopping "
                    "snapshot sentinel"
                )
                self._stop_event.set()
                return
            except Exception as exc:
                if retries % RETRY_LOG_FREQUENCY == 0:
                    logger.warning(
                        "[snapshot] Checkpoint not reached, will retry: %s", exc
                    )
                retries += 1
                self._stop_event.wait(RETRY_INTERVAL)

    def _call_resume(self) -> None:
        retries = 0
        # Resume runs after restore, so resolve the new Pod IP once and reuse
        # it for all request retries.
        host = get_local_ip()
        while not self._stop_event.is_set():
            model_load_path = None
            data_parallel_master_ip = None
            try:
                model_load_path = load_snapshot_metadata(
                    self._snapshot_metadata, "model_load_path"
                )
                data_parallel_master_ip = load_snapshot_metadata(
                    self._snapshot_metadata, "data_parallel_master_ip"
                )
                self._request(
                    "POST",
                    "/resume",
                    RESUME_TIMEOUT,
                    host,
                    {
                        "model_path": model_load_path,
                        "data_parallel_master_ip": data_parallel_master_ip,
                    },
                )
                logger.info(
                    "[snapshot] Resume completed, model_path=%r, "
                    "data_parallel_master_ip=%r",
                    model_load_path,
                    data_parallel_master_ip,
                )
                return
            except Exception as exc:
                if retries % RETRY_LOG_FREQUENCY == 0:
                    logger.warning(
                        "[snapshot] Resume request failed %s times, will retry, "
                        "model_path=%r, data_parallel_master_ip=%r: %s",
                        retries,
                        model_load_path,
                        data_parallel_master_ip,
                        exc,
                    )
                retries += 1
                self._stop_event.wait(RETRY_INTERVAL)
