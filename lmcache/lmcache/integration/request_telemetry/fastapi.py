# SPDX-License-Identifier: Apache-2.0
"""
FastAPI-based request telemetry reporter.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
import json
import threading

# First Party
from lmcache.integration.request_telemetry.base import RequestTelemetry
from lmcache.logging import init_logger

logger = init_logger(__name__)


class FastAPIRequestTelemetry(RequestTelemetry):
    """
    FastAPI-based request telemetry reporter.

    Sends telemetry events to a FastAPI server endpoint asynchronously
    using a thread pool to avoid blocking the main execution path.

    Config dict keys:
        endpoint: The FastAPI endpoint URL to send telemetry events to. (required)
        timeout: Timeout in seconds for HTTP requests. Defaults to 5.0.
        max_workers: Maximum number of threads for async HTTP requests.
            Defaults to 2.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        endpoint = config.get("endpoint", None)
        if endpoint is None:
            raise ValueError(
                "FastAPIRequestTelemetry requires setting endpoint. "
                "Please set LMCACHE_REQUEST_TELEMETRY_ENDPOINT envvar."
            )

        self._endpoint = endpoint
        self._timeout = config.get("timeout", 5.0)
        self._executor = ThreadPoolExecutor(
            max_workers=config.get("max_workers", 2),
            thread_name_prefix="fastapi-telemetry",
        )
        self._lock = threading.Lock()
        self._closed = False

        logger.info(
            "LMCache FastAPIRequestTelemetry initialized with endpoint: %s",
            self._endpoint,
        )

    def on_request_store_finished(
        self,
        request_ids_set: set[str],
        model_name: str,
        world_size: int,
        kv_rank: int,
    ) -> None:
        """Send request store finished event to the FastAPI endpoint."""
        if self._closed:
            return

        payload = {
            "event": "request_store_finished",
            "request_ids_set": list(request_ids_set),
            "model_name": model_name,
            "world_size": world_size,
            "kv_rank": kv_rank,
        }

        # Submit async HTTP request
        self._executor.submit(self._send_event, payload)

    def _send_event(self, payload: dict[str, Any]) -> None:
        """Send an event to the FastAPI endpoint (runs in thread pool)."""
        try:
            data = json.dumps(payload).encode("utf-8")
            request = Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # TODO: this introduces blocking IO, should use asyncio.
            with urlopen(request, timeout=self._timeout) as response:
                if response.status >= 400:
                    logger.warning(
                        f"FastAPI telemetry request failed with"
                        f" status {response.status}"
                    )
        except URLError as e:
            logger.warning(f"FastAPI telemetry request failed: {e}")
        except Exception as e:
            logger.warning(f"FastAPI telemetry request failed unexpectedly: {e}")

    def close(self) -> None:
        """Shutdown the thread pool executor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._executor.shutdown(wait=True)
        logger.info("FastAPIRequestTelemetry closed")
