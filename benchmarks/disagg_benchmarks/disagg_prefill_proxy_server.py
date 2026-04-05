# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import logging
import os
import time
import uuid
from urllib.parse import urlparse

import aiohttp
from quart import Quart, Response, make_response, request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy server")

    # Add args
    parser.add_argument(
        "--timeout",
        type=float,
        default=6 * 60 * 60,
        help="Timeout for backend service requests in seconds (default: 21600)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--prefill-url",
        type=str,
        default="http://localhost:8100",
        help="Prefill service base URL (protocol + host[:port])",
    )
    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://localhost:8200",
        help="Decode service base URL (protocol + host[:port])",
    )
    parser.add_argument(
        "--kv-host",
        type=str,
        default="localhost",
        help="Hostname or IP used by KV transfer (default: localhost)",
    )
    parser.add_argument(
        "--prefill-kv-port",
        type=int,
        default=14579,
        help="Prefill KV port (default: 14579)",
    )
    parser.add_argument(
        "--decode-kv-port",
        type=int,
        default=14580,
        help="Decode KV port (default: 14580)",
    )

    return parser.parse_args()


def main():
    """parse command line arguments"""
    args = parse_args()

    # Initialize configuration using command line parameters
    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PORT = args.port

    PREFILL_KV_ADDR = f"{args.kv_host}:{args.prefill_kv_port}"
    DECODE_KV_ADDR = f"{args.kv_host}:{args.decode_kv_port}"

    logger.info(
        "Proxy resolved KV addresses -> prefill: %s, decode: %s",
        PREFILL_KV_ADDR,
        DECODE_KV_ADDR,
    )

    app = Quart(__name__)

    # Attach the configuration object to the application instance so helper
    # coroutines can read the resolved backend URLs and timeouts without using
    # globals.
    app.config.update(
        {
            "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
            "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
            "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
            "PREFILL_KV_ADDR": PREFILL_KV_ADDR,
            "DECODE_KV_ADDR": DECODE_KV_ADDR,
        }
    )

    def _normalize_base_url(url: str) -> str:
        """Remove any trailing slash so path joins behave predictably."""
        return url.rstrip("/")

    def _get_host_port(url: str) -> str:
        """Return the hostname:port portion for logging and KV headers."""
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port
        if port is None:
            port = 80 if parsed.scheme == "http" else 443
        return f"{host}:{port}"

    PREFILL_BASE = _normalize_base_url(PREFILL_SERVICE_URL)
    DECODE_BASE = _normalize_base_url(DECODE_SERVICE_URL)
    KV_TARGET = _get_host_port(DECODE_SERVICE_URL)

    def _build_headers(request_id: str) -> dict[str, str]:
        """Construct the headers expected by vLLM's P2P disagg connector."""
        headers: dict[str, str] = {"X-Request-Id": request_id, "X-KV-Target": KV_TARGET}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _run_prefill(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ) -> dict:
        url = f"{PREFILL_BASE}{request_path}"
        start_ts = time.perf_counter()
        logger.info("[prefill] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Prefill backend error {resp.status}: {error_text}"
                    )
                response_data = await resp.json()
                logger.info(
                    "[prefill] done request_id=%s status=%s elapsed=%.2fs",
                    request_id,
                    resp.status,
                    time.perf_counter() - start_ts,
                )
                return response_data
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Prefill service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Prefill service unavailable at {url}") from exc

    async def _stream_decode(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ):
        url = f"{DECODE_BASE}{request_path}"
        # Stream tokens from the decode service once the prefill stage has
        # materialized KV caches on the target workers.
        logger.info("[decode] start request_id=%s url=%s", request_id, url)
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Decode backend error %s - %s", resp.status, error_text
                    )
                    err_msg = (
                        '{"error": "Decode backend error ' + str(resp.status) + '"}'
                    )
                    yield err_msg.encode()
                    return
                logger.info(
                    "[decode] streaming response request_id=%s status=%s",
                    request_id,
                    resp.status,
                )
                async for chunk_bytes in resp.content.iter_chunked(1024):
                    yield chunk_bytes
                logger.info("[decode] finished streaming request_id=%s", request_id)
        except asyncio.TimeoutError:
            logger.error("Decode service timeout at %s", url)
            yield b'{"error": "Decode service timeout"}'
        except aiohttp.ClientError as exc:
            logger.error("Decode service error at %s: %s", url, exc)
            yield b'{"error": "Decode service unavailable"}'

    async def process_request():
        """Process a single request through prefill and decode stages"""
        try:
            original_request_data = await request.get_json()

            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            prefill_request["stream"] = False
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1
            prefill_request["kv_transfer_params"] = {
                "remote_kv_addr": DECODE_KV_ADDR,
            }

            request_id = str(uuid.uuid4())

            headers = _build_headers(request_id)
            prefill_response = await _run_prefill(
                request.path, prefill_request, headers, request_id
            )

            kv_transfer_params = prefill_response.get("kv_transfer_params", {})
            logger.info("[proxy] kv_transfer_params: %s", kv_transfer_params)

            decode_request = original_request_data.copy()
            if kv_transfer_params:
                decode_request["kv_transfer_params"] = kv_transfer_params

            generator = _stream_decode(
                request.path, decode_request, headers, request_id
            )
            response = await make_response(generator)
            response.timeout = None  # Disable timeout for streaming response
            return response

        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/v1/completions", methods=["POST"])
    async def handle_request():
        """Handle incoming API requests with concurrency and rate limiting"""
        try:
            return await process_request()
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    # Start the Quart server with host can be set to 0.0.0.0
    app.run(port=PORT)


if __name__ == "__main__":
    main()
