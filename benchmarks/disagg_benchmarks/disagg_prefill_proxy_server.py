# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import logging
import os

import aiohttp
from quart import Quart, Response, make_response, request
from rate_limiter import RateLimiter
from request_queue import RequestQueue

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
        default=300,
        help="Timeout for backend service requests in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum concurrent requests to backend services (default: 100)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=500,
        help="Maximum number of requests in the queue (default: 500)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=40,
        help="Maximum requests per second (default: 40)",
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
        default="http://localhost:8100/v1/completions",
        help="Prefill service endpoint URL",
    )
    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://localhost:8200/v1/completions",
        help="Decode service endpoint URL",
    )

    return parser.parse_args()


def main():
    """parse command line arguments"""
    args = parse_args()

    # Initialize configuration using command line parameters
    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    REQUEST_QUEUE_SIZE = args.queue_size
    RATE_LIMIT = args.rate_limit
    PREFILL_SERVICE_URL = args.prefill_url
    DECODE_SERVICE_URL = args.decode_url
    PORT = args.port

    app = Quart(__name__)

    # Initialize the rate limiter and request queue
    rate_limiter = RateLimiter(RATE_LIMIT)
    request_queue = RequestQueue(MAX_CONCURRENT_REQUESTS, REQUEST_QUEUE_SIZE)

    # Attach the configuration object to the application instance
    app.config.update(
        {
            "AIOHTTP_TIMEOUT": AIOHTTP_TIMEOUT,
            "rate_limiter": rate_limiter,
            "request_queue": request_queue,
            "PREFILL_SERVICE_URL": PREFILL_SERVICE_URL,
            "DECODE_SERVICE_URL": DECODE_SERVICE_URL,
        }
    )

    # Start queue processing on app startup
    @app.before_serving
    async def startup():
        """Start request processing task when app starts serving"""
        asyncio.create_task(request_queue.process())

    async def forward_request(url, data):
        """Forward request to backend service with rate limiting and error handling"""
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        # Use rate limiter as context manager
        async with (
            rate_limiter,
            aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
        ):
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if response.status == 200:
                        # Stream response chunks
                        async for chunk_bytes in response.content.iter_chunked(1024):
                            yield chunk_bytes
                    else:
                        # Handle backend service errors
                        error_text = await response.text()
                        logger.error(
                            "Backend service error: %s - %s",
                            response.status,
                            error_text,
                        )
                        yield b'{"error": "Backend service error"}'
            except aiohttp.ClientError as e:
                # Handle connection errors
                logger.error("Connection error to %s: %s", url, str(e))
                yield b'{"error": "Service unavailable"}'
            except asyncio.TimeoutError:
                # Handle timeout errors
                logger.error("Timeout connecting to %s", url)
                yield b'{"error": "Service timeout"}'

    async def process_request():
        """Process a single request through prefill and decode stages"""
        try:
            original_request_data = await request.get_json()

            # Create prefill request (max_tokens=1)
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1

            # Execute prefill stage
            async for _ in forward_request(PREFILL_SERVICE_URL, prefill_request):
                continue

            # Execute decode stage and stream response
            generator = forward_request(DECODE_SERVICE_URL, original_request_data)
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
        # Create task for request processing
        task = asyncio.create_task(process_request())

        # Enqueue request or reject if queue is full
        if not await request_queue.enqueue(task):
            return Response(
                response=b'{"error": "Server busy, try again later"}',
                status=503,
                content_type="application/json",
            )

        try:
            # Return the response from the processing task
            return await task
        except asyncio.CancelledError:
            # Handle task cancellation (timeout or queue full)
            logger.warning("Request cancelled due to timeout or queue full")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    # Start the Quart server with host can be set to 0.0.0.0
    app.run(port=PORT)


if __name__ == "__main__":
    main()
