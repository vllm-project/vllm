# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

# Timeout for backend service requests (seconds)
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=300)
# Maximum concurrent requests to backend services
MAX_CONCURRENT_REQUESTS = 100
REQUEST_QUEUE_SIZE = 500  # Maximum number of requests in the queue
RATE_LIMIT = 40  # Maximum requests per second (rate limiting)
# Prefill service endpoint
PREFILL_SERVICE_URL = "http://localhost:8100/v1/completions"
# Decode service endpoint
DECODE_SERVICE_URL = "http://localhost:8200/v1/completions"
# run this need pip install quart
app = Quart(__name__)

# Initialize rate limiter and request queue
rate_limiter = RateLimiter(RATE_LIMIT)
request_queue = RequestQueue(MAX_CONCURRENT_REQUESTS, REQUEST_QUEUE_SIZE)


# Start queue processing on app startup
@app.before_serving
async def startup():
    """Start request processing task when app starts serving"""
    asyncio.create_task(request_queue.process())


async def forward_request(url, data):
    """Forward request to backend service with rate limiting and error handling"""
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

    # Use rate limiter as context manager
    async with rate_limiter:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    if response.status == 200:
                        # Stream response chunks
                        async for chunk_bytes in response.content.iter_chunked(1024):
                            yield chunk_bytes
                    else:
                        # Handle backend service errors
                        error_text = await response.text()
                        logger.error("Backend service error: %s - %s",
                                     response.status, error_text)
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


if __name__ == "__main__":
    # Start the Quart server with host can be set to 0.0.0.0
    app.run(port=8000)
