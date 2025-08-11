# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import asyncio
import aiohttp
from quart import Quart, make_response, request, Response
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration parameters
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60)  # Timeout for backend service requests (seconds)
MAX_CONCURRENT_REQUESTS = 10  # Maximum concurrent requests to backend services
REQUEST_QUEUE_SIZE = 50  # Maximum number of requests in the queue
RATE_LIMIT = 5  # Maximum requests per second (rate limiting)
PRE_SERVICE_URL = "http://localhost:8100/v1/completions"  # Prefill service endpoint
DECODE_SERVICE_URL = "http://localhost:8200/v1/completions"  # Decode service endpoint

app = Quart(__name__)


# Token bucket rate limiter implementation
class RateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit  # Requests per second
        self.tokens = rate_limit  # Available tokens
        self.last_refill = time.monotonic()  # Last token refill time
        self.lock = asyncio.Lock()  # Synchronization lock

    async def acquire(self):
        """Acquire a token from the rate limiter"""
        async with self.lock:
            current_time = time.monotonic()
            elapsed = current_time - self.last_refill

            # Refill tokens if more than 1 second has passed
            if elapsed > 1.0:
                self.tokens = self.rate_limit
                self.last_refill = current_time

            # Check if tokens are available
            if self.tokens > 0:
                self.tokens -= 1
                return True

            # Calculate wait time if no tokens available
            wait_time = 1.0 - elapsed
            await asyncio.sleep(wait_time)
            self.last_refill = time.monotonic()
            self.tokens = self.rate_limit - 1
            return True


# Request queue manager with concurrency control
class RequestQueue:
    def __init__(self, max_concurrent, max_queue_size):
        self.max_concurrent = max_concurrent  # Maximum concurrent requests
        self.max_queue_size = max_queue_size  # Maximum queue size
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Concurrency control
        self.queue = deque()  # Request queue
        self.queue_size = 0  # Current queue size

    async def enqueue(self, task):
        """Add a request task to the queue"""
        if self.queue_size >= self.max_queue_size:
            logger.warning("Request queue full, rejecting request")
            return False

        self.queue.append(task)
        self.queue_size += 1
        return True

    async def process(self):
        """Process queued requests using semaphore for concurrency control"""
        while True:
            if self.queue:
                async with self.semaphore:
                    task = self.queue.popleft()
                    self.queue_size -= 1
                    await task
            await asyncio.sleep(0.01)  # Yield control to event loop


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
    # Apply rate limiting before making request
    await rate_limiter.acquire()

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status == 200:
                    # Stream response chunks
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    # Handle backend service errors
                    error_text = await response.text()
                    logger.error(f"Backend service error: {response.status} - {error_text}")
                    yield b'{"error": "Backend service error"}'
        except aiohttp.ClientError as e:
            # Handle connection errors
            logger.error(f"Connection error to {url}: {str(e)}")
            yield b'{"error": "Service unavailable"}'
        except asyncio.TimeoutError:
            # Handle timeout errors
            logger.error(f"Timeout connecting to {url}")
            yield b'{"error": "Service timeout"}'


async def process_request():
    """Process a single request through prefill and decode stages"""
    try:
        original_request_data = await request.get_json()

        # Create prefill request (max_tokens=1)
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1

        # Execute prefill stage
        async for _ in forward_request(PRE_SERVICE_URL, prefill_request):
            continue

        # Execute decode stage and stream response
        generator = forward_request(DECODE_SERVICE_URL, original_request_data)
        response = await make_response(generator)
        response.timeout = None  # Disable timeout for streaming response
        return response

    except Exception as e:
        # Handle internal server errors
        import traceback
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            response=b'{"error": "Internal server error"}',
            status=500,
            content_type="application/json"
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
            content_type="application/json"
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
            content_type="application/json"
        )


if __name__ == "__main__":
    # Start the Quart server
    app.run(port=8000, host="0.0.0.0")