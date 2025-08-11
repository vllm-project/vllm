# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import asyncio
import time
from collections import deque
from quart import Quart, make_response, request, jsonify, g
import aiohttp

# Configuration parameters
MAX_CONCURRENT_REQUESTS = 10  # Maximum concurrent requests allowed
REQUEST_RATE_LIMIT = 100  # Token bucket capacity (max burst requests)
REQUEST_RATE_REFILL = 5  # Token refill rate per second
CONCURRENCY_TIMEOUT = 5.0  # Timeout for acquiring concurrency slot (seconds)
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=1 * 60 * 60)  # Backend request timeout  (unit:hour)

app = Quart(__name__)


# Rate limiter implementation using token bucket algorithm
class RateLimiter:
    def __init__(self, rate_limit, refill_rate):
        self.rate_limit = rate_limit  # Maximum tokens (requests) in bucket
        self.refill_rate = refill_rate  # Tokens added per second
        self.tokens = rate_limit  # Current token count
        self.last_refill = time.monotonic()  # Last refill timestamp
        self.lock = asyncio.Lock()  # Thread-safe lock

    async def acquire(self):
        """Acquire a token from the bucket if available."""
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_refill
            if elapsed > 0:
                # Calculate tokens to add and cap at bucket capacity
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + elapsed * self.refill_rate
                )
                self.last_refill = now

            # Check if token is available
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


# Concurrency limiter using semaphore
class ConcurrencyLimiter:
    def __init__(self, max_concurrent):
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Semaphore for concurrency control

    async def acquire(self):
        """Acquire a concurrency slot with timeout."""
        try:
            # Try to acquire slot with timeout
            await asyncio.wait_for(self.semaphore.acquire(), timeout=CONCURRENCY_TIMEOUT)
            return True
        except asyncio.TimeoutError:
            return False

    def release(self):
        """Release a concurrency slot."""
        self.semaphore.release()


# Initialize limiters
rate_limiter = RateLimiter(REQUEST_RATE_LIMIT, REQUEST_RATE_REFILL)
concurrency_limiter = ConcurrencyLimiter(MAX_CONCURRENT_REQUESTS)

# Backend service endpoints with load balancing
SERVICE_ENDPOINTS = {
    "prefill": [
        "http://localhost:8100/v1/completions",
        # Add more instances for horizontal scaling:
        # "http://prefill-instance2:8100/v1/completions",
    ],
    "decode": [
        "http://localhost:8200/v1/completions",
        # Add more instances for horizontal scaling:
        # "http://decode-instance2:8200/v1/completions",
    ]
}

# Initialize round-robin state
endpoint_pointers = {service: 0 for service in SERVICE_ENDPOINTS}
endpoint_locks = {service: asyncio.Lock() for service in SERVICE_ENDPOINTS}


def get_next_endpoint(service_type):
    """
    Get next endpoint using round-robin load balancing.

    Args:
        service_type: 'prefill' or 'decode'

    Returns:
        str: Endpoint URL
    """

    async def _get_next():
        async with endpoint_locks[service_type]:
            endpoints = SERVICE_ENDPOINTS[service_type]
            if not endpoints:
                raise ValueError(f"No endpoints available for {service_type}")

            # Round-robin selection
            idx = endpoint_pointers[service_type]
            endpoint = endpoints[idx]
            endpoint_pointers[service_type] = (idx + 1) % len(endpoints)
            return endpoint

    return asyncio.run(_get_next())


async def forward_request(url, data):
    """
    Forward request to backend service with streaming response.

    Args:
        url: Backend service URL
        data: Request payload

    Yields:
        bytes: Response chunks
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                # Stream response in 1KB chunks
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


@app.before_request
async def before_request_handler():
    """
    Pre-request handler for rate limiting and concurrency control.

    Returns:
        Response: Error response if limits are exceeded
    """
    # Initialize concurrency tracking
    g.concurrency_acquired = False

    # Rate limiting check
    if not await rate_limiter.acquire():
        app.logger.warning("Rate limit exceeded")
        return jsonify({"error": "Too many requests"}), 429

    # Concurrency limiting check
    if not await concurrency_limiter.acquire():
        app.logger.warning("Concurrency limit reached")
        return jsonify({"error": "Service busy, please try again later"}), 503

    # Mark concurrency slot as acquired
    g.concurrency_acquired = True


@app.teardown_request
async def teardown_request(exc):
    """
    Post-request handler to release concurrency slot.

    Args:
        exc: Exception if any occurred during handling
    """
    if getattr(g, 'concurrency_acquired', False):
        concurrency_limiter.release()


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    """
    Main request handler for completion requests.

    Returns:
        Response: Streaming response or error
    """
    try:
        # Parse incoming request
        original_request_data = await request.get_json()

        # Phase 1: Prefill request
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1  # Force prefill-only
        prefill_url = get_next_endpoint("prefill")

        # Execute prefill (no streaming needed)
        async for _ in forward_request(prefill_url, prefill_request):
            continue  # Drain prefill response

        # Phase 2: Decode request
        decode_url = get_next_endpoint("decode")
        generator = forward_request(decode_url, original_request_data)

        # Create and return streaming response
        response = await make_response(generator)
        response.timeout = None  # Disable Quart timeout for streaming
        return response

    except Exception as e:
        # Error handling and logging
        import sys
        import traceback

        exc_info = sys.exc_info()
        app.logger.error("Error in proxy server")
        app.logger.error(f"Exception: {str(e)}")
        app.logger.error("Traceback:\n" + "".join(traceback.format_exception(*exc_info)))

        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    # Start server on all interfaces
    app.run(port=8000, host="0.0.0.0")