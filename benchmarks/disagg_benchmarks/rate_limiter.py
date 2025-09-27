# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, rate_limit):
        self.rate_limit = rate_limit  # Requests per second
        self.num_available_tokens = rate_limit  # Available tokens
        self.last_refill = time.monotonic()  # Last token refill time
        self.lock = asyncio.Lock()  # Synchronization lock

    async def acquire(self):
        """Acquire a token from the rate limiter"""
        while True:
            async with self.lock:
                current_time = time.monotonic()
                elapsed = current_time - self.last_refill

                # Refill num_available_tokens if more than 1 second has passed
                if elapsed > 1.0:
                    self.num_available_tokens = self.rate_limit
                    self.last_refill = current_time

                # Check if num_available_tokens are available
                if self.num_available_tokens > 0:
                    self.num_available_tokens -= 1
                    return True

                # Calculate wait time if no num_available_tokens available
                wait_time = 1.0 - elapsed
            await asyncio.sleep(wait_time)

    async def __aenter__(self):
        """Enter async context manager - acquire token"""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit async context manager - no cleanup needed"""
        pass
