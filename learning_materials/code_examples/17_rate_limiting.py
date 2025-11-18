"""
Example 17: Rate Limiting Implementation

Shows how to implement rate limiting for API protection.

Usage:
    python 17_rate_limiting.py
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: int = 10):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token for rate limiting."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Check if we have tokens
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class UserRateLimiter:
    """Per-user rate limiting."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.user_requests: Dict[str, list] = defaultdict(list)

    def can_proceed(self, user_id: str) -> bool:
        """Check if user can make request."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if req_time > minute_ago
        ]

        # Check limit
        if len(self.user_requests[user_id]) < self.rpm:
            self.user_requests[user_id].append(now)
            return True
        return False


async def simulate_requests():
    """Simulate rate-limited requests."""
    print("=== Rate Limiting Demo ===\n")

    limiter = RateLimiter(requests_per_second=5)
    user_limiter = UserRateLimiter(requests_per_minute=10)

    # Test global rate limiter
    print("Testing global rate limiter (5 req/s):")
    successful = 0
    for i in range(20):
        if await limiter.acquire():
            successful += 1
            print(f"  Request {i+1}: ALLOWED")
        else:
            print(f"  Request {i+1}: RATE LIMITED")
        await asyncio.sleep(0.1)

    print(f"\nSuccessful: {successful}/20\n")

    # Test per-user rate limiter
    print("Testing per-user rate limiter (10 req/min):")
    for i in range(15):
        if user_limiter.can_proceed("user1"):
            print(f"  User1 Request {i+1}: ALLOWED")
        else:
            print(f"  User1 Request {i+1}: RATE LIMITED")


if __name__ == "__main__":
    asyncio.run(simulate_requests())
