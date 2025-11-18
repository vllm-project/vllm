"""Lab 06: Error Handling - Complete Solution"""

import asyncio
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def generate_with_retry(prompt: str) -> str:
    """Generate with automatic retry on failure."""
    # Simulated generation
    return f"Generated: {prompt}"


async def generate_with_timeout(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Generate with timeout."""
    try:
        return await asyncio.wait_for(generate_with_retry(prompt), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Timeout after {timeout}s")
        return None


class CircuitBreaker:
    """Circuit breaker for error handling."""

    def __init__(self, failure_threshold: int = 5):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.state = "CLOSED"

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker."""
        if self.state == "OPEN":
            raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise


async def main():
    """Main error handling demo."""
    print("=== Error Handling Lab ===\n")

    # Test retry
    result = await generate_with_retry("Test prompt")
    print(f"Retry result: {result}\n")

    # Test timeout
    result = await generate_with_timeout("Test", timeout=5.0)
    print(f"Timeout result: {result}\n")

    # Test circuit breaker
    cb = CircuitBreaker(failure_threshold=3)
    print(f"Circuit breaker state: {cb.state}")


if __name__ == "__main__":
    asyncio.run(main())
