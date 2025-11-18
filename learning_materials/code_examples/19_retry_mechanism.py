"""
Example 19: Retry Mechanism with Exponential Backoff

Implements robust retry logic for handling transient failures.

Usage:
    python 19_retry_mechanism.py
"""

import asyncio
import random
from typing import Optional, Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransientError(Exception):
    """Simulated transient error."""
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientError),
    before_sleep=before_sleep_log(logger, logging.INFO)
)
async def generate_with_retry(prompt: str, fail_rate: float = 0.3) -> str:
    """
    Simulate generation with potential failures.

    Args:
        prompt: Input prompt
        fail_rate: Probability of failure (for simulation)

    Returns:
        Generated text
    """
    # Simulate random failures
    if random.random() < fail_rate:
        logger.warning(f"Transient failure for prompt: '{prompt}'")
        raise TransientError("Simulated transient error")

    # Simulate successful generation
    await asyncio.sleep(0.1)
    return f"Generated response for: {prompt}"


async def robust_generate(prompt: str) -> Optional[str]:
    """
    Generate with full error handling.

    Returns None if all retries exhausted.
    """
    try:
        result = await generate_with_retry(prompt, fail_rate=0.5)
        logger.info(f"Success: {prompt}")
        return result
    except TransientError as e:
        logger.error(f"Failed after all retries: {prompt}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


async def main():
    """Demo retry mechanism."""
    print("=== Retry Mechanism Demo ===\n")

    prompts = [
        "What is machine learning?",
        "Explain deep learning",
        "What is vLLM?",
    ]

    results = await asyncio.gather(*[
        robust_generate(prompt) for prompt in prompts
    ])

    print("\n=== Results ===")
    for prompt, result in zip(prompts, results):
        status = "SUCCESS" if result else "FAILED"
        print(f"{status}: {prompt}")
        if result:
            print(f"  {result}")


if __name__ == "__main__":
    asyncio.run(main())
