"""
Example 18: Response Caching Strategy

Implements caching to reduce redundant inference calls.

Usage:
    python 18_caching_strategy.py
"""

import hashlib
import time
from typing import Dict, Optional
from functools import lru_cache
from vllm import LLM, SamplingParams


class ResponseCache:
    """Simple in-memory cache for inference results."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds

    def _get_key(self, prompt: str, params: dict) -> str:
        """Generate cache key from prompt and parameters."""
        content = f"{prompt}:{str(sorted(params.items()))}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt: str, params: dict) -> Optional[str]:
        """Get cached result if available and not expired."""
        key = self._get_key(prompt, params)

        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                # Expired, remove
                del self.cache[key]

        return None

    def set(self, prompt: str, params: dict, result: str) -> None:
        """Cache a result."""
        key = self._get_key(prompt, params)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]

        self.cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


def generate_with_cache(
    llm: LLM,
    cache: ResponseCache,
    prompt: str,
    params: dict
) -> tuple[str, bool]:
    """Generate with caching. Returns (result, was_cached)."""
    # Check cache
    cached_result = cache.get(prompt, params)
    if cached_result:
        return cached_result, True

    # Generate
    sampling_params = SamplingParams(**params)
    output = llm.generate([prompt], sampling_params)[0]
    result = output.outputs[0].text

    # Cache result
    cache.set(prompt, params, result)

    return result, False


def main():
    """Demo caching strategy."""
    print("=== Response Caching Demo ===\n")

    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
    cache = ResponseCache(max_size=100, ttl=3600)

    params = {"temperature": 0.8, "max_tokens": 50}

    # First request (cache miss)
    prompt = "The future of technology"
    print(f"Request 1: '{prompt}'")
    start = time.time()
    result, cached = generate_with_cache(llm, cache, prompt, params)
    elapsed = time.time() - start
    print(f"  Cached: {cached}, Time: {elapsed:.3f}s")
    print(f"  Result: {result[:60]}...\n")

    # Second request (cache hit)
    print(f"Request 2: '{prompt}' (same prompt)")
    start = time.time()
    result, cached = generate_with_cache(llm, cache, prompt, params)
    elapsed = time.time() - start
    print(f"  Cached: {cached}, Time: {elapsed:.3f}s")
    print(f"  Result: {result[:60]}...\n")

    # Third request (different prompt, cache miss)
    prompt2 = "Machine learning is"
    print(f"Request 3: '{prompt2}' (different prompt)")
    start = time.time()
    result, cached = generate_with_cache(llm, cache, prompt2, params)
    elapsed = time.time() - start
    print(f"  Cached: {cached}, Time: {elapsed:.3f}s")
    print(f"  Result: {result[:60]}...\n")

    print(f"Cache size: {len(cache.cache)} entries")


if __name__ == "__main__":
    main()
