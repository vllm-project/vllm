from pathlib import Path

from vllm.envs import VLLM_ASSETS_CACHE


def get_cache_dir():
    """Get the path to the cache for storing downloaded assets."""
    path = Path(VLLM_ASSETS_CACHE)
    path.mkdir(parents=True, exist_ok=True)

    return path
