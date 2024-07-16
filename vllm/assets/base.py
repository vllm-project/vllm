from pathlib import Path

import vllm.envs as envs


def get_cache_dir():
    """Get the path to the cache for storing downloaded assets."""
    path = Path(envs.VLLM_ASSETS_CACHE)
    path.mkdir(parents=True, exist_ok=True)

    return path
