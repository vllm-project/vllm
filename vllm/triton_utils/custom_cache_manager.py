import os

from triton.runtime.cache import (FileCacheManager, default_cache_dir,
                                  default_dump_dir, default_override_dir)

from vllm.logger import init_logger

logger = init_logger(__name__)


def maybe_set_triton_cache_manager() -> None:
    """Set environment variable to tell Triton to use a
    custom cache manager"""
    cache_manger = os.environ.get("TRITON_CACHE_MANAGER", None)
    if cache_manger is None:
        manager = "vllm.triton_utils.custom_cache_manager:CustomCacheManager"
        logger.info("Setting Triton cache manager to: %s", manager)
        os.environ["TRITON_CACHE_MANAGER"] = manager


class CustomCacheManager(FileCacheManager):
    """Re-implements Triton's cache manager, ensuring that a
    unique cache directory is created for each process. This is
    needed to avoid collisions when running with tp>1 and
    using multi-processing as the distributed backend.

    Note this issue was fixed by triton-lang/triton/pull/4295,
    but the fix is not yet included in triton==v3.0.0. However,
    it should be included in the subsequent version.
    """

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = os.getenv("TRITON_CACHE_DIR",
                                       "").strip() or default_cache_dir()
            if self.cache_dir:
                self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")
