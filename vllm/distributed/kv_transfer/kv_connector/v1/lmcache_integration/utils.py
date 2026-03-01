# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
import os
import threading
from typing import TYPE_CHECKING, Union

import torch
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig as V1Config

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"

# Thread-safe singleton storage
_config_instance: V1Config | None = None
_config_lock = threading.Lock()


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_or_create_config() -> V1Config:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.

    This function is thread-safe and implements singleton pattern,
    ensuring the configuration is loaded only once.
    """
    global _config_instance

    # Double-checked locking for thread-safe singleton
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:  # Check again within lock
                LMCacheEngineConfig = V1Config  # type: ignore[assignment]

                if "LMCACHE_CONFIG_FILE" not in os.environ:
                    logger.warning(
                        "No LMCache configuration file is set. Trying to read"
                        " configurations from the environment variables."
                    )
                    logger.warning(
                        "You can set the configuration file through "
                        "the environment variable: LMCACHE_CONFIG_FILE"
                    )
                    _config_instance = LMCacheEngineConfig.from_env()
                else:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    logger.info("Loading LMCache config file %s", config_file)
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                    # Update config from environment variables
                    _config_instance.update_config_from_env()
    return _config_instance


def hex_hash_to_int16(s: str) -> int:
    """
    Convert a hex hash string to a 16-bit integer.
    """
    return int(s, 16) & 0xFFFF


def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
) -> torch.Tensor:
    """
    Overwrite token_ids in-place for multimodal placeholders using
    efficient slice assignments.
    """
    n = token_ids.size(0)
    for hash_str, placeholder in zip(mm_hashes, mm_positions):
        start, length = placeholder.offset, placeholder.length
        if start >= n:
            continue
        end = min(start + length, n)
        token_ids[start:end] = hex_hash_to_int16(hash_str)
    return token_ids


def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )


def create_lmcache_metadata(
    vllm_config=None, model_config=None, parallel_config=None, cache_config=None
):
    """
    Create LMCacheEngineMetadata from vLLM configuration.

    This function extracts common metadata creation logic that was duplicated
    across multiple files.

    Args:
        vllm_config (VllmConfig): vLLM configuration object containing model,
                                  parallel, and cache configs (alternative to
                                  individual config parameters)
        model_config (ModelConfig): Model configuration (alternative to
                                    vllm_config)
        parallel_config (ParallelConfig): Parallel configuration (alternative
                                          to vllm_config)
        cache_config (CacheConfig): Cache configuration (alternative to
                                    vllm_config)
    """
    # Third Party
    # First Party
    from lmcache.config import LMCacheEngineMetadata

    from vllm.utils.torch_utils import get_kv_cache_torch_dtype

    config = lmcache_get_or_create_config()
    # Support both vllm_config object and individual config parameters
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        if model_config is None or parallel_config is None or cache_config is None:
            raise ValueError(
                "Either vllm_config must be provided, or all of "
                "model_config, parallel_config, and cache_config must be provided."
            )
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    # Get KV cache dtype
    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)

    # Check if MLA is enabled
    use_mla = mla_enabled(model_cfg)

    # Construct KV shape (for memory pool)
    num_layer = model_cfg.get_num_layers(parallel_cfg)
    chunk_size = config.chunk_size
    num_kv_head = model_cfg.get_num_kv_heads(parallel_cfg)
    head_size = model_cfg.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    # Create metadata
    metadata = LMCacheEngineMetadata(
        model_cfg.model,
        parallel_cfg.world_size,
        parallel_cfg.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
    )

    return metadata, config


def extract_mm_features(
    request: Union["Request", "NewRequestData"], modify: bool = False
) -> tuple[list[str], list["PlaceholderRange"]]:
    """
    Normalize multimodal information from a Request into parallel lists.

    This helper reads either:
      1) `request.mm_features` (objects each exposing `.identifier` and
      `.mm_position`), or
      2) legacy fields `request.mm_hashes` and `request.mm_positions`.

    It returns two equally sized lists: the multimodal hash identifiers and
    their corresponding positions. If the request contains no multimodal info,
    it returns `([], [])`.

    Args:
        request (Request): The source object.
        modify (bool):
            Controls copy semantics for the legacy-path return values.
            - If True and legacy fields are used, shallow-copies are returned so
              the caller can mutate the lists without affecting `request`.
            - If False, the original legacy sequences are returned as-is
              (zero-copy); treat them as read-only.

    Returns:
        tuple[list[str], list[PlaceholderRange]]: (`mm_hashes`, `mm_positions`).
        May be `([], [])` when no multimodal data is present.
    """
    if getattr(request, "mm_features", None):
        mm_hashes, mm_positions = zip(
            *((f.identifier, f.mm_position) for f in request.mm_features)
        )
        return (list(mm_hashes), list(mm_positions))
    elif getattr(request, "mm_hashes", None):
        if modify:
            return (
                request.mm_hashes.copy(),  # type: ignore
                request.mm_positions.copy(),  # type: ignore
            )
        else:
            return (request.mm_hashes, request.mm_positions)  # type: ignore
    else:
        return ([], [])
