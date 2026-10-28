# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing.synchronize import Lock as LockType
from typing import Literal

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .base import (
    BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache,
    MultiModalProcessorOnlyCache,
)
from .lru import (
    LruKeyReplicatedReceiverCache,
    LruKeyReplicatedSenderCache,
)
from .shm import (
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
)

logger = init_logger(__name__)


def _get_cache_type(
    vllm_config: VllmConfig,
) -> Literal[None, "processor_only", "lru", "shm"]:
    model_config = vllm_config.model_config
    if not model_config.supports_multimodal_inputs:
        return None

    # Check if the cache is disabled.
    mm_config = model_config.get_multimodal_config()
    if mm_config.mm_processor_cache_gb <= 0:
        return None

    # cuda_ipc_pool items reference pool slices that are recycled once the
    # workers release them, so they must not be cached across requests.
    if mm_config.mm_tensor_ipc == "cuda_ipc_pool":
        logger.info_once(
            "Disabling the multi-modal processor cache: "
            "mm_tensor_ipc='cuda_ipc_pool' items cannot outlive a request."
        )
        return None

    # Check if IPC caching is supported.
    parallel_config = vllm_config.parallel_config
    is_ipc_supported = parallel_config._api_process_count == 1 and (
        parallel_config.data_parallel_size == 1
        or parallel_config.data_parallel_external_lb
    )

    if not is_ipc_supported:
        return "processor_only"

    mm_config = model_config.get_multimodal_config()
    return mm_config.mm_processor_cache_type


def processor_cache_from_config(
    vllm_config: VllmConfig,
) -> BaseMultiModalProcessorCache | None:
    """Return a `BaseMultiModalProcessorCache`, if enabled."""
    cache_type = _get_cache_type(vllm_config)
    if cache_type is None:
        return None
    elif cache_type == "processor_only":
        return MultiModalProcessorOnlyCache(vllm_config.model_config)
    elif cache_type == "lru":
        return LruKeyReplicatedSenderCache(vllm_config.model_config)
    elif cache_type == "shm":
        return ShmObjectStoreSenderCache(vllm_config)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")


def processor_only_cache_from_config(
    vllm_config: VllmConfig,
) -> MultiModalProcessorOnlyCache | None:
    """Return a `MultiModalProcessorOnlyCache`, if enabled."""
    cache_type = _get_cache_type(vllm_config)
    if cache_type is None:
        return None

    return MultiModalProcessorOnlyCache(vllm_config.model_config)


def engine_receiver_cache_from_config(
    vllm_config: VllmConfig,
) -> BaseMultiModalReceiverCache | None:
    """Return a `BaseMultiModalReceiverCache` for the engine process."""
    cache_type = _get_cache_type(vllm_config)
    if cache_type in (None, "processor_only", "shm"):
        return None
    elif cache_type == "lru":
        return LruKeyReplicatedReceiverCache(vllm_config.model_config)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")


def worker_receiver_cache_from_config(
    vllm_config: VllmConfig,
    shared_worker_lock: LockType | None,
) -> BaseMultiModalReceiverCache | None:
    """Return a `BaseMultiModalReceiverCache` for the worker process."""
    cache_type = _get_cache_type(vllm_config)
    if cache_type in (None, "processor_only", "lru"):
        return None
    elif cache_type == "shm":
        if shared_worker_lock is None:
            raise ValueError(
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

        return ShmObjectStoreReceiverCache(vllm_config, shared_worker_lock)
    else:
        raise ValueError(f"Unknown cache type: {cache_type!r}")
