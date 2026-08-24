# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM-side routing and topology rules for LMCache MP deployments."""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

LMCacheMPDeployment = Literal["local", "external"]


def get_lmcache_mp_deployment(
    kv_transfer_config: "KVTransferConfig",
) -> LMCacheMPDeployment | None:
    """Return the explicit MP deployment for the LMCache MP connector."""
    if kv_transfer_config.kv_connector != "LMCacheMPConnector":
        return None
    deployment = kv_transfer_config.get_from_extra_config(
        "lmcache.mp.deployment", "external"
    )
    if isinstance(deployment, str):
        deployment = deployment.strip().lower()
    if deployment not in {"local", "external"}:
        raise ValueError(
            f"lmcache.mp.deployment must be 'local' or 'external', got {deployment!r}"
        )
    return deployment


def validate_lmcache_mp_local_topology(
    parallel_config: "ParallelConfig",
) -> None:
    """Validate the topology implemented by the embedded local facade."""
    executor_backend = parallel_config.distributed_executor_backend
    if parallel_config.nnodes != 1 or executor_backend not in {"uni", "mp"}:
        raise ValueError(
            "LMCache local MP deployment requires a single-machine vLLM "
            "'uni' or 'mp' executor, but distributed_executor_backend="
            f"{executor_backend!r} and nnodes={parallel_config.nnodes}. "
            "Select lmcache.mp.deployment='external' for a cross-machine "
            "or externally managed deployment."
        )

    pp_size = parallel_config.pipeline_parallel_size
    dp_size = parallel_config.data_parallel_size
    if pp_size != 1 or dp_size != 1:
        raise ValueError(
            "LMCache local MP deployment supports tensor parallelism on one "
            "machine, but pipeline_parallel_size and data_parallel_size must "
            f"both be 1; got TP={parallel_config.tensor_parallel_size}, "
            f"PP={pp_size}, DP={dp_size}. Select "
            "lmcache.mp.deployment='external' for this topology."
        )


def validate_lmcache_mp_hma_block_sizes(
    kv_cache_config: "KVCacheConfig | None",
) -> None:
    """Reject HMA layouts the LMCache MP recovery path cannot represent.

    The current scheduler recovery code uses one block size for all group
    block tables. LMCache MP therefore only supports HMA configurations whose
    groups use the same block size. This is intentionally an LMCache-specific
    guard; other connectors retain their existing behavior.
    """
    if kv_cache_config is None:
        return

    block_sizes = {
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    }
    if len(block_sizes) > 1:
        sizes = ", ".join(str(size) for size in sorted(block_sizes))
        raise ValueError(
            "LMCache MP HMA currently requires all KV cache groups to use the "
            f"same block size; found [{sizes}]. Disable HMA or use a model "
            "with uniform KV block sizes."
        )
