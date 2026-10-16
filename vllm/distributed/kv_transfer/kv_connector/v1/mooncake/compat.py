# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility factors for the Mooncake P2P connectors.

The push and pull connectors hand each other KV cache bytes, so they must agree
on the model, the KV cache format, and the protocol. Both sides hash those
factors at startup and compare the digests during the handshake, which turns a
mismatched deployment into one failed request with a reason instead of a
transfer neither side can interpret.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Increment this whenever an incompatible change lands in either connector's
# handshake or wire format, so mismatched peers fail the handshake instead of
# transferring bytes neither side can read.
#
# History:
#   1: pull-mode handshake (address book, block ids, completion report)
MOONCAKE_CONNECTOR_VERSION: int = 1


def get_speculative_compatibility_factors(
    vllm_config: "VllmConfig",
) -> dict[str, Any] | None:
    """Compatibility factors of a speculator that moves KV state between peers."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_eagle():
        return None

    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None
    auxiliary_layer_ids = getattr(
        draft_model_config.hf_config,
        "eagle_aux_hidden_state_layer_ids",
        None,
    )
    # kv_cache_dtype defaults to None, meaning "inherit from the target model".
    # Resolve it so an explicit setting on one side and inheritance on the other
    # do not look like a mismatch.
    kv_cache_dtype = (
        speculative_config.kv_cache_dtype or vllm_config.cache_config.cache_dtype
    )
    return {
        "method": speculative_config.method,
        "model": draft_model_config.model,
        "revision": draft_model_config.revision,
        "code_revision": draft_model_config.code_revision,
        "parallel_drafting": speculative_config.parallel_drafting,
        "kv_cache_dtype": str(kv_cache_dtype),
        "auxiliary_layer_ids": (
            tuple(auxiliary_layer_ids) if auxiliary_layer_ids is not None else None
        ),
    }


def compute_mooncake_compatibility_hash(
    vllm_config: "VllmConfig",
    attn_backend_names: Sequence[str],
    transfer_mode: str = "pull",
) -> str:
    """Hash the factors that decide whether two Mooncake peers can transfer KV.

    Factors included:
    - vLLM version and Mooncake connector version
    - The weights that were loaded (name, revision, code revision,
      quantization) so two checkpoints of one model path are not confused
    - Model architecture (dtype, KV heads, head size, layers)
    - KV cache format (dtype, hybrid KV cache manager)
    - Every attention backend holding a KV cache, not just the first one
    - EAGLE/MTP configuration that affects transferred state
    - Transfer mode (pull vs push)

    The transfer mode is included because the pull (READ) and push (WRITE)
    connectors use incompatible transfer protocols; a pull connector and a push
    connector must never complete a handshake with each other.

    Factors like tensor_parallel_size, block_size and kv_cache_layout are
    validated against the peer's address book at runtime and stay out of the
    hash to support heterogeneous deployments.

    Args:
        vllm_config: Configuration of this instance.
        attn_backend_names: Names of every attention backend holding a cache in
            this instance. The whole set participates, so hybrid models whose
            backend sets differ are refused even when the first backend agrees.
        transfer_mode: "pull" or "push".

    Returns:
        SHA-256 hex digest of the factors above.

    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    factors = {
        "vllm_version": vllm_version,
        "mooncake_connector_version": MOONCAKE_CONNECTOR_VERSION,
        "model": model_config.model,
        "revision": model_config.revision,
        "code_revision": model_config.code_revision,
        "quantization": model_config.quantization,
        "quantization_config": model_config.quantization_config,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "attn_backend_names": tuple(sorted(set(attn_backend_names))),
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
        ),
        "speculative_config": get_speculative_compatibility_factors(vllm_config),
        "transfer_mode": transfer_mode,
    }
    compat_hash = hash_factors(factors)
    logger.debug(
        "Mooncake compatibility hash: %s (model=%s revision=%s, dtype=%s, "
        "num_kv_heads=%d, cache_dtype=%s, attn_backends=%s, transfer_mode=%s)",
        compat_hash,
        factors["model"],
        factors["revision"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        factors["attn_backend_names"],
        transfer_mode,
    )
    return compat_hash
