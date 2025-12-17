# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook for intercepting KV tensors during decode phase."""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_dump.accumulator import get_accumulator_manager
from vllm.v1.kv_cache_dump.config import get_kv_dump_config

if TYPE_CHECKING:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)

# Thread-local storage for request context
_request_context = threading.local()

# Regex to extract layer index from layer name like "model.layers.15.self_attn"
_LAYER_IDX_PATTERN = re.compile(r"layers\.(\d+)")

# Track if we've logged shape info (do it once per layer)
_shape_logged: set[int] = set()


def set_request_context(req_id: str | None):
    """
    Set the current request ID in thread-local storage.

    This should be called before the model forward pass when batch size is 1.

    Args:
        req_id: Request ID or None to clear
    """
    _request_context.req_id = req_id


def _get_request_id() -> str | None:
    """Get the current request ID from thread-local storage."""
    return getattr(_request_context, "req_id", None)


def _parse_layer_idx(layer_name: str) -> int | None:
    """
    Parse layer index from layer name.

    Args:
        layer_name: Layer name like "model.layers.15.self_attn"

    Returns:
        Layer index or None if not found
    """
    match = _LAYER_IDX_PATTERN.search(layer_name)
    if match:
        return int(match.group(1))
    return None


def maybe_dump_kv_decode(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
    attn_metadata: FlashAttentionMetadata,
) -> None:
    """
    Hook to intercept and accumulate KV tensors during decode phase.

    This function is called BEFORE reshape_and_cache in the attention backend.
    It checks if dumping is enabled and if the current phase is decode,
    then accumulates the KV tensors for later saving.

    Args:
        key: Key tensor [num_tokens, num_kv_heads, head_size]
        value: Value tensor [num_tokens, num_kv_heads, head_size]
        layer_name: Name of the attention layer (e.g., "model.layers.15.self_attn")
        attn_metadata: Attention metadata containing phase information
    """
    config = get_kv_dump_config()

    # Fast path: return immediately if not enabled
    if not config.enabled:
        return

    # Only dump during decode phase (max_query_len == 1)
    if attn_metadata.max_query_len != 1:
        return

    # Parse layer index from layer name
    layer_idx = _parse_layer_idx(layer_name)
    if layer_idx is None:
        return

    # Check if this layer should be dumped
    if not config.should_dump_layer(layer_idx):
        return

    # Get current request ID
    req_id = _get_request_id()
    if req_id is None:
        # Debug: log once when req_id is None during decode
        if not hasattr(maybe_dump_kv_decode, "_logged_no_req_id"):
            logger.warning(
                f"[KV Dump] req_id is None during decode for layer {layer_idx}. "
                "Request context not set?"
            )
            maybe_dump_kv_decode._logged_no_req_id = True
        return

    # Get the actual number of tokens (may be padded)
    num_actual_tokens = attn_metadata.num_actual_tokens

    # Slice to actual tokens
    key_actual = key[:num_actual_tokens]
    value_actual = value[:num_actual_tokens]

    # === VALIDATION 1: Verify T == 1 for decode ===
    T = key_actual.shape[0]
    if T != 1:
        logger.warning(
            f"[KV Dump] Expected T=1 during decode, got T={T} "
            f"for layer {layer_idx}, request {req_id}. Skipping this step."
        )
        return

    # === VALIDATION 2: Verify 3D shape (T, H_kv, d) - not GQA-expanded ===
    if key_actual.ndim != 3 or value_actual.ndim != 3:
        logger.error(
            f"[KV Dump] Expected 3D tensors (T, H_kv, d), got "
            f"key.ndim={key_actual.ndim}, value.ndim={value_actual.ndim}. "
            f"This may indicate GQA-expanded tensors. Skipping."
        )
        return

    # Validate shape matches config expectations
    _, H_kv, d = key_actual.shape
    if config.num_kv_heads > 0 and H_kv != config.num_kv_heads:
        logger.warning(
            f"[KV Dump] H_kv mismatch: got {H_kv}, expected {config.num_kv_heads}"
        )
    if config.head_size > 0 and d != config.head_size:
        logger.warning(
            f"[KV Dump] head_size mismatch: got {d}, expected {config.head_size}"
        )

    # Log shape info once per layer
    if layer_idx not in _shape_logged:
        logger.info(
            f"[KV Dump] Layer {layer_idx}: "
            f"key shape={tuple(key_actual.shape)}, "
            f"value shape={tuple(value_actual.shape)}, "
            f"dtype={key_actual.dtype}"
        )
        _shape_logged.add(layer_idx)

    # === VALIDATION 3: Check max tokens cap ===
    manager = get_accumulator_manager()
    if not manager.has_request(req_id):
        # Debug: log once when request not found
        if not hasattr(maybe_dump_kv_decode, "_logged_no_request"):
            maybe_dump_kv_decode._logged_no_request = set()
        if req_id not in maybe_dump_kv_decode._logged_no_request:
            logger.warning(
                f"[KV Dump] Request {req_id} not found in accumulator manager. "
                "Request not started?"
            )
            maybe_dump_kv_decode._logged_no_request.add(req_id)
        return

    current_tokens = manager.get_decode_token_count(req_id)
    if config.max_decode_tokens > 0 and current_tokens >= config.max_decode_tokens:
        # Already at cap, skip
        return

    # Clone to CPU to avoid affecting GPU memory
    key_cpu = key_actual.detach().cpu().clone()
    value_cpu = value_actual.detach().cpu().clone()

    # Accumulate
    manager.accumulate(req_id, layer_idx, key_cpu, value_cpu)
