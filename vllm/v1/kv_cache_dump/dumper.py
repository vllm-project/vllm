# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dumper for saving accumulated KV tensors to safetensors format."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_dump.accumulator import RequestKVAccumulator
    from vllm.v1.kv_cache_dump.config import KVCacheDumpConfig

logger = init_logger(__name__)


def _get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def save_request_kv(
    acc: RequestKVAccumulator,
    config: KVCacheDumpConfig,
) -> str | None:
    """
    Save accumulated KV tensors for a request to a safetensors file.

    Args:
        acc: The request KV accumulator
        config: KV dump configuration

    Returns:
        Path to the saved file, or None if save failed
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.error(
            "safetensors not installed. Please install it with: pip install safetensors"
        )
        return None

    if not acc.layer_data:
        logger.warning(f"No KV data to save for request {acc.req_id}")
        return None

    # Build tensors dict
    tensors: dict[str, torch.Tensor] = {}
    num_decode_tokens = acc.num_decode_tokens

    for layer_idx in sorted(acc.layer_data.keys()):
        kv = acc.get_accumulated_kv(layer_idx)
        if kv is None:
            continue

        K, V = kv
        # Tensors are [S, H_kv, d] where S = number of decode tokens

        # === VALIDATION: Check S matches across layers and token_ids ===
        S_k, S_v = K.shape[0], V.shape[0]
        if S_k != S_v:
            logger.error(
                f"[KV Dump] K/V shape mismatch for layer {layer_idx}: "
                f"K.shape[0]={S_k}, V.shape[0]={S_v}"
            )
        if S_k != num_decode_tokens:
            logger.warning(
                f"[KV Dump] S mismatch for layer {layer_idx}: "
                f"K.shape[0]={S_k}, len(token_ids)={num_decode_tokens}. "
                f"Possible missing decode steps or token tracking issue."
            )

        tensors[f"K_layer{layer_idx}"] = K
        tensors[f"V_layer{layer_idx}"] = V

    # Add token IDs
    if acc.decode_token_ids:
        tensors["token_ids"] = torch.tensor(acc.decode_token_ids, dtype=torch.int32)
    else:
        # If no decode tokens tracked, create empty tensor
        tensors["token_ids"] = torch.tensor([], dtype=torch.int32)

    # Add prompt token count
    tensors["prompt_token_count"] = torch.tensor(
        [acc.num_prompt_tokens], dtype=torch.int32
    )

    # Build metadata
    metadata = {
        "model": config.model_name,
        "layer_ids": str(sorted(acc.layer_data.keys())),
        "H_q": str(config.num_q_heads),
        "H_kv": str(config.num_kv_heads),
        "d": str(config.head_size),
        "kv_dtype": str(config.dtype),
        "phase": "decode-only",
        "prompt_id": acc.req_id,
        "num_decode_tokens": str(acc.num_decode_tokens),
        "num_prompt_tokens": str(acc.num_prompt_tokens),
    }

    # Generate filename
    timestamp = _get_timestamp_ms()
    # Sanitize request ID for filename (replace special chars)
    safe_req_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in acc.req_id)
    filename = f"{safe_req_id}_{timestamp}.safetensors"
    filepath = Path(config.output_dir) / filename

    # Save
    try:
        save_file(tensors, str(filepath), metadata=metadata)
        logger.info(
            f"Saved KV dump for request {acc.req_id}: "
            f"{acc.num_decode_tokens} decode tokens, "
            f"{len(acc.layer_data)} layers -> {filepath}"
        )
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save KV dump for request {acc.req_id}: {e}")
        return None
