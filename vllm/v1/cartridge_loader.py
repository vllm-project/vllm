# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cartridge loader for loading pre-computed KV cache data."""

import logging
from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.utils.cartridge_manager import get_cartridge_manager

logger = init_logger(__name__)


class CartridgeData:
    """
    Container for loaded cartridge data.

    Expected cartridge format (.pt file):
    {
        'token_ids': torch.Tensor,  # Shape: (num_tokens,)
        'kv_cache': dict[str, torch.Tensor],  # Per-layer KV cache tensors
        'metadata': dict,  # Optional metadata (model info, etc.)
    }

    Or simplified format:
    {
        'token_ids': list[int] or torch.Tensor,
        'num_tokens': int,
    }
    """

    def __init__(
        self,
        token_ids: torch.Tensor | list[int],
        kv_cache: Optional[dict[str, torch.Tensor]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.token_ids = token_ids
        self.kv_cache = kv_cache or {}
        self.metadata = metadata or {}
        self.num_tokens = len(token_ids)

    @classmethod
    def from_dict(cls, data: dict) -> "CartridgeData":
        """Load CartridgeData from a dictionary (loaded .pt file)."""
        token_ids = data.get("token_ids")
        if token_ids is None:
            raise ValueError("Cartridge must contain 'token_ids' field")

        kv_cache = data.get("kv_cache")
        metadata = data.get("metadata")

        return cls(token_ids=token_ids, kv_cache=kv_cache, metadata=metadata)

    def __repr__(self) -> str:
        return (
            f"CartridgeData(num_tokens={self.num_tokens}, "
            f"has_kv_cache={bool(self.kv_cache)}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


def load_cartridge(
    cartridge_id: str,
    source: str = "s3",
    force_redownload: bool = False,
) -> CartridgeData:
    """
    Load a KV cache cartridge.

    Args:
        cartridge_id: The identifier/path of the cartridge
        source: Source type ('s3' or 'local')
        force_redownload: If True, re-download even if cached

    Returns:
        CartridgeData containing the loaded cartridge

    Raises:
        ValueError: If cartridge format is invalid
        RuntimeError: If loading fails
    """
    logger.info(
        f"Loading cartridge: {cartridge_id} (source={source}, "
        f"force_redownload={force_redownload})"
    )

    # Get the cartridge manager and download/load the cartridge
    manager = get_cartridge_manager()
    cartridge_tensor = manager.get_cartridge(
        cartridge_id=cartridge_id,
        source=source,
        force_redownload=force_redownload,
    )

    # Parse the cartridge data
    if isinstance(cartridge_tensor, dict):
        cartridge_data = CartridgeData.from_dict(cartridge_tensor)
    else:
        raise ValueError(
            f"Invalid cartridge format. Expected dict, got {type(cartridge_tensor)}"
        )

    logger.info(f"Successfully loaded cartridge: {cartridge_data}")
    return cartridge_data


def load_cartridges_from_request(
    cartridges_spec: list[dict[str, Any]],
) -> list[CartridgeData]:
    """
    Load multiple cartridges from a request specification.

    Args:
        cartridges_spec: List of cartridge specifications from the request.
            Each spec should have: id, source, force_redownload

    Returns:
        List of loaded CartridgeData objects

    Raises:
        RuntimeError: If any cartridge fails to load
    """
    loaded_cartridges = []

    for spec in cartridges_spec:
        cartridge_id = spec.get("id")
        source = spec.get("source", "s3")
        force_redownload = spec.get("force_redownload", False)

        if not cartridge_id:
            logger.warning(f"Skipping cartridge with missing id: {spec}")
            continue

        try:
            cartridge_data = load_cartridge(
                cartridge_id=cartridge_id,
                source=source,
                force_redownload=force_redownload,
            )
            loaded_cartridges.append(cartridge_data)
        except Exception as e:
            logger.error(f"Failed to load cartridge {cartridge_id}: {e}")
            raise RuntimeError(f"Failed to load cartridge {cartridge_id}: {e}") from e

    return loaded_cartridges
