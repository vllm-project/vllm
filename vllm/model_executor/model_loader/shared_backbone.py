# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime registry for shared backbone tensors."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config.load import SharedBackboneConfig


class SharedBackboneRegistry:
    """Caches tensors belonging to a shared backbone checkpoint."""

    _lock = threading.Lock()
    _cache: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @classmethod
    def register(
        cls, backbone_id: str | None, tensor_name: str, tensor: torch.Tensor
    ) -> None:
        if backbone_id is None:
            return
        with cls._lock:
            cls._cache[backbone_id][tensor_name] = tensor

    @classmethod
    def get(cls, backbone_id: str | None, tensor_name: str) -> torch.Tensor | None:
        if backbone_id is None:
            return None
        with cls._lock:
            return cls._cache.get(backbone_id, {}).get(tensor_name)

    @classmethod
    def tensors_for(cls, backbone_id: str | None) -> Mapping[str, torch.Tensor]:
        if backbone_id is None:
            return {}
        with cls._lock:
            return dict(cls._cache.get(backbone_id, {}))

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def get_stats(cls) -> dict[str, int]:
        with cls._lock:
            return {
                backbone_id: len(tensors) for backbone_id, tensors in cls._cache.items()
            }


def wrap_shared_backbone(
    iterator: Generator[tuple[str, torch.Tensor], None, None],
    model_name: str,
    shared_cfg: SharedBackboneConfig | None,
    logger: logging.Logger,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Apply shared-backbone caching/reuse to a weight iterator."""

    if shared_cfg is None or not shared_cfg.tensor_prefixes:
        yield from iterator
        return

    logger.info(
        "[SharedBackbone] Enabled for model=%s, backbone_id=%s, prefixes=%s",
        model_name,
        shared_cfg.backbone_key(),
        shared_cfg.tensor_prefixes,
    )

    registered_count = 0
    reused_count = 0
    registered_mb = 0.0
    reused_mb = 0.0

    for tensor_name, tensor in iterator:
        original_ptr = tensor.data_ptr()
        result_tensor = _maybe_share_tensor(
            shared_cfg, model_name, tensor_name, tensor, logger
        )

        if result_tensor.data_ptr() != original_ptr:
            reused_count += 1
            reused_mb += _tensor_size_mb(result_tensor)
        elif shared_cfg.matches_tensor(tensor_name) and shared_cfg.is_backbone_model(
            model_name
        ):
            registered_count += 1
            registered_mb += _tensor_size_mb(tensor)

        yield tensor_name, result_tensor

    if registered_count > 0:
        logger.info(
            "[SharedBackbone] SUMMARY: Registered %d tensors (%.2f MB total)",
            registered_count,
            registered_mb,
        )
    if reused_count > 0:
        logger.info(
            "[SharedBackbone] SUMMARY: Reused %d tensors (%.2f MB saved)",
            reused_count,
            reused_mb,
        )


def _tensor_size_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024**2)


def _maybe_share_tensor(
    shared_cfg: SharedBackboneConfig,
    model_name: str,
    tensor_name: str,
    tensor: torch.Tensor,
    logger: logging.Logger,
) -> torch.Tensor:
    if not shared_cfg.matches_tensor(tensor_name):
        return tensor

    backbone_id = shared_cfg.backbone_key()
    if backbone_id is None:
        return tensor

    if shared_cfg.is_backbone_model(model_name):
        SharedBackboneRegistry.register(backbone_id, tensor_name, tensor)
        logger.info(
            "[SharedBackbone] REGISTERED: %s (%.2f MB, shape=%s, dtype=%s)",
            tensor_name,
            _tensor_size_mb(tensor),
            tuple(tensor.shape),
            tensor.dtype,
        )
        return tensor

    cached = SharedBackboneRegistry.get(backbone_id, tensor_name)
    if cached is not None:
        logger.info(
            "[SharedBackbone] REUSED: %s (%.2f MB saved, ptr=%s)",
            tensor_name,
            _tensor_size_mb(cached),
            hex(cached.data_ptr()),
        )
        return cached

    logger.warning(
        "[SharedBackbone] FALLBACK: %s requested for %s before backbone %s was loaded. "
        "Using local checkpoint weights.",
        tensor_name,
        model_name,
        backbone_id,
    )
    return tensor
