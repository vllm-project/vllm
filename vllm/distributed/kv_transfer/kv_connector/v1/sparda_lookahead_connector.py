# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental layer-ahead KV connector for SparDA lookahead prefetch."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class SparDALookaheadMetadata(KVConnectorMetadata):
    """Worker metadata for deterministic layer-ahead prefetch planning."""

    layer_names: tuple[str, ...] = ()
    prefetch_distance: int = 1
    prefetch_plan: dict[str, tuple[int, ...]] = field(default_factory=dict)


class SparDALookaheadConnector(KVConnectorBase_V1, SupportsHMA):
    """Connector shell for experimenting with layer-ahead KV prefetch.

    The first implementation keeps the prefetch selector deterministic and
    metadata-driven. Forecast projection can later populate ``prefetch_plan``
    without changing the per-layer connector control flow.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        self._configured_layer_names = tuple(extra_config.get("layer_names", ()))
        self._prefetch_distance = int(extra_config.get("prefetch_distance", 1))
        if self._prefetch_distance < 1:
            raise ValueError("prefetch_distance must be >= 1")

        self._active_layer_names: tuple[str, ...] = ()
        self._prefetch_plan: dict[str, tuple[int, ...]] = {}
        self._scheduled_prefetches: dict[str, tuple[int, ...]] = {}
        self._scheduled_layers: list[str] = []
        self._pending_layers: set[str] = set()
        self._loaded_layers: set[str] = set()

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return bool(extra_config.get("enable_layer_prefetch", True))

    def bind_connector_metadata(
        self,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, SparDALookaheadMetadata)
        self._active_layer_names = (
            connector_metadata.layer_names or self._configured_layer_names
        )
        self._prefetch_distance = connector_metadata.prefetch_distance
        self._prefetch_plan = dict(connector_metadata.prefetch_plan)
        self._scheduled_prefetches.clear()
        self._scheduled_layers.clear()
        self._pending_layers.clear()
        self._loaded_layers.clear()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self._active_layer_names:
            self._schedule_layer(self._active_layer_names[0], ())

    def wait_for_layer_load(self, layer_name: str) -> None:
        if layer_name in self._pending_layers:
            self._pending_layers.remove(layer_name)
            self._loaded_layers.add(layer_name)
        self._schedule_next_layer(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        return

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return SparDALookaheadMetadata(
            layer_names=self._configured_layer_names,
            prefetch_distance=self._prefetch_distance,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        return ()

    @property
    def scheduled_layers(self) -> tuple[str, ...]:
        return tuple(self._scheduled_layers)

    @property
    def loaded_layers(self) -> frozenset[str]:
        return frozenset(self._loaded_layers)

    @property
    def scheduled_prefetches(self) -> dict[str, tuple[int, ...]]:
        return dict(self._scheduled_prefetches)

    def _schedule_next_layer(self, layer_name: str) -> None:
        next_layer = self._layer_after(layer_name, self._prefetch_distance)
        if next_layer is not None:
            self._schedule_layer(next_layer, self._prefetch_plan.get(layer_name, ()))

    def _layer_after(self, layer_name: str, distance: int) -> str | None:
        try:
            index = self._active_layer_names.index(layer_name)
        except ValueError:
            return None
        next_index = index + distance
        if next_index >= len(self._active_layer_names):
            return None
        return self._active_layer_names[next_index]

    def _schedule_layer(self, layer_name: str, block_ids: tuple[int, ...]) -> None:
        if layer_name in self._loaded_layers or layer_name in self._pending_layers:
            return
        self._pending_layers.add(layer_name)
        self._scheduled_prefetches[layer_name] = block_ids
        self._scheduled_layers.append(layer_name)
