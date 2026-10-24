# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned capture sources; persistence is owned by AuxOutput."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class IndexerCaptureSink:
    def __init__(self, rows: torch.Tensor) -> None:
        self.rows = rows

    def capture(self, indices: torch.Tensor, token_start: int = 0) -> None:
        if indices.ndim != 2 or indices.shape[1] != self.rows.shape[1]:
            raise ValueError("Indexer selection width changed")
        end = token_start + len(indices)
        if token_start < 0 or end > len(self.rows):
            raise ValueError("Indexer selection exceeds capture token capacity")
        if not len(indices):
            return
        self.rows[token_start:end].copy_(indices)


class IndexerCaptureSource:
    """Explicit capability for a logical TopK producer's execution path."""

    indexer_capture: IndexerCaptureSink | None = None

    def get_indexer_capture_metadata(self) -> tuple[int, int] | None:
        """Return (layer_id, topk), or None for a shared-index consumer."""
        raise NotImplementedError

    def set_indexer_capture_sink(self, sink: IndexerCaptureSink | None) -> None:
        self.indexer_capture = sink


@dataclass
class IndexerSnapshot:
    rows: torch.Tensor
    slot: int


def _get_indexer_capture_sources(
    model: torch.nn.Module, num_layers: int
) -> list[tuple[int, int, IndexerCaptureSource]]:
    sources = []
    for module in model.modules():
        if not isinstance(module, IndexerCaptureSource):
            continue
        metadata = module.get_indexer_capture_metadata()
        if metadata is None:
            continue
        layer_id, topk = metadata
        if layer_id >= num_layers:
            continue
        sources.append((layer_id, topk, module))
    sources.sort(key=lambda entry: entry[0])
    if not sources:
        raise ValueError(
            "No backbone indexer capture sources: this model/backend "
            "does not support ordinary indexer replay"
        )
    layer_ids = tuple(layer_id for layer_id, _, _ in sources)
    topks = {topk for _, topk, _ in sources}
    if (
        min(layer_ids) < 0
        or len(set(layer_ids)) != len(layer_ids)
        or len(topks) != 1
        or min(topks) <= 0
    ):
        raise ValueError(
            "Ordinary indexer replay requires unique layers and uniform positive topk"
        )
    return sources


class IndexerCapturer:
    """Own raw storage and snapshots for the discovered source layout."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: VllmConfig,
        *,
        output_rank: bool = True,
    ) -> None:
        self._num_layers = config.model_config.get_total_num_hidden_layers()
        sources = _get_indexer_capture_sources(model, self._num_layers)
        layer_ids = tuple(layer_id for layer_id, _, _ in sources)
        topk = sources[0][1]
        self.source_columns = {
            layer_id: column for column, layer_id in enumerate(layer_ids)
        }
        self.shape_per_token = (len(layer_ids), topk)
        self.fixed_bytes_per_token = len(layer_ids) * topk * 4
        self._sources: list[IndexerCaptureSource] = []
        self._closed = False
        model_config = config.model_config
        identity = json.dumps(
            {
                "format": "indexer-int32-v1",
                "layer_ids": layer_ids,
                "topk": topk,
                "model": model_config.model,
                "revision": model_config.revision,
                "commit": getattr(model_config.hf_config, "_commit_hash", None),
                "code_revision": model_config.code_revision,
                "config": model_config.hf_text_config.to_dict(),
                "dtype": str(model_config.dtype),
                "quantization": model_config.quantization,
            },
            sort_keys=True,
            default=str,
        )
        self.namespace = hashlib.sha256(identity.encode()).hexdigest()
        self.device_buffer = torch.empty(
            (
                config.scheduler_config.max_num_batched_tokens,
                *self.shape_per_token,
            ),
            dtype=torch.int32,
            device=config.device_config.device,
        )
        num_slots = config.max_concurrent_batches if output_rank else 0
        self._snapshot_rows = [
            torch.empty_like(self.device_buffer) for _ in range(num_slots)
        ]
        self._free_snapshot_slots = list(range(num_slots))

    def snapshot(self, num_rows: int) -> IndexerSnapshot:
        assert 0 <= num_rows <= len(self.device_buffer)
        assert self._free_snapshot_slots, "Indexer snapshots were not consumed in order"
        slot = self._free_snapshot_slots.pop()
        rows = self._snapshot_rows[slot][:num_rows]
        rows.copy_(self.device_buffer[:num_rows])
        return IndexerSnapshot(rows, slot)

    def release_snapshot(self, slot: int) -> None:
        assert (
            0 <= slot < len(self._snapshot_rows)
            and slot not in self._free_snapshot_slots
        )
        self._free_snapshot_slots.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        for source in self._sources:
            source.set_indexer_capture_sink(None)
        self._sources.clear()
        self._snapshot_rows.clear()
        self._free_snapshot_slots.clear()
        del self.device_buffer
        self._closed = True


def bind_indexer_capturer(
    model: torch.nn.Module,
    capturer: IndexerCapturer,
) -> None:
    """Attach sinks to the model's logical TopK producers."""
    if capturer._closed:
        raise ValueError("Cannot bind a closed Indexer capturer")
    sources = _get_indexer_capture_sources(model, capturer._num_layers)
    expected = [
        (layer_id, capturer.shape_per_token[1]) for layer_id in capturer.source_columns
    ]
    if [(layer_id, topk) for layer_id, topk, _ in sources] != expected:
        raise ValueError("Indexer capture source layout changed after allocation")
    for layer_id, _, source in sources:
        if source.indexer_capture is not None:
            raise ValueError(
                f"Indexer capture source layer {layer_id} is already bound"
            )
    try:
        for layer_id, _, source in sources:
            column = capturer.source_columns[layer_id]
            capturer._sources.append(source)
            source.set_indexer_capture_sink(
                IndexerCaptureSink(capturer.device_buffer[:, column])
            )
    except Exception:
        capturer.close()
        raise
