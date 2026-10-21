# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned capture sources; persistence is owned by AuxOutput."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class IndexerCaptureSink:
    def __init__(self, rows: torch.Tensor, valid: torch.Tensor) -> None:
        self.rows = rows
        self.valid = valid

    def capture(self, indices: torch.Tensor, token_start: int = 0) -> None:
        if indices.ndim != 2 or indices.shape[1] != self.rows.shape[1]:
            raise ValueError("Indexer selection width changed")
        end = token_start + len(indices)
        if token_start < 0 or end > len(self.rows):
            raise ValueError("Indexer selection exceeds capture token capacity")
        if not len(indices):
            return
        self.rows[token_start:end].copy_(indices)
        self.valid[token_start:end].fill_(True)


class IndexerCaptureSource(Protocol):
    indexer_capture_layer_id: int
    indexer_capture_topk: int
    indexer_capture: IndexerCaptureSink | None


@dataclass
class IndexerSnapshot:
    rows: torch.Tensor
    valid: torch.Tensor
    slot: int


class IndexerCapturer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: VllmConfig,
        *,
        output_rank: bool = True,
    ) -> None:
        sources: list[IndexerCaptureSource] = []
        for module in model.modules():
            if hasattr(module, "indexer_capture_layer_id"):
                sources.append(cast(IndexerCaptureSource, module))
        sources.sort(key=lambda source: source.indexer_capture_layer_id)
        if not sources:
            raise ValueError(
                "No backbone indexer capture sources: this model/backend "
                "does not support ordinary indexer replay"
            )
        layer_ids = tuple(source.indexer_capture_layer_id for source in sources)
        topks = {source.indexer_capture_topk for source in sources}
        if len(set(layer_ids)) != len(layer_ids) or len(topks) != 1 or min(topks) <= 0:
            raise ValueError(
                "Ordinary indexer replay requires unique layers and uniform topk"
            )
        topk = topks.pop()
        self.shape_per_token = (len(sources), topk)
        self.fixed_bytes_per_token = len(sources) * topk * 4
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
            device=current_platform.device_type,
        )
        self.valid = torch.zeros(
            (len(self.device_buffer), len(sources)),
            dtype=torch.bool,
            device=self.device_buffer.device,
        )
        num_slots = config.max_concurrent_batches if output_rank else 0
        self._snapshot_rows = [
            torch.empty_like(self.device_buffer) for _ in range(num_slots)
        ]
        self._snapshot_valid = [torch.empty_like(self.valid) for _ in range(num_slots)]
        self._free_snapshot_slots = list(range(num_slots))
        self._sources = sources
        for column, source in enumerate(sources):
            source.indexer_capture = IndexerCaptureSink(
                self.device_buffer[:, column], self.valid[:, column]
            )

    def begin_step(self) -> None:
        self.valid.zero_()

    def snapshot(self, num_rows: int) -> IndexerSnapshot:
        assert 0 <= num_rows <= len(self.device_buffer)
        assert self._free_snapshot_slots, "Indexer snapshots were not consumed in order"
        slot = self._free_snapshot_slots.pop()
        rows = self._snapshot_rows[slot][:num_rows]
        valid = self._snapshot_valid[slot][:num_rows]
        rows.copy_(self.device_buffer[:num_rows])
        valid.copy_(self.valid[:num_rows])
        return IndexerSnapshot(rows, valid, slot)

    def release_snapshot(self, slot: int) -> None:
        assert (
            0 <= slot < len(self._snapshot_rows)
            and slot not in self._free_snapshot_slots
        )
        self._free_snapshot_slots.append(slot)

    def close(self) -> None:
        if not hasattr(self, "device_buffer"):
            return
        for source in self._sources:
            source.indexer_capture = None
        self._sources.clear()
        self._snapshot_rows.clear()
        self._snapshot_valid.clear()
        self._free_snapshot_slots.clear()
        del self.device_buffer
        del self.valid


def configure_indexer_capture(
    module: torch.nn.Module,
    config: VllmConfig,
    layer_id: int,
    topk: int,
) -> None:
    capture_source = cast(IndexerCaptureSource, module)
    capture_source.indexer_capture = None
    if (
        config.aux_output_config.enable_return_indexer_topk
        and layer_id < config.model_config.get_total_num_hidden_layers()
    ):
        capture_source.indexer_capture_layer_id = layer_id
        capture_source.indexer_capture_topk = topk
