# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned capture sources; persistence is owned by AuxOutput."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch

from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

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


def capture_indexer_topk(
    sink: IndexerCaptureSink | None,
    topk_indices_buffer: torch.Tensor | None,
    num_tokens: int,
) -> None:
    if sink is None:
        return
    assert topk_indices_buffer is not None
    # DBO ubatches share the parent batch's capture rows, so each ubatch writes
    # at its own offset instead of overwriting row 0.
    sink.capture(
        topk_indices_buffer[:num_tokens, : sink.rows.shape[1]],
        token_start=dbo_current_ubatch_token_offset(),
    )


class IndexerCaptureSource:
    """Explicit capability for a logical TopK producer's execution path."""

    indexer_capture: IndexerCaptureSink | None = None

    def get_indexer_capture_metadata(self) -> tuple[int, int] | None:
        """Return (layer_id, topk), or None for a shared-index consumer."""
        raise NotImplementedError

    def set_indexer_capture_sink(self, sink: IndexerCaptureSink | None) -> None:
        self.indexer_capture = sink


def _get_indexer_capture_sources(
    model: torch.nn.Module, num_layers: int
) -> list[tuple[int, int, IndexerCaptureSource]]:
    sources: list[tuple[int, int, IndexerCaptureSource]] = []
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
        sources = _get_indexer_capture_sources(
            model, config.model_config.get_total_num_hidden_layers()
        )
        for layer_id, _, source in sources:
            if source.indexer_capture is not None:
                raise ValueError(
                    f"Indexer capture source layer {layer_id} is already bound"
                )
        layer_ids = tuple(layer_id for layer_id, _, _ in sources)
        topk = sources[0][1]
        self.shape_per_token = (len(layer_ids), topk)
        self.fixed_bytes_per_token = len(layer_ids) * topk * 4
        self._bindings: list[tuple[IndexerCaptureSource, IndexerCaptureSink]] = []
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
        self._output_rank = output_rank
        self.device_buffer = torch.empty(
            (
                config.scheduler_config.max_num_batched_tokens,
                *self.shape_per_token,
            ),
            dtype=torch.int32,
            device=config.device_config.device,
        )

        try:
            for column, (_, _, source) in enumerate(sources):
                sink = IndexerCaptureSink(self.device_buffer[:, column])
                self._bindings.append((source, sink))
                source.set_indexer_capture_sink(sink)
        except BaseException:
            self.close()
            raise

    def snapshot(self, num_rows: int) -> torch.Tensor:
        """Copy the forwarded rows into a snapshot owned by the caller."""
        assert self._output_rank, "Indexer snapshots require the output rank"
        assert 0 <= num_rows <= len(self.device_buffer)
        return self.device_buffer[:num_rows].clone()

    def close(self) -> None:
        """Unbind this capturer's sinks and release its buffers."""
        if self._closed:
            return
        for source, sink in reversed(self._bindings):
            if source.indexer_capture is sink:
                source.set_indexer_capture_sink(None)
        self._bindings.clear()
        del self.device_buffer
        self._closed = True
