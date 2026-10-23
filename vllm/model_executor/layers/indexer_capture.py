# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned capture sources; persistence is owned by AuxOutput."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from vllm.distributed.aux_output_connector.profile import (
    AuxOutputProfile,
    IndexerSource,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class IndexerCaptureSink:
    def __init__(self, source: IndexerSource, rows: torch.Tensor) -> None:
        self.source = source
        self.rows = rows
        self.region_local: torch.Tensor
        self.region_replicas: int

    def capture(self, indices: torch.Tensor, token_start: int = 0) -> None:
        shape = self.source.shape
        if indices.ndim != len(shape) + 1 or tuple(indices.shape[1:]) != shape:
            raise ValueError(
                f"Indexer selection shape changed: {indices.shape}, {shape}"
            )
        end = token_start + len(indices)
        if token_start < 0 or end > len(self.rows):
            raise ValueError("Indexer selection exceeds capture token capacity")
        if not len(indices):
            return
        self.rows[token_start:end].copy_(indices.reshape(len(indices), -1))

    def capture_pools(self, tokens: torch.Tensor) -> None:
        """Encode expanded pools, including a short causal pool's validity mask."""
        r, topk = self.source.ratio, self.source.topk
        history = tokens[:, :topk].reshape(len(tokens), topk // r, r)
        base = torch.div(history[:, :, 0].to(torch.int64), r, rounding_mode="floor")
        mask = (history >= 0).to(torch.int64)
        bits = 1 << torch.arange(r, device=tokens.device, dtype=torch.int64)
        packed = ((base + 1) << r) | (mask * bits).sum(dim=-1)
        self.rows[: len(tokens), : topk // r].copy_(packed)
        self.rows[: len(tokens), topk // r :].copy_(tokens[:, topk : topk + r - 1])

    def capture_qsa(
        self,
        blocks: torch.Tensor,
        positions: torch.Tensor,
        visible: torch.Tensor,
    ) -> None:
        n, k = blocks.shape
        if k + 2 != self.source.width or n > len(self.rows):
            raise ValueError("QSA capture does not match its profile/capacity")
        self.rows[:n, :k].copy_(blocks)
        self.rows[:n, k].copy_(positions)
        self.rows[:n, k + 1].copy_(visible)

    def capture_regions(
        self,
        regions: torch.Tensor,
        token_start: int = 0,
        group: int | None = None,
    ) -> None:
        groups, width = self.source.shape
        if not len(regions):
            return
        local_groups = self.region_local.shape[1]
        if group is None:
            n = regions.shape[0] // local_groups
            values = regions.reshape(n, local_groups, -1)
            target = self.region_local[token_start : token_start + n]
        else:
            n = len(regions)
            values = regions
            target = self.region_local[token_start : token_start + n, group]
        target.fill_(-1)
        target[..., : values.shape[-1]].copy_(values)
        target[..., -1].fill_(values.shape[-1])
        if group is not None and group != local_groups - 1:
            return
        values = self.region_local[token_start : token_start + n]
        if groups != local_groups or self.region_replicas > 1:
            from vllm.distributed.parallel_state import get_tp_group

            values = get_tp_group().all_gather(values.contiguous(), dim=1)
            values = values[:, :: self.region_replicas]
        self.rows[token_start : token_start + n].copy_(
            values.reshape(n, groups * width)
        )


class IndexerCaptureSource(Protocol):
    indexer_capture_spec: IndexerSource
    indexer_capture: IndexerCaptureSink | None
    indexer_capture_local_groups: int
    indexer_capture_group_replicas: int


@dataclass
class IndexerSnapshot:
    rows: torch.Tensor
    slot: int


class IndexerCapturer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: VllmConfig,
        *,
        output_rank: bool = True,
    ) -> None:
        sources: list[tuple[IndexerCaptureSource, IndexerSource]] = []
        for module in model.modules():
            source = getattr(module, "indexer_capture_spec", None)
            if isinstance(source, IndexerSource):
                sources.append((cast(IndexerCaptureSource, module), source))
        sources.sort(key=lambda entry: entry[1].layer_id)
        if not sources:
            raise ValueError(
                "No backbone indexer capture sources: this model/backend "
                "does not support indexer replay"
            )
        model_config = config.model_config
        identity = json.dumps(
            {
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
        self.profile = AuxOutputProfile(
            hashlib.sha256(identity.encode()).hexdigest(),
            tuple(source for _, source in sources),
        )
        self.profile_metadata = self.profile.to_dict()
        self.device_buffer = torch.empty(
            (
                config.scheduler_config.max_num_batched_tokens,
                *self.profile.shape_per_token,
            ),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        num_slots = config.max_concurrent_batches if output_rank else 0
        self._snapshot_rows = [
            torch.empty_like(self.device_buffer) for _ in range(num_slots)
        ]
        self._free_snapshot_slots = list(range(num_slots))
        self._sources = sources
        offset = 0
        for capture_source, source in sources:
            sink = IndexerCaptureSink(
                source,
                self.device_buffer[:, offset : offset + source.width],
            )
            if source.semantic == "region_id":
                local_groups = capture_source.indexer_capture_local_groups
                sink.region_local = torch.empty(
                    (len(self.device_buffer), local_groups, source.shape[-1]),
                    device=self.device_buffer.device,
                    dtype=torch.int32,
                )
                sink.region_replicas = capture_source.indexer_capture_group_replicas
            capture_source.indexer_capture = sink
            offset += source.width

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
        if not hasattr(self, "device_buffer"):
            return
        for source, _ in self._sources:
            source.indexer_capture = None
        self._sources.clear()
        self._snapshot_rows.clear()
        self._free_snapshot_slots.clear()
        del self.device_buffer


def configure_indexer_capture(
    module: torch.nn.Module,
    config: VllmConfig,
    source: IndexerSource | None,
    *,
    local_groups: int = 0,
    group_replicas: int = 1,
) -> None:
    capture_source = cast(IndexerCaptureSource, module)
    capture_source.indexer_capture = None
    if (
        config.aux_output_config.enable_return_indexer_topk
        and source is not None
        and source.layer_id < config.model_config.get_total_num_hidden_layers()
    ):
        capture_source.indexer_capture_spec = source
        capture_source.indexer_capture_local_groups = local_groups
        capture_source.indexer_capture_group_replicas = group_replicas
