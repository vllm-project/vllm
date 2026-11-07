# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline dependency planning, cache replicas and intermediate tensor transport."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v41.attention import (
        DeepseekV4Attention,
        DeepseekV4IndexerCache,
    )
    from vllm.sequence import IntermediateTensors
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class PipelineBoundary:
    kv_source: int | None = None
    index_source: int | None = None
    candidate_source: int | None = None

    @property
    def needs_relay(self) -> bool:
        return any(
            source is not None
            for source in (self.kv_source, self.index_source, self.candidate_source)
        )


def get_pipeline_boundaries(
    config: Any, stage_ranges: list[tuple[int, int]]
) -> tuple[PipelineBoundary, ...]:
    """Resolve cache, top-k and candidate lifetimes at every stage cut."""
    num_layers = config.num_hidden_layers
    if (
        not stage_ranges
        or stage_ranges[0][0] != 0
        or stage_ranges[-1][1] != num_layers
        or any(start >= end for start, end in stage_ranges)
        or any(a[1] != b[0] for a, b in zip(stage_ranges, stage_ranges[1:]))
    ):
        raise ValueError("DeepSeek-V4.1 pipeline stages must partition all layers")
    ratios = config.compress_ratios
    if len(ratios) < num_layers or any(r not in (0, 1, 2) for r in ratios):
        raise ValueError("Invalid DeepSeek-V4.1 compress_ratios")
    sources = {}
    for kind, field in (
        ("kv", "kv_source_layer_ids"),
        ("index", "index_source_layer_ids"),
    ):
        values = tuple(getattr(config, field, ()) or ())
        if (
            values != tuple(sorted(set(values)))
            or any(type(s) is not int or not 0 <= s < num_layers for s in values)
            or any(ratios[s] == 0 for s in values)
        ):
            raise ValueError(f"Invalid DeepSeek-V4.1 {field}")
        sources[kind] = values
    if not set(sources["kv"]).issubset(sources["index"]):
        raise ValueError("KV sources must also publish index keys")
    dependencies: dict[str, list[tuple[int, int]]] = {"kv": [], "index": []}
    for layer in range(num_layers):
        if ratios[layer] == 0:
            continue
        for kind, values in sources.items():
            preceding = [s for s in values if s <= layer]
            if not preceding:
                raise ValueError(f"Layer {layer} has no preceding {kind} source")
            source = preceding[-1]
            if ratios[source] != ratios[layer]:
                raise ValueError(
                    f"Layer {layer} and its {kind} source disagree on ratio"
                )
            dependencies[kind].append((source, layer))
    candidate = getattr(config, "candidate_source_layer_id", -1)
    candidate_enabled = (
        candidate >= 0 and getattr(config, "candidate_topk_blocks", 0) > 0
    )
    if candidate_enabled and candidate not in sources["index"]:
        raise ValueError("Candidate source must be an index source")
    boundaries = []
    for _, cut in stage_ranges[:-1]:
        live = {}
        for kind, edges in dependencies.items():
            crossing = {s for s, consumer in edges if s < cut <= consumer}
            assert len(crossing) <= 1
            live[kind] = next(iter(crossing), None)
        boundaries.append(
            PipelineBoundary(
                kv_source=live["kv"],
                index_source=live["index"],
                candidate_source=(
                    candidate
                    if candidate_enabled
                    and candidate < cut
                    and any(s >= cut for s in sources["index"])
                    else None
                ),
            )
        )
    return tuple(boundaries)


class PipelineCache(nn.Module, AttentionLayerBase):
    """A source-named replica participates in the same scheduler cache group."""

    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.spec: KVCacheSpec | None = None
        self.backend: type[AttentionBackend] | None = None
        context = vllm_config.compilation_config.static_forward_context
        assert prefix not in context
        context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        assert self.spec is not None
        return self.spec

    def get_attn_backend(self) -> type[AttentionBackend]:
        assert self.backend is not None
        return self.backend

    def forward(self):
        raise RuntimeError("Pipeline cache replicas do not execute attention")


class PipelineRelay(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        incoming: PipelineBoundary,
        outgoing: PipelineBoundary,
    ):
        from vllm.models.deepseek_v41.attention import (
            DeepseekV4IndexerCache,
            _indexer_k_cache_head_dim,
        )
        from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.incoming = incoming
        self.outgoing = outgoing
        self.context = vllm_config.compilation_config.static_forward_context
        self.main_cache: PipelineCache | None = None
        self.index_cache: DeepseekV4IndexerCache | None = None
        self.main_row_bytes = 0
        self.index_row_bytes = _indexer_k_cache_head_dim(
            self.config.index_head_dim, dsa_indexer_uses_fp4(vllm_config)
        )
        if incoming.kv_source is not None:
            source = incoming.kv_source
            source_prefix = f"{prefix}.layers.{source}.attn"
            self.main_cache = PipelineCache(vllm_config, source_prefix)
            self.index_cache = DeepseekV4IndexerCache(
                head_dim=self.index_row_bytes,
                dtype=torch.uint8,
                prefix=f"{source_prefix}.indexer.k_cache",
                cache_config=vllm_config.cache_config,
                compress_ratio=self.config.compress_ratios[source],
            )

    def configure(self, attention: DeepseekV4Attention, vllm_config: VllmConfig):
        self.main_row_bytes = attention.compressed_bytes_per_token
        if self.main_cache is not None:
            source = self.incoming.kv_source
            assert source is not None
            self.main_cache.spec = replace(
                attention.get_compressed_kv_cache_spec(vllm_config),
                tokens_per_state=self.config.compress_ratios[source],
            )
            self.main_cache.backend = attention.get_attn_backend()

    def _cache_entries(self, boundary: PipelineBoundary):
        source = boundary.kv_source
        if source is not None:
            prefix = f"{self.prefix}.layers.{source}.attn"
            yield (
                f"pp_main_kv_{source}",
                prefix,
                self.main_row_bytes,
                (576 if self.main_row_bytes == 584 else self.main_row_bytes),
            )
            yield (
                f"pp_index_k_{source}",
                f"{prefix}.indexer.k_cache",
                self.index_row_bytes,
                128 if self.index_row_bytes == 132 else 64,
            )

    def empty_tensors(
        self, boundary: PipelineBoundary, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        tensors = {
            key: torch.zeros((batch_size, width), dtype=torch.uint8, device=device)
            for key, _, width, _ in self._cache_entries(boundary)
        }
        if boundary.index_source is not None:
            tensors["pp_topk"] = torch.full(
                (batch_size, self.config.index_topk),
                -1,
                dtype=torch.int32,
                device=device,
            )
        if boundary.candidate_source is not None:
            tensors["pp_candidates"] = torch.full(
                (batch_size, self.config.candidate_topk_blocks),
                -1,
                dtype=torch.int32,
                device=device,
            )
        return tensors

    def transfer(
        self,
        positions: torch.Tensor,
        topk: torch.Tensor,
        candidates: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> dict[str, torch.Tensor]:
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            copy_pipeline_cache_rows,
        )

        importing = intermediate_tensors is not None
        boundary = self.incoming if importing else self.outgoing
        if intermediate_tensors is not None:
            tensors = intermediate_tensors.tensors
        else:
            tensors = self.empty_tensors(boundary, positions.numel(), positions.device)
        context = get_forward_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            return tensors
        for key, prefix, _, value_bytes in self._cache_entries(boundary):
            copy_pipeline_cache_rows(
                self.context[prefix].kv_cache,
                tensors[key],
                positions,
                cast(Any, metadata[prefix]).slot_mapping,
                self.config.compress_ratios[boundary.kv_source],
                value_bytes,
                scatter=importing,
            )
        for key, buffer in (("pp_topk", topk), ("pp_candidates", candidates)):
            if key in tensors:
                assert buffer is not None
                if importing:
                    buffer[: positions.numel()].copy_(tensors[key])
                else:
                    tensors[key].copy_(buffer[: positions.numel()])
        return tensors
