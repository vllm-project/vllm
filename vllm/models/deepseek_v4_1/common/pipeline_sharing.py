# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental eager relay of KV, index keys, indices and candidate blocks."""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.attention import (
    DeepseekV4Attention,
    make_pipeline_cache_replica,
)

from .pipeline import SharingDependency
from .pipeline_transfer import (
    SharingRoute,
    get_sharing_routes,
    restore_cache_blocks,
    snapshot_cache_blocks,
)


class PipelineSharing(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        stage: int,
        dependencies: tuple[SharingDependency, ...],
        attn_cls: type[DeepseekV4Attention],
        max_bytes: int,
    ) -> None:
        super().__init__()
        parallel = vllm_config.parallel_config
        if (
            not vllm_config.model_config.enforce_eager
            or vllm_config.use_v2_model_runner
            or parallel.use_ubatching
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or vllm_config.speculative_config is not None
            or vllm_config.kv_transfer_config is not None
            or parallel.distributed_executor_backend == "external_launcher"
        ):
            raise ValueError(
                "deepseek_v41_pp_sharing requires eager V1 execution without "
                "microbatching, context parallelism, speculative decoding, KV transfer "
                "or external_launcher"
            )
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError(
                "deepseek_v41_pp_share_max_bytes must be a positive integer"
            )
        self.max_bytes = max_bytes
        routes = get_sharing_routes(dependencies)
        self.inbound = tuple(r for r in routes if r.receiver == stage)
        self.outbound = tuple(r for r in routes if r.sender == stage)
        self.payload_keys = frozenset(k for r in routes for k in r.payload_keys)
        self._context = vllm_config.compilation_config.static_forward_context
        self._prefix = prefix
        self.replicas = nn.ModuleList()
        # Resolve the main cache dtype before constructing index-K replicas.
        for kind in ("kv", "index_k"):
            for route in self.inbound:
                if route.kind == kind:
                    self.replicas.append(
                        make_pipeline_cache_replica(
                            vllm_config,
                            self._source_prefix(route),
                            route.source_layer,
                            kind,
                            attn_cls,
                        )
                    )

    def _source_prefix(self, route: SharingRoute) -> str:
        return f"{self._prefix}.layers.{route.source_layer}.attn"

    def _cache_prefix(self, route: SharingRoute) -> str:
        prefix = self._source_prefix(route)
        return f"{prefix}.indexer.k_cache" if route.kind == "index_k" else prefix

    def receive(
        self,
        tensors: dict[str, torch.Tensor],
        topk: torch.Tensor,
        candidates: torch.Tensor | None,
        num_tokens: int,
    ) -> None:
        for route in self.inbound:
            if any(key not in tensors for key in route.payload_keys):
                raise ValueError(f"Missing pipeline sharing payload: {route.key}")
            if route.kind in ("kv", "index_k"):
                cache = self._context[self._cache_prefix(route)].kv_cache
                restore_cache_blocks(
                    cache, tensors[f"{route.key}.ids"], tensors[f"{route.key}.blocks"]
                )
            else:
                target = topk if route.kind == "index" else candidates
                assert target is not None
                values = tensors[route.key]
                if values.shape != target[:num_tokens].shape:
                    raise ValueError(
                        f"Pipeline sharing token shape mismatch: {route.key}"
                    )
                target[:num_tokens].copy_(values)

    def send(
        self,
        metadata: dict[str, Any],
        topk: torch.Tensor,
        candidates: torch.Tensor | None,
        num_tokens: int,
    ) -> dict[str, torch.Tensor]:
        payload = {}
        remaining = self.max_bytes
        for route in self.outbound:
            if route.kind in ("kv", "index_k"):
                prefix = self._cache_prefix(route)
                layer = self._context[prefix]
                meta = metadata[prefix]
                host_table = getattr(meta, "block_table_cpu", None)
                if host_table is not None:
                    tables = [host_table]
                elif route.kind == "kv":
                    tables = [meta.block_table[: meta.num_reqs]]
                else:
                    tables = []
                    if meta.decode is not None:
                        tables.append(meta.decode.block_table)
                    if meta.prefill is not None:
                        tables.extend(
                            chunk.block_table for chunk in meta.prefill.chunks
                        )
                ids, blocks = snapshot_cache_blocks(layer.kv_cache, tables, remaining)
                values = {f"{route.key}.ids": ids, f"{route.key}.blocks": blocks}
            else:
                source = topk if route.kind == "index" else candidates
                assert source is not None
                values = {route.key: source[:num_tokens].clone()}
            remaining -= sum(t.numel() * t.element_size() for t in values.values())
            if remaining < 0:
                raise ValueError("Pipeline sharing snapshot exceeds its byte budget")
            payload.update(values)
        return payload
