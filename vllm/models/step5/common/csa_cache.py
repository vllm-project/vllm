# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache-manager side caches for Step-5 CSA: per-token index rows and
per-region summaries."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.step5.common.config import get_step5_sparse_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)


@dataclass
class Step5CSAMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_actual_tokens: int
    num_reqs: int
    max_seq_len: int
    token_req: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    completed_regions: torch.Tensor | None = None


def _completed_regions(m: CommonAttentionMetadata, region_size: int) -> torch.Tensor:
    """``(request, region)`` pairs whose last token was cached this step.

    Built on device from exact ``seq_lens``, padded with ``-1`` rows to a
    host-known upper bound so the host never reads the count.
    """
    seq = m.seq_lens[: m.num_reqs]
    first = torch.div(
        m.compute_num_computed_tokens()[: m.num_reqs],
        region_size,
        rounding_mode="floor",
    )
    counts = (torch.div(seq, region_size, rounding_mode="floor") - first).clamp_min(0)
    ends = torch.cumsum(counts, 0)
    capacity = m.num_actual_tokens // region_size + m.num_reqs
    slot = torch.arange(capacity, device=seq.device)
    req = torch.searchsorted(ends, slot, right=True)
    live = req < m.num_reqs
    req = req.clamp_max(m.num_reqs - 1)
    region = first[req] + slot - (ends - counts)[req]
    work = torch.stack([torch.where(live, req, -1), torch.where(live, region, -1)], -1)
    return work.to(torch.int32)


class Step5CSAMetadataBuilder(AttentionMetadataBuilder[Step5CSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sparse = get_step5_sparse_config(vllm_config.model_config.hf_text_config)
        assert sparse is not None
        self.region_size = sparse.region_block_size
        self.per_token = int(kv_cache_spec.tokens_per_state) == 1
        self.token_req_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step5CSAMetadata:
        del common_prefix_len, fast_build
        m = common_attn_metadata
        md = Step5CSAMetadata(
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping[: m.num_actual_tokens],
            query_start_loc=m.query_start_loc,
            query_start_loc_cpu=m.query_start_loc_cpu,
            seq_lens=m.seq_lens,
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_seq_len=m.max_seq_len,
        )
        if not self.per_token:
            return md
        n = m.num_actual_tokens
        md.token_req = m.token_to_req_indices(self.token_req_buffer)
        if m.positions is None:
            positions = (
                m.compute_num_computed_tokens().index_select(0, md.token_req)
                + torch.arange(n, device=md.token_req.device)
                - m.query_start_loc.index_select(0, md.token_req)
            )
        else:
            positions = m.positions[:n]
        md.positions = positions.to(torch.int32)
        md.completed_regions = _completed_regions(m, self.region_size)
        return md


class Step5CSABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "STEP5_CSA_SIDE_CACHE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("CSA side caches have no attention impl")

    @staticmethod
    def get_builder_cls() -> type[Step5CSAMetadataBuilder]:
        return Step5CSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class _Step5CSACache(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        *,
        width: int,
        tokens_per_state: int,
        cache_config: CacheConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        if cache_config.block_size % tokens_per_state:
            raise ValueError(
                f"KV block size {cache_config.block_size} must be a multiple of "
                f"the CSA region size {tokens_per_state}"
            )
        self.width = width
        self.tokens_per_state = tokens_per_state
        self.cache_config = cache_config
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def forward(self) -> None: ...

    @property
    def pages(self) -> torch.Tensor:
        return self.kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Step5CSABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.width,
            dtype=torch.bfloat16,
            tokens_per_state=self.tokens_per_state,
            cache_role=SparseCacheRole.INDEXER,
        )


class Step5CSAIndexCache(_Step5CSACache):
    def __init__(self, *, proxy_dim: int, **kwargs) -> None:
        super().__init__(width=2 * proxy_dim, tokens_per_state=1, **kwargs)


class Step5CSASummaryCache(_Step5CSACache):
    def __init__(self, *, proxy_dim: int, region_size: int, **kwargs) -> None:
        super().__init__(width=proxy_dim, tokens_per_state=region_size, **kwargs)
