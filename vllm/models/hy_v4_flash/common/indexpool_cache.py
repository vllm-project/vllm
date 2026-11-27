# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned pool pages and speculative tail rings for Flash IndexPool."""

from dataclasses import dataclass, replace
from math import lcm

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

POOL_PAGE_SIZE = 64


@dataclass
class IndexPoolMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int
    max_seq_len: int


class IndexPoolMetadataBuilder(AttentionMetadataBuilder[IndexPoolMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.max_model_len = vllm_config.model_config.max_model_len

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm = common_attn_metadata
        return IndexPoolMetadata(
            query_start_loc=cm.query_start_loc[: cm.num_reqs + 1],
            seq_lens=cm.seq_lens[: cm.num_reqs],
            block_table=cm.block_table_tensor[: cm.num_reqs],
            num_actual_tokens=cm.num_actual_tokens,
            max_seq_len=cm.max_seq_len,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        """Reserve scoring width for replay beyond the dummy's short context."""
        metadata = self.build(0, common_attn_metadata)
        return replace(metadata, max_seq_len=self.max_model_len)


class IndexPoolBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        if kv_cache_spec is None:
            config = get_current_vllm_config().model_config.hf_text_config
            ratio = config.compressed_indexer_compress_ratio
        else:
            ratio = int(kv_cache_spec.tokens_per_state)
        return [ratio * POOL_PAGE_SIZE]

    @classmethod
    def get_supported_head_sizes(cls):
        return [132]

    @classmethod
    def supports_pcp(cls):
        return False


class IndexPoolTailBackend(KpoolTailBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL_TAIL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False


def tail_ring_size(ratio: int, speculative_tokens: int) -> int:
    return ratio * next_power_of_2(cdiv(ratio + speculative_tokens, ratio))


class IndexPoolCache(DeepseekV32IndexerCache):
    def __init__(self, prefix, cache_config, ratio, *, tail=False):
        super().__init__(
            head_dim=131 if tail else 132,
            dtype=torch.float32 if tail else torch.uint8,
            prefix=prefix,
            cache_config=cache_config,
        )
        self.ratio = ratio
        self.tail = tail

    def get_attn_backend(self):
        return IndexPoolTailBackend if self.tail else IndexPoolBackend

    def get_kv_cache_config(self, vllm_config):
        ring = tail_ring_size(self.ratio, vllm_config.num_speculative_tokens)
        if self.tail:
            spec = AttentionSpec(
                block_size=ring,
                num_kv_heads=1,
                head_size=131,
                head_size_v=0,
                dtype=torch.float32,
                dcp_sharded=False,
            )
            return CircularBufferManager.layer_config(spec)
        spec = MLAAttentionSpec(
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
            block_size=lcm(
                self.cache_config.block_size, self.ratio * POOL_PAGE_SIZE, ring
            ),
            tokens_per_state=self.ratio,
        )
        return FullAttentionManager.layer_config(spec)

    def bind_kv_cache(self, kv_cache):
        super().bind_kv_cache(kv_cache)
        if not self.tail and self.kv_cache.shape[1] != POOL_PAGE_SIZE:
            raise ValueError("IndexPool must be bound as 64-state kernel pages")
