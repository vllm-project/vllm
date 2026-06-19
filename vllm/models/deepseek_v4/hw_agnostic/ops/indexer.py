# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 indexer attention backend — hw-agnostic copy.

Vendored and pruned from
``vllm/v1/attention/backends/mla/indexer.py``.

The upstream module pairs the ``DeepseekV4IndexerBackend`` class with a
DeepGEMM-backed metadata builder
(``DeepseekV32IndexerMetadataBuilder``) that handles chunked prefill,
uniform-decode, and tile-scheduling for the FlashInfer sparse-MLA
kernels. None of that is needed on the hw-agnostic path because the
local ``SparseAttnIndexer`` stub (see ``ops/sparse_attn_indexer.py``)
is the only consumer of the indexer cache and it has no portable
forward implementation. The vendored copy therefore keeps only:

  * the backend class itself (referenced by
    ``DeepseekV4IndexerCache.get_attn_backend``);
  * the ``get_max_prefill_buffer_size`` helper used by the indexer
    layer to size its workspace.

OOT plugins that want a real metadata builder subclass
``DeepseekV4IndexerBackend`` and override ``get_builder_cls``.
"""

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder as _IndexerMetadataBuilderStub,
)


class DeepseekV4IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return _IndexerMetadataBuilderStub

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


def get_max_prefill_buffer_size(vllm_config: VllmConfig) -> int:
    """Indexer workspace sizing heuristic (DSv4 / V3.2 ports).

    Verbatim from the upstream comment: ``40`` is a magic constant chosen
    so the indexer prefill buffer (132 B per entry) fits inside the
    FlashMLA-sparse workspace (576 * 2 B per entry, 5 * max_model_len
    entries). ``40 = (576 * 2 // 132) * 5``.
    """
    max_model_len = vllm_config.model_config.max_model_len
    return 40 * max_model_len
