# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1 DSA indexer cache, shared by the NVIDIA and ROCm stacks."""

import math
from dataclasses import replace

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import MXFP4_BLOCK_SIZE
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    dsa_indexer_uses_fp4,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode
from vllm.v1.kv_cache_layout import KVCacheLayout


def kimi_k3_indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """FP8 (132 B/token) indexer keys by default (``indexer_kv_dtype='auto'``);
    MXFP4 (68 B/token) only when requested with ``--attention-config
    '{"indexer_kv_dtype": "mxfp4"}'``. MXFP4 needs SM10x (checked by
    ``dsa_indexer_uses_fp4``), the fused key insert (layernorm k-norm, no indexer
    RoPE) and no PCP; anything else raises instead of writing another format."""
    if not dsa_indexer_uses_fp4(vllm_config):
        return False
    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    dsa_config = getattr(text_config, "dsa_config", None)
    if dsa_config is not None and not (
        getattr(dsa_config, "index_k_norm_type", "rmsnorm") == "layernorm"
        and getattr(dsa_config, "index_rope_dim", 0) == 0
    ):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' needs the fused Kimi K3 DSA key insert "
            "(index_k_norm_type='layernorm', index_rope_dim=0)."
        )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if getattr(parallel_config, "prefill_context_parallel_size", 1) > 1:
        raise ValueError(
            "indexer_kv_dtype='mxfp4' is not supported with prefill context "
            "parallelism for Kimi K3 DSA."
        )
    return True


def indexer_cache_entry_bytes(head_dim: int, use_fp4_cache: bool) -> int:
    """Indexer key cache bytes per token: FP8 values + one 4 B scale, or MXFP4
    values (two per byte) + one UE8M0 byte per 32 values."""
    if use_fp4_cache:
        return head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
    return head_dim + 4


class KimiK3IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Writes the indexer keys at slots derived from the block table, as the
    compressed indexers are written: in a packed block row the block table
    addresses 64-token kernel pages, which manager slots do not."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_pcp:
            raise NotImplementedError("Kimi-K3.1 packed DSA caches with PCP")
        # The K3.1 key format, validated as KimiK3DSAIndexer selects it.
        self.indexer_uses_fp4 = kimi_k3_indexer_uses_fp4(self.vllm_config)
        self.slot_mapping_buffer = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        slot_mapping = get_compressed_slot_mapping(
            common_attn_metadata.num_actual_tokens,
            common_attn_metadata.slot_mapping,
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            common_attn_metadata.block_table_tensor,
            self.kv_cache_spec.block_size,
            compress_ratio=1,
            out=self.slot_mapping_buffer,
            dcp_world_size=self.dcp_world_size,
            cp_interleave=self.cp_kv_cache_interleave_size,
        )
        metadata = super().build(
            common_prefix_len,
            replace(common_attn_metadata, slot_mapping=slot_mapping),
            fast_build,
        )
        return metadata


class KimiK3IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        if kv_cache_spec is not None:
            fp4 = kv_cache_spec.head_size == 68
        else:
            fp4 = (
                get_current_vllm_config().attention_config.resolve_indexer_kv_dtype(
                    "fp8"
                )
                == "mxfp4"
            )
        return (
            [64]
            if fp4
            else DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(
                kv_cache_spec
            )
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Pack the indexer pages beside the MLA pages inside each block, so a
        # KDA state page sizes the whole block row rather than one MLA page.
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def writes_kv_cache_from_block_table(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[KimiK3IndexerMetadataBuilder]:
        return KimiK3IndexerMetadataBuilder


class KimiK3IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return KimiK3IndexerBackend


# Block sizes stay divisible across TP sizes up to this one, so P/D can pair
# blocks of different TP sizes.
_PD_REFERENCE_TP = 16


def dense_mla_draft_bytes_per_token(vllm_config: VllmConfig) -> int:
    """Per-token bytes of a dense MLA draft's caches, which share the DSA row:
    their full-attention MLA specs group with the target's (see
    ``configure_dense_draft``)."""
    spec_config = vllm_config.speculative_config
    if spec_config is None or spec_config.method not in ("eagle3", "dspark"):
        return 0
    draft_model_config = spec_config.draft_model_config
    draft_config = draft_model_config.hf_text_config
    kv_lora_rank = getattr(draft_config, "kv_lora_rank", None)
    if kv_lora_rank is None:
        return 0
    cache_dtype = spec_config.kv_cache_dtype or vllm_config.cache_config.cache_dtype
    page = MLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=kv_lora_rank + draft_config.qk_rope_head_dim,
        dtype=kv_cache_dtype_str_to_dtype(cache_dtype, draft_model_config),
        cache_dtype_str=cache_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    ).page_size_bytes
    return draft_config.num_hidden_layers * page


def packed_dsa_block_size(
    vllm_config: VllmConfig,
    mla_bytes_per_token: int,
    state_bytes: int,
    alignment: int,
) -> int | None:
    """Block size of the packed cache of a KDA-hybrid DSA model, else None.

    A block row holds every DSA layer's MLA and indexer pages, plus a dense
    MLA draft's pages; size it to fit as many KDA states as there are DSA
    layers. The KDA state shards with TP and the row does not, so the block is
    rounded at ``_PD_REFERENCE_TP`` and scaled by ``_PD_REFERENCE_TP / tp``.
    Block sizes of different TP sizes stay divisible, and the rounding stays
    fine: a power of two would leave up to half the row empty once spec
    decoding grows the conv state past one.
    """
    text_config = vllm_config.model_config.hf_text_config
    dsa_config = getattr(text_config, "dsa_config", None)
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if dsa_config is None or linear_attn_config is None:
        return None
    num_dsa = len(linear_attn_config["full_attn_layers"])
    num_states = len(linear_attn_config["kda_layers"])
    states_per_row = cdiv(num_states, cdiv(num_states, num_dsa))
    # The indexer keeps fp8 values plus one fp32 scale per token, or MXFP4.
    index_bytes = indexer_cache_entry_bytes(
        dsa_config.index_head_dim, kimi_k3_indexer_uses_fp4(vllm_config)
    )
    row_bytes_per_token = num_dsa * (
        mla_bytes_per_token + index_bytes
    ) + dense_mla_draft_bytes_per_token(vllm_config)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    reference_tp = math.lcm(_PD_REFERENCE_TP, tp_size)
    # TP-invariant: the unsharded state bytes, in reference-TP block units.
    reference_units = cdiv(
        states_per_row * state_bytes * tp_size,
        alignment * reference_tp * row_bytes_per_token,
    )
    return alignment * reference_units * (reference_tp // tp_size)
