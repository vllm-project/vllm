# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 FlashMLA sparse backend, metadata, and metadata builders."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C1A,
    _LAYER_TYPE_C2A,
    _LAYER_TYPE_SWAONLY,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# v4.1 per-layer compress ratios: 0 = pure sliding window, 1 = full-length
# compressed cache, 2 = ratio-2 compressed cache. Ratio-1 and ratio-2 layers
# both attend over indexer topk indices into a shared compressed cache but
# differ in compressed page block size (block_size // ratio), so each needs its
# own FlashMLA tile-scheduler plan.
_V41_LAYER_TYPES: dict[int, str] = {
    0: _LAYER_TYPE_SWAONLY,
    1: _LAYER_TYPE_C1A,
    2: _LAYER_TYPE_C2A,
}


def deepseek_v41_layer_type(compress_ratio: int) -> str:
    layer_type = _V41_LAYER_TYPES.get(compress_ratio)
    if layer_type is None:
        raise ValueError(
            f"Unsupported DeepSeek V4.1 compress_ratio={compress_ratio}; "
            "expected 0, 1, or 2."
        )
    return layer_type


class DeepseekV41SparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """SWA metadata builder base for v4.1.

    The shared builder classifies decode layer types by the v4.0 ratios
    (1 = SWA-only, 4, 128). v4.1 uses 0 / 1 / 2, so recompute the set of
    tile-scheduler plans from the v4.1 topology.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        compress_ratios = getattr(
            self.vllm_config.model_config.hf_config, "compress_ratios", None
        ) or [0]
        self._layer_types = {
            deepseek_v41_layer_type(int(ratio)) for ratio in compress_ratios
        }


class DeepseekV4SparseMLABackend(AttentionBackend):
    """DeepSeek-V4.1 sparse-MLA backend base.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): DeepSeek-V4.1 runs its own attention layer
    (``DeepseekV4Attention``), so it does not reuse the V3.2 builder or impl, and
    only needs to declare its own metadata builder, KV-cache layout, and the
    sparse-MLA capability flags.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
        "nvfp4_ds_mla",  # V4.1 fp8 SWA rows + V4.1 fp4 compressed rows (SM100)
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64 if current_platform.is_device_capability_family(90) else 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4SparseMLAMetadataBuilder"]:
        return DeepseekV4SparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        # DeepSeek-V4.1 runs its attention through ``DeepseekV4Attention.forward``,
        # not the generic ``Attention``/``MLAAttention`` layer, so the backend's
        # impl class is never instantiated.
        raise NotImplementedError(
            "DeepseekV4SparseMLABackend has no separate impl class; DeepSeek-V4.1 "
            "attention runs through DeepseekV4Attention."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512.
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [9, 10]

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype == "nvfp4_ds_mla" and device_capability.major != 10:
            return "nvfp4_ds_mla needs SM100 (FlashMLA V4.1 fp8/fp4 sparse decode)"
        return None


@dataclass
class DeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int


class DeepseekV4SparseMLAMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4FlashMLAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        # Classify single-token queries (plus num_speculative_tokens via
        # supports_spec_as_decode=True) as decodes; longer queries go to prefill.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.topk_tokens = self.model_config.hf_config.index_topk

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,), dtype=torch.int32, device=device
        )

        assert isinstance(self.kv_cache_spec.tokens_per_state, int)
        self.compress_ratio = self.kv_cache_spec.tokens_per_state
        # Only kv-source layers own a compressed cache, and v4.1 compresses at
        # ratio 1 or 2.
        if self.compress_ratio not in (1, 2):
            raise ValueError(
                "DeepSeek V4.1 compressed-KV caches use compress_ratio 1 or 2; "
                f"got {self.compress_ratio}."
            )

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
        # stability when compress_ratio > 1.
        if self.compress_ratio > 1:
            self.compressed_slot_mapping_buffer = torch.empty(
                max_num_batched_tokens, dtype=torch.int64, device=device
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4FlashMLAMetadata:
        cm = common_attn_metadata
        req_id_per_token = cm.token_to_req_indices(self.req_id_per_token_buffer)

        slot_mapping = cm.slot_mapping
        if self.compress_ratio > 1:
            slot_mapping = get_compressed_slot_mapping(
                cm.num_actual_tokens,
                cm.query_start_loc,
                cm.seq_lens,
                cm.block_table_tensor.clamp_(min=0),
                int(self.kv_cache_spec.num_states),
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )

        return DeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class DeepseekV4FlashMLAMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekV4FlashMLABackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE_DSV41"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4FlashMLAMetadataBuilder]:
        return DeepseekV4FlashMLAMetadataBuilder
