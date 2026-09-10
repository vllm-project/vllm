# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton image attention with FlashInfer causal attention and one KV cache."""

from typing import Any, TypeAlias

import torch

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout, KVCacheSpec

CompositeMetadata: TypeAlias = FlashInferMetadata | TritonAttentionMetadata


def requires_mm_prefix(
    metadata: CommonAttentionMetadata, *, unclamped_window: bool = False
) -> bool:
    ranges = metadata.mm_req_doc_ranges
    # With one query per request there are no future keys to unmask. This
    # also keeps single-token prefills on the captured FlashInfer path.
    if not ranges or (metadata.max_query_len <= 1 and not unclamped_window):
        return False
    seq_lens = metadata.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    starts = metadata.query_start_loc_cpu
    for req_idx, spans in ranges.items():
        query_len = int(starts[req_idx + 1] - starts[req_idx])
        if query_len == 0 or (query_len == 1 and not unclamped_window):
            continue
        end = int(seq_lens[req_idx])
        begin = end - query_len
        # An unclamped image can also expose keys behind the sliding window.
        if unclamped_window and any(
            start < stop and start < end and stop >= begin for start, stop in spans
        ):
            return True
        # Prefill lengths are exact. Historical image ranges must not divert
        # later text/decode queries to the image backend.
        if any(
            start < stop and start < end - 1 and stop > begin for start, stop in spans
        ):
            return True
    return False


def _has_unclamped_window(layers) -> bool:
    return any(
        getattr(layer, "sliding_window", None) is not None
        and not getattr(layer, "mm_prefix_clamp_sliding_window", False)
        for layer in layers
    )


class TritonFlashInferMetadataBuilder(AttentionMetadataBuilder[CompositeMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.causal = FlashInferMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )
        self.image = TritonAttentionMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, device
        )
        self.reorder_batch_threshold = self.causal.reorder_batch_threshold
        layers = vllm_config.compilation_config.static_forward_context
        self.unclamped_window = _has_unclamped_window(
            layers[name] for name in layer_names
        )

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        # Multi-token image prefills must not replay a causal attention graph.
        if _has_unclamped_window(
            vllm_config.compilation_config.static_forward_context.values()
        ):
            return AttentionCGSupport.NEVER
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def set_kernel_block_size(self, kernel_block_size: int) -> None:
        super().set_kernel_block_size(kernel_block_size)
        self.causal.set_kernel_block_size(kernel_block_size)
        self.image.set_kernel_block_size(kernel_block_size)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompositeMetadata:
        builder = (
            self.image
            if requires_mm_prefix(
                common_attn_metadata, unclamped_window=self.unclamped_window
            )
            else self.causal
        )
        return builder.build(common_prefix_len, common_attn_metadata, fast_build)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashInferMetadata:
        return self.causal.build_for_cudagraph_capture(common_attn_metadata)


class TritonFlashInferImpl(FlashInferImpl):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image = TritonAttentionBackend.get_impl_cls()(*args, **kwargs)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        self.image.process_weights_after_loading(act_dtype)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CompositeMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(attn_metadata, TritonAttentionMetadata):
            return self.image.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class TritonFlashInferBackend(AttentionBackend):
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_FLASHINFER_COMPOSITE"

    @staticmethod
    def get_impl_cls() -> type[TritonFlashInferImpl]:
        return TritonFlashInferImpl

    @staticmethod
    def get_builder_cls() -> type[TritonFlashInferMetadataBuilder]:
        return TritonFlashInferMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [256, 512]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        return FlashInferBackend.supported_kv_cache_layouts()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        config = get_current_vllm_config_or_none()
        if config is not None and config.model_config is not None:
            model = config.model_config
            if model.rswa_window is not None:
                return "R-SWA is not supported by the causal backend"
            num_kv_heads = model.get_num_kv_heads(config.parallel_config)
            num_q_heads = model.get_num_attention_heads(config.parallel_config)
            if num_kv_heads <= 0 or num_q_heads <= num_kv_heads:
                return "128-token FlashInfer pages require grouped-query attention"
        return None
