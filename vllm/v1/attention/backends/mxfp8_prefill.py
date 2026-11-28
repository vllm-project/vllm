# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in MXFP8 full prefill with native FlashAttention cache and decode."""

from dataclasses import dataclass
from math import isclose

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.mxfp8_prefill import cudnn_frontend_problem


def _full_prefill_tokens(common: CommonAttentionMetadata, common_prefix_len=0):
    """Use exact CPU context counts; never synchronize device metadata."""
    if (
        common.num_reqs != 1
        or common.causal is not True
        or common_prefix_len
        or common.mm_req_doc_ranges
        or common.rswa_prefix_lens is not None
    ):
        return 0
    # This bound is exact outside async speculative decode. The builder disables
    # the custom route whenever speculative decoding is configured.
    lengths = common.seq_lens_cpu_upper_bound
    if lengths is None:
        return 0
    starts = common.query_start_loc_cpu
    n = int(starts[1]) - int(starts[0])
    if (
        int(starts[0]) != 0
        or n < 128
        or n % 128
        or n != common.num_actual_tokens
        or int(lengths[0]) != n
    ):
        return 0
    return n


@dataclass
class Mxfp8PrefillMetadata(FlashAttentionMetadata):
    mxfp8_prefill_tokens: int = 0


class Mxfp8PrefillMetadataBuilder(FlashAttentionMetadataBuilder):
    # Only native decode is captured in CUDA graphs; MXFP8 prefills run eagerly.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if self.vllm_config.speculative_config is not None:
            return metadata
        n = _full_prefill_tokens(common_attn_metadata, common_prefix_len)
        if not n:
            return metadata
        return Mxfp8PrefillMetadata(**vars(metadata), mxfp8_prefill_tokens=n)


class Mxfp8PrefillBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "MXFP8_PREFILL"

    @staticmethod
    def get_impl_cls():
        return Mxfp8PrefillImpl

    @staticmethod
    def get_builder_cls():
        return Mxfp8PrefillMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

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
        return cudnn_frontend_problem() or super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )


class Mxfp8PrefillImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config_or_none()
        self._mxfp8_enabled = (
            self.head_size == 128
            and self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            and self.num_heads + 2 * self.num_kv_heads <= 65535
            and isclose(self.scale, 128**-0.5, rel_tol=1e-12)
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.kv_sharing_target_layer_name is None
            and self.kv_cache_dtype in ("auto", "bfloat16")
            and not self.batch_invariant_enabled
            and self.dcp_world_size == 1
            and config is not None
            and config.parallel_config.prefill_context_parallel_size == 1
            and config.speculative_config is None
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        n = getattr(attn_metadata, "mxfp8_prefill_tokens", 0)
        if (
            not self._mxfp8_enabled
            or not n
            or key is None
            or value is None
            or any(x.dtype != torch.bfloat16 for x in (query, key, value, output))
            or query.device.type != "cuda"
            or any(x.device != query.device for x in (key, value, output))
            or n * self.num_heads * 128 >= 2**31
            or any(
                (n - 1) * x.stride(0) + (x.shape[1] - 1) * x.stride(1) + 127 >= 2**31
                for x in (query, key, value)
            )
            or any(x.stride(-1) != 1 for x in (query, key, value, output))
            or not output[:n].is_contiguous()
            or output_scale is not None
            or output_block_scale is not None
            or torch.cuda.is_current_stream_capturing()
        ):
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
        # Imported lazily so the native paths never load cuDNN Frontend.
        from vllm.v1.attention.ops.mxfp8_prefill.boundary import forward

        q, k, v = (x[:n].transpose(0, 1).unsqueeze(0) for x in (query, key, value))
        result = forward(q, k, v)
        output[:n].copy_(result.squeeze(0).transpose(0, 1))
        return output
