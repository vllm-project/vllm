# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def fused_output_quant_supported(self, quant_key) -> bool:
        """Check if this backend supports fused output quantization
        for the given quant_key.
        """
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic64Sym,
            kFp8Dynamic128Sym,
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
        )

        supported_keys = {
            kFp8StaticTensorSym,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
            kNvfp4Dynamic,
        }

        return quant_key in supported_keys

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # Minimum work per split
            # hardware dependent
            min_work_per_split = 512

            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)

            # use power of 2 to avoid excessive kernel instantiations
            ideal_splits = triton.next_power_of_2(ideal_splits)

            # Calculate SM-based maximum splits with occupancy multiplier
            # 2-4x allows multiple blocks per SM for latency hiding
            # hardware dependent
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Stage1: attention computation (always use standard kernel)
        from vllm.v1.attention.ops.triton_decode_attention import (
            _decode_grouped_att_m_fwd,
        )
        _decode_grouped_att_m_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            attn_logits,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            logit_cap=0.0,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        # Stage2: reduction + quantization (fused when quant params provided)
        if output_scale is not None or output_block_scale is not None:
            from vllm.v1.attention.ops.mla_attn_quant_fused import (
                decode_softmax_reducev_fwd_fused_fp8_static,
                decode_softmax_reducev_fwd_fused_fp8_group,
            )

            if output_block_scale is not None:
                # Per-group FP8 quantization
                assert quant_group_size is not None
                decode_softmax_reducev_fwd_fused_fp8_group(
                    attn_logits,
                    q,
                    o,
                    lse,
                    attn_metadata.decode.seq_lens,
                    num_kv_splits,
                    output_block_scale,
                    quant_group_size,
                )
            else:
                # Static FP8 quantization
                decode_softmax_reducev_fwd_fused_fp8_static(
                    attn_logits,
                    q,
                    o,
                    lse,
                    attn_metadata.decode.seq_lens,
                    num_kv_splits,
                    output_scale,
                )
        else:
            # Standard stage2 without quantization
            from vllm.v1.attention.ops.triton_decode_attention import (
                _decode_softmax_reducev_fwd,
            )
            _decode_softmax_reducev_fwd(
                attn_logits,
                q,
                o,
                lse,
                kv_c_cache,
                attn_metadata.decode.seq_lens,
                num_kv_splits,
            )

        return o, lse
