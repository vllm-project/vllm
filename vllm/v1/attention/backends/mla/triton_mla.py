# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

from vllm import envs
from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)

logger = init_logger(__name__)


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

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

        self.use_aiter_prefill = False
        if current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER_MLA_PREFILL:
            logger.info_once("Using Aiter prefill for MLA")
            self.use_aiter_prefill = True

        if is_quantized_kv_cache(self.kv_cache_dtype):
            if self.kv_cache_dtype.startswith("fp8"):
                logger.warning_once(
                    """TritonMLA V1 with FP8 KV only works when all
                    layer scales are 1.0. Any other value will raise
                    an error."""
                )
            else:
                raise NotImplementedError(
                    "TritonMLA V1 with quantized KV cache not yet supported"
                )

        self.force_num_kv_splits = envs.VLLM_FORCE_NUM_KV_SPLITS
        if self.force_num_kv_splits:
            logger.info_once(
                "Forcing TritonMLA num_kv_splits to %d",
                self.force_num_kv_splits,
            )

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        if self.use_aiter_prefill:
            from aiter import flash_attn_varlen_func as aiter_fa_func

            return aiter_fa_func(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                return_lse=return_softmax_lse,
                **kwargs,
            )
        return super()._flash_attn_varlen_diff_headdims(
            q,
            k,
            v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8") and (
            layer._q_scale != 1.0 or layer._k_scale != 1.0 or layer._v_scale != 1.0
        ):
            raise NotImplementedError(
                """
                FP8 Triton MLA with q,k,v scales
                {layer._q_scale},{layer._k_scale},{layer._v_scale}
                not yet supported. Only values of 1.0 supported.
            """
            )

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
        num_kv_splits = 1 if vllm_is_batch_invariant() else 4
        if self.force_num_kv_splits:
            num_kv_splits = self.force_num_kv_splits

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
        )

        return o, lse
