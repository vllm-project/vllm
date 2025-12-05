# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import einops
import torch
import torch.nn.functional as F

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.platforms import current_platform

logger = init_logger(__name__)


def maybe_get_vit_flash_attn_backend(
    attn_backend: AttentionBackendEnum | None,
) -> Callable | None:
    # At this point,
    # we already have the attn_backend,
    # overriding logic is done in the platform-specific implementation.
    # so we don't need to override backend here.
    # Just return the attn_backend and flash_attn_varlen_func.

    if (
        attn_backend == AttentionBackendEnum.FLASH_ATTN
        and current_platform.is_cuda_alike()
    ):
        from flash_attn import flash_attn_varlen_func
    elif attn_backend == AttentionBackendEnum.FLASH_ATTN and current_platform.is_xpu():
        from vllm.attention.utils.fa_utils import flash_attn_varlen_func
    elif attn_backend == AttentionBackendEnum.ROCM_AITER_FA:
        from aiter import flash_attn_varlen_func
    else:
        flash_attn_varlen_func = None

    # if attn_backend is TORCH_SDPA,
    # it will reach here and the flash_attn_varlen_func will be None.
    return flash_attn_varlen_func


@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """Multi-headed attention without any cache, used for multimodal encoder."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        # This has no effect, it is only here to make it easier to swap
        # between Attention and MultiHeadAttention
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Try to get vision attention backend from multimodal_config.
        attn_backend_override = None
        if multimodal_config is not None:
            attn_backend_override = multimodal_config.mm_encoder_attn_backend

        # Get device-specific vision attention backend.
        backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
            attn_backend_override=attn_backend_override,
        )
        self.attn_backend = backend

        self.flash_attn_varlen_func = maybe_get_vit_flash_attn_backend(
            self.attn_backend,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    def reshape_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape query, key, value to 4D tensors:
        (batch_size, seq_len, num_heads, head_size)
        """
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        return query, key, value

    def reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape query, key, value to 3D tensors:
        (batch_size * seq_len, num_heads, head_size)
        """
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        if cu_seqlens is None:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
            out = out.transpose(1, 2).reshape(q_len, bsz, -1)
        else:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = query[:, start_idx:end_idx]
                k_i = key[:, start_idx:end_idx]
                v_i = value[:, start_idx:end_idx]
                q_i, k_i, v_i = (
                    einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
                )
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            out = torch.cat(outputs, dim=1)
            out = einops.rearrange(out, "b s h d -> s b (h d)").contiguous()

        return out

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        assert self.flash_attn_varlen_func is not None, (
            "Flash attention function is not set."
        )

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_3d(
            query, key, value, bsz, q_len, kv_len
        )

        if cu_seqlens is None:
            cu_seqlens_q = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k = torch.arange(
                0, (bsz + 1) * kv_len, step=kv_len, dtype=torch.int32, device=key.device
            )
        else:
            # Use pre-computed cu_seqlens from ViT.
            cu_seqlens_q = cu_seqlens
            cu_seqlens_k = cu_seqlens

        max_seqlen = max_seqlen.item() if max_seqlen else None

        output = self.flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        output = einops.rearrange(output, "(b s) h d -> s b (h d)", b=bsz).contiguous()
        return output

    def _forward_pallas(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from torch_xla.experimental.custom_kernel import flash_attention

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        if cu_seqlens is None:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2).reshape(q_len, bsz, -1)
        else:
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = query[:, start_idx:end_idx]
                k_i = key[:, start_idx:end_idx]
                v_i = value[:, start_idx:end_idx]
                q_i, k_i, v_i = (
                    einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
                )
                output_i = flash_attention(q_i, k_i, v_i, sm_scale=self.scale)
                output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            out = torch.cat(outputs, dim=1)
            out = einops.rearrange(out, "b s h d -> s b (h d)").contiguous()

        return out

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        raise ValueError(
            f"Unsupported mm attention backend for CUDA: {self.attn_backend}"
        )

    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_tpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        assert self.attn_backend == AttentionBackendEnum.PALLAS, (
            f"MMEncoderAttention on TPU only supports PALLAS backend, "
            f"but got {self.attn_backend}."
        )
        return self._forward_pallas(query, key, value, cu_seqlens)
