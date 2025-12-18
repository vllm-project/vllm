# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_torch_sdpa_wrapper,
)
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.config import MultiModalConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.models.vision import get_vit_attn_backend

logger = init_logger(__name__)


def maybe_get_vit_flash_attn_backend(
    attn_backend: AttentionBackendEnum | None,
) -> Callable | None:
    # At this point,
    # we already have the attn_backend,
    # overriding logic is done in the platform-specific implementation.
    # so we don't need to override backend here.
    # Just return the attn_backend and flash_attn_varlen_func.

    if attn_backend == AttentionBackendEnum.FLASH_ATTN:
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
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MultiHeadAttention
            multimodal_config: configs for multi-modal.
        """
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
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
            attn_backend_override=attn_backend_override,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        self.flash_attn_varlen_func = maybe_get_vit_flash_attn_backend(
            self.attn_backend,
        )

        if self.is_flash_attn_backend:
            assert self.flash_attn_varlen_func is not None
            self._fa_version = get_flash_attn_version()

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    @classmethod
    def enabled(cls) -> bool:
        return True

    def reshape_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 4D tensors:
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
        """
        Reshape query, key, value to 3D tensors:
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
        # TODO(Isotr0py): Migrate MultiHeadAttention
        assert cu_seqlens is not None

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = vit_torch_sdpa_wrapper(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens,
        )
        return output

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
        # # TODO(Isotr0py): Migrate MultiHeadAttention
        assert cu_seqlens is not None and max_seqlen is not None

        bsz = query.shape[0]

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=bsz,
            is_rocm_aiter=(self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA),
            fa_version=self._fa_version,
        )
        return output

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
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for CUDA: "
                f"{self.attn_backend}."
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
        assert self.is_flash_attn_backend, (
            "XPU only supports FLASH_ATTN for vision attention."
        )
        return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)

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
        if cu_seqlens is None:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            from torch_xla.experimental.custom_kernel import flash_attention

            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)
            return out
        logger.warning_once(
            "PALLAS backend with cu_seqlens is not supported for ViT yet. ",
            "Falling back to SDPA implementation.",
        )
        return self._forward_sdpa(query, key, value, cu_seqlens)
