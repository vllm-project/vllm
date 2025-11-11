# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.attention.backends.registry import _Backend
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.platforms import current_platform

logger = init_logger(__name__)
USE_XFORMERS_OPS = None


def check_xformers_availability():
    global USE_XFORMERS_OPS
    if USE_XFORMERS_OPS is not None:
        return USE_XFORMERS_OPS

    if current_platform.is_cuda() and current_platform.has_device_capability(100):
        # Xformers FA is not compatible with B200
        USE_XFORMERS_OPS = False
    else:
        try:
            from importlib.util import find_spec

            find_spec("xformers.ops")
            USE_XFORMERS_OPS = True
        except ImportError:
            USE_XFORMERS_OPS = False

    # the warning only needs to be shown once
    if not USE_XFORMERS_OPS:
        logger.warning("Xformers is not available, falling back.")

    return USE_XFORMERS_OPS


def check_upstream_fa_availability(dtype: torch.dtype):
    if (
        dtype in (torch.float16, torch.bfloat16)
        and current_platform.is_cuda()
        and current_platform.has_device_capability(80)
    ):
        from transformers.utils import is_flash_attn_2_available

        return is_flash_attn_2_available()
    if current_platform.is_rocm():
        from importlib.util import find_spec

        return find_spec("flash_attn") is not None
    return False


def maybe_get_vit_flash_attn_backend(
    attn_backend: _Backend, use_upstream_fa: bool
) -> tuple[_Backend, Callable]:
    if (
        attn_backend != _Backend.FLASH_ATTN
        and attn_backend != _Backend.ROCM_AITER_FA
        and check_upstream_fa_availability(torch.get_default_dtype())
    ):
        attn_backend = _Backend.FLASH_ATTN
        use_upstream_fa = True

    if current_platform.is_rocm() and attn_backend == _Backend.FLASH_ATTN:
        use_upstream_fa = True

    if attn_backend in {_Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA}:
        if attn_backend == _Backend.ROCM_AITER_FA:
            from aiter import flash_attn_varlen_func
        else:
            if use_upstream_fa:
                from flash_attn import flash_attn_varlen_func
            else:
                from vllm.vllm_flash_attn import flash_attn_varlen_func
    else:
        flash_attn_varlen_func = None

    return attn_backend, flash_attn_varlen_func


@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """Multi-headed attention without any cache, used for multimodal encoder."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        # This has no effect, it is only here to make it easier to swap
        # between Attention and MMEncoderAttention
        prefix: str = "",
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

        # Determine the attention backend
        backend = get_vit_attn_backend(head_size=head_size, dtype=dtype)

        # Some auto-selected backends can be upgraded
        # to upstream flash attention if available.
        # If vllm native fa is selected, we use it directly.
        use_upstream_fa = False

        self.attn_backend: _Backend = backend
        self.attn_backend, self._flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
            )
        )

        if self.attn_backend == _Backend.XFORMERS and not check_xformers_availability():
            self.attn_backend = _Backend.TORCH_SDPA

        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN,
            _Backend.ROCM_AITER_FA,
        }

        # this condition is just to make sure that the
        # use_upstream_fa in the log is correct
        if current_platform.is_rocm() and self.attn_backend == _Backend.FLASH_ATTN:
            use_upstream_fa = True

        logger.info_once(
            f"MMEncoderAttention attn_backend: {self.attn_backend}, "
            f"use_upstream_fa: {use_upstream_fa}"
        )

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

    def _forward_pallas(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        from torch_xla.experimental.custom_kernel import flash_attention

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        out = flash_attention(query, key, value, sm_scale=self.scale)
        out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)

    def _forward_xformers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        from xformers import ops as xops

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        out = xops.memory_efficient_attention_forward(
            query, key, value, scale=self.scale
        )
        return out.reshape(bsz, q_len, -1)

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_3d(
            query, key, value, bsz, q_len, kv_len
        )

        cu_seqlens_q = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device
        )
        cu_seqlens_k = torch.arange(
            0, (bsz + 1) * kv_len, step=kv_len, dtype=torch.int32, device=key.device
        )

        out = self._flash_attn_varlen_func(
            query,
            key,
            value,
            softmax_scale=self.scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
        return out.reshape(bsz, q_len, -1)

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query, key, value = self.reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
        out = out.transpose(1, 2)
        return out.reshape(bsz, q_len, -1)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value)
        elif self.attn_backend == _Backend.XFORMERS:
            return self._forward_xformers(query, key, value)
        elif self.attn_backend == _Backend.TORCH_SDPA:
            return self._forward_sdpa(query, key, value)
        raise ValueError(
            f"Unsupported mm attention backend for CUDA: {self.attn_backend}"
        )

    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        return self._forward_sdpa(query, key, value)

    def forward_tpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        assert self.attn_backend == _Backend.PALLAS, (
            f"MMEncoderAttention on TPU only supports PALLAS backend, "
            f"but got {self.attn_backend}."
        )
        return self._forward_pallas(query, key, value)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        return self._forward_sdpa(query, key, value)
