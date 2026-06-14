# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform


def _is_flash_attn_available() -> bool:
    if current_platform.is_rocm():
        try:
            from aiter import flash_attn_varlen_func  # noqa: F401

            return True
        except ImportError:
            return False
    if current_platform.is_xpu():
        try:
            from vllm.v1.attention.backends.fa_utils import (
                flash_attn_varlen_func,  # noqa: F401
            )

            return True
        except ImportError:
            return False
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            is_fa_version_supported,  # noqa: F401
        )

        return True
    except ImportError:
        return False


def _is_triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


FLASH_ATTN_AVAILABLE = _is_flash_attn_available()
TRITON_AVAILABLE = _is_triton_available()

FLASH_ATTN_SUPPORTED = FLASH_ATTN_AVAILABLE and (
    current_platform.is_cuda_alike() or current_platform.is_xpu()
)
TRITON_SUPPORTED = TRITON_AVAILABLE and (
    current_platform.is_cuda_alike() or current_platform.is_xpu()
)


def _fa_supports_args(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> bool:
    if query.dtype == torch.float32:
        return False
    head_size = query.size(-1)
    return head_size % 8 == 0 and head_size <= 256


@ir.ops.mm_encoder_attn.register_impl(
    "flash_attn", supports_args=_fa_supports_args, supported=FLASH_ATTN_SUPPORTED
)
def flash_attn_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

    bsz = query.size(0)
    head_size = query.size(-1)
    fa_version = get_flash_attn_version(head_size=head_size)
    max_seqlen_t = torch.tensor(max_seqlen) if max_seqlen is not None else None
    return vit_flash_attn_wrapper(
        q=query,
        k=key,
        v=value,
        batch_size=bsz,
        is_rocm_aiter=current_platform.is_rocm(),
        fa_version=fa_version,
        scale=scale,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen_t,
    )


def _triton_supports_args(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> bool:
    return query.size(2) == key.size(2)


@ir.ops.mm_encoder_attn.register_impl(
    "triton", supports_args=_triton_supports_args, supported=TRITON_SUPPORTED
)
def triton_impl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    from vllm.v1.attention.ops.vit_attn_wrappers import vit_triton_attn_wrapper

    bsz = query.size(0)
    max_seqlen_t = torch.tensor(max_seqlen) if max_seqlen is not None else None
    return vit_triton_attn_wrapper(
        q=query,
        k=key,
        v=value,
        batch_size=bsz,
        scale=scale,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen_t,
    )
