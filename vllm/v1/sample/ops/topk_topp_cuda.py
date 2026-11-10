# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking with the CUDA filter kernel (csrc/topk_topp_filter.cu).

The kernel reads each row once from HBM, keeps the max of every 16 and every 256
logits on chip, and selects the cutoff exactly through these maxima, with ties
broken by the lowest index. It runs on SM90 and SM100 with FP32 logits and
per-row device k and p.
"""

import functools

import torch

import vllm.envs as envs
from vllm.platforms import current_platform

# Shared memory holds one max per 16 logits of a row (one CTA per row at large
# batch), which fits vocabularies up to about 750k.
_MAX_VOCAB_SIZE = 1 << 19


@functools.cache
def topk_topp_cuda_available(device_index: int = 0) -> bool:
    """Whether the kernel is built for, and supports, the GPU."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability(device_index)
    if capability is None or not (
        capability.major == 9 or (capability.major == 10 and capability.minor in (0, 3))
    ):
        return False
    import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)

    return hasattr(torch.ops._C, "top_k_top_p_filter")


def can_use_topk_topp_cuda(logits: torch.Tensor) -> bool:
    return (
        logits.dtype == torch.float32
        and logits.ndim == 2
        and _supports(logits.device, logits.shape[-1])
    )


def _supports(device: torch.device, vocab_size: int) -> bool:
    return (
        envs.VLLM_USE_CUDA_TOPK_TOPP_FILTER
        and device.type == "cuda"
        and vocab_size < _MAX_VOCAB_SIZE
        and topk_topp_cuda_available(_index(device))
    )


def apply_top_k_top_p_cuda(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask logits outside the per-row top-k, then outside the top-p nucleus.

    Args:
        logits: [batch_size, vocab_size] float32 CUDA tensor.
        k: [batch_size] top-k per row, or None. k >= vocab_size disables it.
        p: [batch_size] top-p per row, or None. p >= 1 disables it.

    Returns:
        The masked logits. They alias the input unless its rows were not
        contiguous.

    """
    if k is None and p is None:
        return logits
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    if k is not None:
        k = k.to(torch.int32).contiguous()
    if p is not None:
        p = p.to(torch.float32).contiguous()
    torch.ops._C.top_k_top_p_filter(logits, k, p)
    return logits


def _index(device: torch.device) -> int:
    if device.index is None:
        return torch.accelerator.current_device_index()
    return device.index
