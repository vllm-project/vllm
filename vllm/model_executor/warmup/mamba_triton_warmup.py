# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Mamba-style Triton kernels shared across GDN / Mamba / KDA models."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _has_mamba_style_cache(runner: "GPUModelRunner") -> bool:
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    for group in runner.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if isinstance(spec, MambaSpec):
            return True
    return False


def _warm_batch_memcpy_kernel(device: torch.device) -> None:
    """Warm the Mamba prefix-cache state copy specialization."""
    from vllm.v1.worker.mamba_utils import batch_memcpy

    src = torch.empty(1024, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    batch_memcpy(
        torch.tensor([src.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([dst.data_ptr()], dtype=torch.uint64, device=device),
        # Keep this int32: Triton's compile key includes pointer element dtypes,
        # and the production prefix-cache path passes an int32 sizes tensor.
        torch.tensor([src.numel()], dtype=torch.int32, device=device),
    )
    logger.info("Warmed Mamba batch_memcpy_kernel.")


@torch.inference_mode()
def mamba_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm prefix-cache memcpy for every Mamba-style KV cache group."""
    device = runner.device
    if device.type != "cuda":
        return
    if not _has_mamba_style_cache(runner):
        return
    _warm_batch_memcpy_kernel(device)
