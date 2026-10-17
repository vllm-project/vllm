# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm Mamba-style Triton kernels shared across GDN / Mamba / KDA models."""

from math import gcd
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


def _warm_mamba_align_state_copy_kernels(runner: "GPUModelRunner") -> None:
    if runner.cache_config.mamba_cache_mode != "align":
        return
    ctx = runner._get_mamba_bufs().postprocess_align
    if ctx is None:
        return

    ctx.initialize_from_forward_context(
        runner.kv_cache_config,
        runner.compilation_config.static_forward_context,
        runner._get_mamba_state_copy_funcs(),
        [
            runner.input_batch.block_table[gid].get_device_tensor(1)
            for gid in ctx.mamba_group_ids
        ],
    )
    zeros = torch.zeros(1, dtype=torch.int32, device=runner.device)
    ones = torch.ones_like(zeros)
    # Fresh pre-copy and an unaligned post-step count both skip state writes.
    ctx.run_fused_precopy(1, zeros, -ones, zeros, idx_mapping=None)
    ctx.run_fused_postprocess(1, ones, zeros, ones, zeros, zeros)
    logger.info("Warmed Mamba align state-copy kernels.")


def _warm_short_conv_prefill_kernel(
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_speculative_blocks: int,
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    device = weight.device
    x = torch.empty((1, weight.size(0)), dtype=weight.dtype, device=device).t()
    state_columns = 1 + num_speculative_blocks
    indices = torch.full(
        (4, state_columns), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )
    has_initial_state = torch.zeros(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    # Prefill takes column zero after slicing away the decode rows.
    with gpu_sync_allowed():
        for offset in range(4 // gcd(4, state_columns)):
            causal_conv1d_fn(
                x,
                weight,
                bias,
                conv_state,
                query_start_loc,
                cache_indices=indices[offset : offset + 1, 0],
                has_initial_state=has_initial_state,
                activation=None,
            )


def _warm_short_conv_prefill_kernels(runner: "GPUModelRunner") -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.short_conv import ShortConv
    from vllm.v1.worker.mamba_utils import _get_mamba_spec_for_layer

    seen = set()
    for group in runner.kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            layer = runner.compilation_config.static_forward_context.get(name)
            if not isinstance(layer, ShortConv):
                continue
            conv_state = layer.kv_cache[0]
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            weight = layer.conv.weight.view(layer.conv.weight.size(0), -1)
            spec = _get_mamba_spec_for_layer(group, name)
            key = (
                weight.shape,
                weight.dtype,
                layer.conv.bias is not None,
                conv_state.shape,
                conv_state.stride(),
                conv_state.dtype,
                spec.num_speculative_blocks,
            )
            if key in seen:
                continue
            seen.add(key)
            _warm_short_conv_prefill_kernel(
                conv_state, weight, layer.conv.bias, spec.num_speculative_blocks
            )


@torch.inference_mode()
def mamba_triton_warmup(
    runner: "GPUModelRunner", *, use_v2_model_runner: bool = False
) -> None:
    """Warm prefix-cache and V1 Mamba kernels outside graph capture."""
    device = runner.device
    if device.type != "cuda":
        return
    if not _has_mamba_style_cache(runner):
        return
    _warm_batch_memcpy_kernel(device)
    if not use_v2_model_runner:
        _warm_mamba_align_state_copy_kernels(runner)
        _warm_short_conv_prefill_kernels(runner)
