# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Triton kernels for hybrid Mamba2 models (e.g. NemotronH).

Extends the Qwen Triton warmup (#46750 / #47546) to hybrid models built on
``MambaMixer2``. Without this, the JIT monitor reports these kernels
compiling during the first inference request:

- ``_causal_conv1d_fwd_kernel``: the Mamba2 SSD warmup covers only the SSD
  chunk kernels, and it runs during the profile pass before the conv cache
  exists, so the prefill conv kernel cannot be warmed there.
- ``_zero_kv_blocks_kernel``: only warmed for Qwen model types.
- ``_compute_slot_mapping_kernel``: the generic block-table warmup misses
  the ``block_table_stride == 1`` specialization that hybrid models hit
  because their large mamba-aligned attention block size usually yields a
  single block per request.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _synchronize_device,
    _warm_compute_slot_mapping_kernel,
    _warm_zero_kv_blocks_kernel,
    _warm_zero_kv_blocks_with_runner_zeroer,
    _zero_kv_warmup_config,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _iter_mamba_mixer2_layers(static_forward_context: object):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    if not isinstance(static_forward_context, dict):
        return
    for module in static_forward_context.values():
        if isinstance(module, MambaMixer2):
            yield module


def _get_conv_state(layer: object) -> torch.Tensor | None:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )

    kv_cache = getattr(layer, "kv_cache", None)
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 1:
        return None
    conv_cache = kv_cache[0]
    if not isinstance(conv_cache, torch.Tensor) or conv_cache.numel() == 0:
        return None
    return conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)


import itertools

# In production the prefill tensors (query_start_loc_p, cache_indices_p,
# has_initial_state_p) are slices offset by num_decodes of the per-step
# batch tensors, so each pointer's 16-byte alignment ("tt.divisibility"
# in the Triton JIT key) varies per step — and independently of each
# other, because cache_indices comes from a 2D block-table slice whose
# element offset is num_decodes * row_width while the others are 1D
# slices offset by num_decodes. Warm all eight aligned/unaligned
# combinations (offset 0 = 16-byte aligned, offset 1 = unaligned).
_CONV1D_WARMUP_SLICE_OFFSETS = tuple(itertools.product((0, 1), repeat=3))


def _warm_mamba2_causal_conv1d_fwd(device: torch.device, layer: object) -> bool:
    """Warm ``_causal_conv1d_fwd_kernel`` with the same JIT keys as a real
    prefill on ``layer``: real conv weights/bias/state (so the constexpr
    dims, dtypes and strides match) and a single dummy token routed to the
    null block (so no real cache line is written)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    conv_state = _get_conv_state(layer)
    if conv_state is None:
        return False

    cache_config = getattr(layer, "cache_config", None)
    mamba_block_size = getattr(cache_config, "mamba_block_size", None)
    if mamba_block_size is None:
        return False

    # Channel-last single-token input, like production's transposed slice
    # of the projected-states buffer. A dense transpose keeps the output
    # token stride > 1 so Triton's ``== 1`` stride specialization matches
    # the production key.
    dim = layer.conv_weights.size(0)
    x = torch.zeros((1, dim), dtype=conv_state.dtype, device=device).t()

    for qsl_offset, ci_offset, bool_offset in _CONV1D_WARMUP_SLICE_OFFSETS:
        query_start_loc = torch.zeros(
            qsl_offset + 2, dtype=torch.int32, device=device
        )[qsl_offset:]
        query_start_loc[1] = 1
        cache_indices = torch.full(
            (ci_offset + 1,), NULL_BLOCK_ID, dtype=torch.int32, device=device
        )[ci_offset:]
        has_initial_state = torch.zeros(
            bool_offset + 1, dtype=torch.bool, device=device
        )[bool_offset:]

        causal_conv1d_fn(
            x,
            layer.conv_weights,
            layer.conv1d.bias,
            activation=layer.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            block_size_to_align=mamba_block_size,
            metadata=None,
            query_start_loc=query_start_loc,
        )
    return True


@torch.inference_mode()
def hybrid_mamba_triton_warmup(
    runner: "GPUModelRunner",
    model_config: object,
) -> None:
    """Warm hybrid-Mamba2 Triton kernels reported by the JIT monitor."""
    if runner.is_pooling_model:
        return

    compilation_config = getattr(runner, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    mixer_layers = list(_iter_mamba_mixer2_layers(static_forward_context))
    if not mixer_layers:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    logger.info("Warming up hybrid Mamba2 Triton kernels.")

    # Zero-KV-blocks kernel: hybrid models always need KV block zeroing
    # (KVCacheConfig.needs_kv_cache_zeroing is True when mamba layers exist).
    zero_config = _zero_kv_warmup_config(runner)
    warmed_zeroer = _warm_zero_kv_blocks_with_runner_zeroer(runner)
    if zero_config is not None:
        _warm_zero_kv_blocks_kernel(device, zero_config)
    elif not warmed_zeroer:
        logger.info("Skipping hybrid zero-kv warmup: no KVBlockZeroer metadata.")

    # Slot-mapping kernel: covers block_table_stride == 1, which hybrid
    # models hit (large mamba-aligned attention block size -> one block per
    # request) and the generic block-table warmup does not reach.
    _warm_compute_slot_mapping_kernel(device)

    # Prefill causal-conv1d kernel: warm one layer per distinct JIT key.
    seen_keys: set[tuple] = set()
    warmed_any = False
    for layer in mixer_layers:
        conv_state = _get_conv_state(layer)
        if conv_state is None:
            continue
        key = (
            layer.conv_weights.size(0),
            layer.conv_weights.size(1),
            conv_state.dtype,
            conv_state.size(0),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _warm_mamba2_causal_conv1d_fwd(device, layer):
            warmed_any = True
    if not warmed_any:
        logger.info(
            "Skipping hybrid causal-conv1d warmup: no bound Mamba2 conv cache found."
        )

    _synchronize_device(device)
