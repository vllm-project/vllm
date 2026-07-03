# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)

_ZERO_KV_N_BLOCKS = (1, 2)

_SLOT_MAPPING_KV_BLOCK_SIZE = 16
_SLOT_MAPPING_CP_KV_CACHE_INTERLEAVE_SIZE = 1
_SLOT_MAPPING_BLOCK_TABLE_STRIDES = (1, 3)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)


@dataclass(frozen=True)
class _ZeroKvWarmupConfig:
    page_size_el: int
    block_size: int
    n_segs: int


@dataclass(frozen=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _get_kv_block_zeroer(runner: object) -> object | None:
    zeroer = getattr(runner, "kv_block_zeroer", None)
    if zeroer is None:
        zeroer = getattr(runner, "_kv_block_zeroer", None)
    return zeroer


def _zero_kv_warmup_config(runner: object) -> _ZeroKvWarmupConfig | None:
    zeroer = _get_kv_block_zeroer(runner)
    meta = getattr(zeroer, "_meta", None)
    if meta is None:
        return None

    _, page_size_el, block_size, n_segs = meta
    return _ZeroKvWarmupConfig(
        page_size_el=int(page_size_el),
        block_size=int(block_size),
        n_segs=int(n_segs),
    )


def _warm_zero_kv_blocks_with_runner_zeroer(runner: object) -> bool:
    zeroer = _get_kv_block_zeroer(runner)
    zero_block_ids = getattr(zeroer, "zero_block_ids", None)
    if not callable(zero_block_ids):
        return False

    for n_blocks in _ZERO_KV_N_BLOCKS:
        zero_block_ids(list(range(n_blocks)))
    return True


def _warm_zero_kv_blocks_kernel(
    device: torch.device, config: _ZeroKvWarmupConfig
) -> None:
    from vllm.v1.worker.utils import _zero_kv_blocks_kernel

    max_n_blocks = max(_ZERO_KV_N_BLOCKS)
    scratch = torch.empty(
        max_n_blocks * config.page_size_el,
        dtype=torch.int32,
        device=device,
    )
    seg_addrs = torch.tensor(
        [scratch.data_ptr()] * config.n_segs,
        dtype=torch.uint64,
        device=device,
    )

    for n_blocks in _ZERO_KV_N_BLOCKS:
        block_ids = torch.arange(n_blocks, dtype=torch.int64, device=device)
        grid = (n_blocks * config.n_segs * (config.page_size_el // config.block_size),)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            block_ids,
            n_blocks,
            N_SEGS=config.n_segs,
            PAGE_SIZE_EL=config.page_size_el,
            BLOCK_SIZE=config.block_size,
        )


def _warm_compute_slot_mapping_kernel(device: torch.device) -> None:
    from vllm.v1.worker.block_table import BlockTable

    # num_tokens/max_num_tokens are do_not_specialize; keep the launch tiny.
    num_tokens = 1
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)

    for block_table_stride in _SLOT_MAPPING_BLOCK_TABLE_STRIDES:
        # Use BlockTable so the JIT key matches the production slot-mapping call.
        block_table = BlockTable(
            block_size=_SLOT_MAPPING_KV_BLOCK_SIZE,
            max_num_reqs=1,
            max_num_blocks_per_req=block_table_stride,
            max_num_batched_tokens=num_tokens,
            pin_memory=False,
            device=device,
            kernel_block_size=_SLOT_MAPPING_KV_BLOCK_SIZE,
            cp_kv_cache_interleave_size=_SLOT_MAPPING_CP_KV_CACHE_INTERLEAVE_SIZE,
        )
        block_table.add_row(list(range(block_table_stride)), 0)
        block_table.commit_block_table(num_reqs=1)
        block_table.compute_slot_mapping(1, query_start_loc, positions)


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=False,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: object,
) -> None:
    """Warm Qwen Triton kernels reported by the JIT monitor."""
    if runner.is_pooling_model:
        return

    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = None
    for config in (hf_text_config, hf_config):
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            model_type = str(model_type)
            break
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = getattr(runner, "device", torch.device("cuda"))
    logger.info("Warming up Qwen Triton kernels for model_type=%s.", model_type)

    zero_config = _zero_kv_warmup_config(runner)
    warmed_zeroer = _warm_zero_kv_blocks_with_runner_zeroer(runner)
    if zero_config is not None:
        _warm_zero_kv_blocks_kernel(device, zero_config)
    else:
        logger.info("Skipping Qwen zero-kv warmup: no KVBlockZeroer metadata.")

    _warm_compute_slot_mapping_kernel(device)
    _synchronize_device(device)

    compilation_config = getattr(runner, "compilation_config", None)
    static_forward_context = getattr(compilation_config, "static_forward_context", None)
    gdn_config = _qwen_gdn_warmup_config(static_forward_context)
    if gdn_config is None:
        return

    _warm_causal_conv1d_fwd_kernel(device, gdn_config)
    _warm_fused_post_conv_kernel(device, gdn_config)
    _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
    _synchronize_device(device)
