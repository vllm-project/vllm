# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level installation of the global expert pool.

Runs once after every layer's process_weights_after_loading: with
``moe_expert_pool_rows > 0`` the MoE layers keep their expert tensors in
pinned host memory (final kernel layout); this allocates one VRAM bank
shared by all of them, fills each layer's initial rows, and binds a
consumer (Marlin) to the bank.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_pool.copy import configure_copy
from vllm.model_executor.layers.fused_moe.expert_pool.layer import PoolLayer
from vllm.model_executor.layers.fused_moe.expert_pool.pool import GlobalPool
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    TENSORS,
    allocate_step_buffers,
    set_control,
)

logger = init_logger(__name__)


# Decode lanes (tokens x top_k) the step program is compiled for. Wider
# inputs take the bank + host-view partition path. Small on purpose: the
# single-program planner loops over WIDTH lanes and scans the pool per miss.
MAX_DECODE_LANES = 64


def _next_power_of_two(value: int) -> int:
    return 1 << max(int(value) - 1, 0).bit_length()


def _check_top_k(top_k: int) -> None:
    if not 0 < top_k <= MAX_DECODE_LANES:
        raise ValueError(
            f"expert pool supports 0 < top_k <= {MAX_DECODE_LANES}, got {top_k}"
        )


def check_pool_layers(layers: list[tuple[str, torch.nn.Module]]) -> None:
    """Reject geometries the pool cannot serve, before anything is allocated:
    every layer must share expert count, top-k, resident rows and backend,
    and the backend must be NVFP4 Marlin (the only consumer bound here)."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )

    first_name, first = layers[0]
    for name, layer in layers:
        method = layer.quant_method
        if not isinstance(method, ModelOptNvFp4FusedMoE):
            raise ValueError(
                f"{name}: expert pool supports ModelOptNvFp4FusedMoE only, got "
                f"{type(method).__name__}"
            )
        if method.nvfp4_backend != NvFp4MoeBackend.MARLIN:
            raise ValueError(
                f"{name}: expert pool supports the Marlin NVFP4 backend only, "
                f"got {method.nvfp4_backend.value}"
            )
        if layer.moe_config.moe_parallel_config.use_ep:
            raise ValueError(f"{name}: expert pool is not compatible with EP")
        for attr in ("local_num_experts", "_moe_expert_pool_rows"):
            if getattr(layer, attr) != getattr(first, attr):
                raise ValueError(
                    f"{name}.{attr}={getattr(layer, attr)} differs from "
                    f"{first_name} ({getattr(first, attr)})"
                )
        if layer.moe_config.experts_per_token != first.moe_config.experts_per_token:
            raise ValueError(f"{name}: top_k differs from {first_name}")


def pool_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if getattr(module, "expert_pool_pending", False)
    ]


def _sources(name: str, layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    sources = {}
    for tensor_name in TENSORS:
        parameter = getattr(layer, tensor_name, None)
        if parameter is None:
            raise RuntimeError(f"{name}.{tensor_name}: missing for the expert pool")
        t = parameter.data
        if t.ndim < 1 or t.shape[0] != layer.local_num_experts:
            raise RuntimeError(
                f"{name}.{tensor_name}: the expert pool needs one row per expert, "
                f"got shape {tuple(t.shape)}"
            )
        if t.device.type != "cpu" or not t.is_pinned() or not t.is_contiguous():
            # The per-expert global scales are small and are not allocated in
            # host memory by create_weights, so the loader leaves them on the
            # device after conversion; take a pinned, dense host copy as the
            # source (explicit contiguous layout: empty_like would keep the
            # input's strides).
            t = torch.empty(
                t.shape, dtype=t.dtype, device="cpu", pin_memory=True
            ).copy_(t)
        assert t.is_contiguous() and t.is_pinned()
        sources[tensor_name] = t
    return sources


def install_expert_pool(
    model: torch.nn.Module, device: torch.device, max_decode_tokens: int = 1
) -> GlobalPool | None:
    """Allocate the shared bank and bind every pending pool layer to it."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    layers = pool_layers(model)
    if not layers:
        return None
    check_pool_layers(layers)
    first_name, first = layers[0]
    num_experts = first.local_num_experts
    top_k = first.moe_config.experts_per_token
    _check_top_k(top_k)
    slots = min(first._moe_expert_pool_rows, num_experts - 1)
    if slots < top_k:
        raise ValueError(
            f"expert pool needs at least top_k={top_k} rows per layer, got {slots}"
        )
    # Decode lanes served by the step program: at most MAX_DECODE_LANES and
    # at least one token; batches beyond that use the partition path.
    decode_tokens = max(1, min(max_decode_tokens, MAX_DECODE_LANES // top_k))
    staging = top_k * decode_tokens
    width = _next_power_of_two(staging)
    sources = [_sources(name, layer) for name, layer in layers]
    for name, src in zip((n for n, _ in layers), sources):
        for tensor_name in TENSORS:
            if (
                src[tensor_name].shape[1:] != sources[0][tensor_name].shape[1:]
                or src[tensor_name].dtype != sources[0][tensor_name].dtype
            ):
                raise RuntimeError(
                    f"{name}.{tensor_name}: layer rows differ from {first_name}"
                )
    pool = GlobalPool(device, sources[0], [slots] * len(layers), staging)
    configure_copy("chunks")
    for index, ((name, layer), src) in enumerate(zip(layers, sources)):
        method = layer.quant_method
        start = pool.offset(index)
        for tensor_name in TENSORS:
            pool.bank[tensor_name][start : start + slots].copy_(
                src[tensor_name][:slots], non_blocking=True
            )
        host_views = {
            tensor_name: get_accelerator_view_from_cpu_tensor(t)
            for tensor_name, t in src.items()
        }
        proxy = SimpleNamespace(
            **{tensor_name: pool.bank[tensor_name] for tensor_name in TENSORS},
            w13_input_scale=None,
            w2_input_scale=None,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            swiglu_alpha=getattr(layer, "swiglu_alpha", None),
            swiglu_beta=getattr(layer, "swiglu_beta", None),
        )
        quant = method.get_fused_moe_quant_config(proxy)
        config = replace(method.moe, num_local_experts=pool.rows)
        kernel = make_nvfp4_moe_kernel(
            quant,
            config,
            method.experts_cls,
            method.nvfp4_backend,
            routing_tables=None,
        )
        if kernel.prepare_finalize.supports_async():
            raise NotImplementedError(
                "expert pool requires synchronous prepare/finalize"
            )
        layer.expert_pool_layer = PoolLayer(
            index=index,
            pool=pool,
            slots=slots,
            sources=src,
            host_views=host_views,
            buffers=allocate_step_buffers(device, num_experts, width),
            experts=kernel.fused_experts,
            marlin_workspace=marlin_make_workspace_new(device, 4),
            num_experts=num_experts,
            top_k=top_k,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
        layer.expert_pool_pending = False
    # Placement policy (the lab run's values): promotions on every forward,
    # first miss promotes, no protection window. The gate stays closed
    # through profiling and graph capture (dummy routing must not move the
    # placement) and is opened by open_pool_gate() at the end of warm-up.
    set_control(
        pool.tables,
        promote_limit=0,
        promote_interval=1,
        promote_min_misses=1,
        protect_recent=0,
        gate=0,
    )
    torch.accelerator.synchronize(device)
    model.expert_pool = pool
    model.expert_pool_sources = sources
    logger.info(
        "Expert pool installed: %d layers, %d/%d rows per layer resident, "
        "%d staging rows (%d decode tokens x top_k %d; wider batches take the "
        "partition path), bank %.1f GiB (%s)",
        len(layers),
        slots,
        num_experts,
        staging,
        decode_tokens,
        top_k,
        (pool.pool_bytes + pool.staging_bytes) / 2**30,
        type(layers[0][1].quant_method).__name__,
    )
    return pool


def open_pool_gate(model: torch.nn.Module, sample_rows: int = 4) -> None:
    """End of warm-up/capture: verify the tables and a sample of bank rows
    against the host source, log the placement, then open the gate.

    Host readback happens here only, never inside a forward. The gate is a
    device scalar at a fixed address, so captured graphs see the change."""
    from vllm.model_executor.layers.fused_moe.expert_pool.pool import (
        verify_bank_rows,
    )
    from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
        check_global_tables,
        resident_per_layer,
        set_gate,
    )

    pool = getattr(model, "expert_pool", None)
    if pool is None:
        return
    device = pool.tables.hot_phys.device
    torch.accelerator.synchronize(device)
    check_global_tables(pool.tables)
    report = verify_bank_rows(pool, model.expert_pool_sources, sample_rows)
    resident = resident_per_layer(pool.tables)
    set_gate(pool.tables, True)
    torch.accelerator.synchronize(device)
    logger.info(
        "Expert pool gate opened after warm-up: tables consistent, %d sampled "
        "bank rows match the host source (%d resident), resident per layer "
        "min/max %d/%d",
        report["rows_checked"],
        report["rows_resident"],
        min(resident),
        max(resident),
    )
