# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trigger native FlashInfer ReplaySSM tuning before CUDA graph capture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    selective_state_update_replayssm_flashinfer,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _find_replayssm_layers(runner: GPUModelRunner) -> tuple[MambaMixer2, ...]:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    return tuple(
        module
        for module in runner.get_model().modules()
        if (
            isinstance(module, MambaMixer2)
            and module.use_replayssm
            and module.mamba_config.backend == MambaBackendEnum.FLASHINFER
        )
    )


def _layer_signature(layer: MambaMixer2) -> tuple[Any, ...]:
    _, state, x_cache, dt_cache, B_cache, *_ = layer.kv_cache
    tensors = (state, x_cache, dt_cache, B_cache, layer.A, layer.D, layer.dt_bias)
    return tuple(
        (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        for tensor in tensors
    ) + (
        layer.mamba_config.enable_stochastic_rounding,
        layer.mamba_config.stochastic_rounding_philox_rounds,
    )


def _uniform_decode_batches(runner: GPUModelRunner) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                desc.num_reqs
                for _, descs in runner.cudagraph_dispatcher.get_capture_descs()
                for desc in descs
                if desc.uniform and desc.num_reqs is not None
            }
        )
    )


def _local_max_tuning_batch(runner: GPUModelRunner, state_capacity: int) -> int:
    capture_batches = _uniform_decode_batches(runner)
    requested = (
        capture_batches[-1] if capture_batches else runner.scheduler_config.max_num_seqs
    )
    return max(
        0,
        min(
            requested,
            runner.scheduler_config.max_num_seqs,
            state_capacity - 1,
        ),
    )


def _distributed_min(value: int) -> int:
    world = get_world_group()
    if world.world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=world.cpu_group,
    )
    return int(tensor.item())


def _distributed_layers_are_compatible(
    layers: tuple[MambaMixer2, ...],
) -> bool:
    world = get_world_group()
    local_signature = _layer_signature(layers[0]) if layers else None
    local_homogeneous = bool(layers) and all(
        _layer_signature(layer) == local_signature for layer in layers[1:]
    )
    reference = world.broadcast_object(
        local_signature if world.rank_in_group == 0 else None,
        src=0,
    )
    local_ok = local_homogeneous and local_signature == reference
    if world.world_size == 1:
        return local_ok
    flag = torch.tensor([int(local_ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag,
        op=torch.distributed.ReduceOp.MIN,
        group=world.cpu_group,
    )
    return bool(flag.item())


def _empty_preserve_strides(tensor: torch.Tensor, cache_capacity: int) -> torch.Tensor:
    return torch.empty_strided(
        (cache_capacity, *tensor.shape[1:]),
        tensor.stride(),
        dtype=tensor.dtype,
        device=tensor.device,
    )


class _ReplaySSMTuningCall:
    """A private, production-layout ReplaySSM invocation for native tuning."""

    def __init__(self, layer: MambaMixer2, batch: int):
        _, live_state, live_x_cache, live_dt_cache, live_B_cache, *_ = layer.kv_cache
        if batch <= 0 or batch >= live_state.shape[0]:
            raise ValueError(
                f"ReplaySSM tuning batch {batch} needs {batch + 1} state slots, "
                f"but only {live_state.shape[0]} are available"
            )

        # Preserve production inner shapes and every stride. Native FlashInfer
        # treats cache capacity as a constrained dimension, so only the active
        # slots plus the reserved padding slot need private storage.
        private_capacity = batch + 1
        self.state = _empty_preserve_strides(live_state, private_capacity)
        self.x_cache = _empty_preserve_strides(live_x_cache, private_capacity)
        self.dt_cache = _empty_preserve_strides(live_dt_cache, private_capacity)
        self.B_cache = _empty_preserve_strides(live_B_cache, private_capacity)
        for tensor in (self.state, self.x_cache, self.dt_cache, self.B_cache):
            tensor[: batch + 1].zero_()

        device = self.state.device
        activation_dtype = self.x_cache.dtype
        nheads = self.state.shape[1]
        headdim = self.state.shape[2]
        dstate = self.state.shape[3]
        ngroups = self.B_cache.shape[1]
        self.batch = batch
        self.logical_window = self.x_cache.shape[2] - 1
        if self.logical_window <= 0:
            raise ValueError("ReplaySSM history window must be positive")

        self.ring_start = torch.zeros(
            private_capacity, dtype=torch.int32, device=device
        )
        self.prev_num_accepted = torch.zeros_like(self.ring_start)
        rows = torch.arange(batch, dtype=torch.int32, device=device)
        self.ring_start[1 : batch + 1] = rows.remainder(self.logical_window + 1)
        self.prev_num_accepted[1 : batch + 1] = rows.remainder(
            self.logical_window
        ).add_(1)
        self.indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)

        self.x = torch.zeros(
            batch, nheads, headdim, dtype=activation_dtype, device=device
        )
        dt_base = torch.zeros(batch, nheads, dtype=activation_dtype, device=device)
        self.dt = dt_base.unsqueeze(-1).expand(batch, nheads, headdim)
        self.B = torch.zeros(
            batch, ngroups, dstate, dtype=activation_dtype, device=device
        )
        self.C = torch.zeros_like(self.B)
        self.out = torch.empty_like(self.x)
        self.A = (
            layer.A[:, None, ...][:, :, None]
            .expand(-1, headdim, dstate)
            .to(dtype=torch.float32)
        )
        self.D = layer.D[:, None, ...].expand(-1, headdim)
        self.dt_bias = layer.dt_bias[:, None, ...].expand(-1, headdim)
        self.rand_seed = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if layer.mamba_config.enable_stochastic_rounding
            else None
        )
        self.philox_rounds = layer.mamba_config.stochastic_rounding_philox_rounds or 10

        k_old = ((self.logical_window + 7) // 8) * 8
        self.cb_scaled = torch.empty(
            batch, nheads, 32, 8, dtype=activation_dtype, device=device
        )
        self.cumAdt_vec = torch.empty(
            batch, nheads, 16, dtype=torch.float32, device=device
        )
        self.cb_old = torch.empty(
            batch,
            nheads,
            32,
            k_old // 2,
            dtype=activation_dtype,
            device=device,
        )

    def run(self) -> None:
        selective_state_update_replayssm_flashinfer(
            self.state,
            self.x,
            self.dt,
            self.A,
            self.B,
            self.C,
            self.out,
            self.x_cache,
            self.B_cache,
            self.dt_cache,
            self.ring_start,
            self.prev_num_accepted,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            state_batch_indices=self.indices,
            cb_scaled=self.cb_scaled,
            cumAdt_vec=self.cumAdt_vec,
            cb_old=self.cb_old,
            algorithm="auto",
        )


@torch.inference_mode()
def trigger_flashinfer_replayssm_autotune(runner: GPUModelRunner) -> None:
    """Make one maximum-batch call so FlashInfer tunes all decode buckets."""
    layers = _find_replayssm_layers(runner)
    if not _distributed_layers_are_compatible(layers):
        logger.warning_once(
            "Skipping native FlashInfer ReplaySSM autotuning because ReplaySSM "
            "layers are absent or have incompatible rank-local geometries."
        )
        return

    layer = layers[0]
    state_capacity = layer.kv_cache[1].shape[0]
    batch = _distributed_min(_local_max_tuning_batch(runner, state_capacity))
    if batch <= 0:
        logger.warning_once(
            "Skipping native FlashInfer ReplaySSM autotuning because no valid "
            "decode batch fits in the state cache."
        )
        return

    tuning_call = _ReplaySSMTuningCall(layer, batch)
    tuning_call.run()
    torch.cuda.synchronize()
    logger.info_once(
        "Triggered native FlashInfer ReplaySSM autotuning through batch %d.", batch
    )
