# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of the TPSP backend."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from vllm.logger import init_logger
from vllm.model_executor.tpsp import TPSPBackend, TPSPContext, TPSPOpsGroup

logger = init_logger(__name__)

_O_PROJ_COMM_SM_MAX = 20
_DOWN_PROJ_COMM_SM_MAX = 10


def _compute_engine_count_min(group_name: str | None, sm_count: int) -> int:
    if group_name == "o_proj":
        reserved = min(sm_count * 50 // 100, _O_PROJ_COMM_SM_MAX)
    elif group_name == "down_proj":
        reserved = min(sm_count * 30 // 100, _DOWN_PROJ_COMM_SM_MAX)
    else:
        reserved = 0
    return sm_count - reserved


@dataclass(frozen=True)
class CudaTPSPConfig:
    chunk_rows: int
    use_p2p: bool
    sm_count_target: int = 0


@dataclass
class CudaTPSPContext:
    device: torch.device
    tp_size: int
    rank: int
    comm_address: int
    max_chunk_rows: int
    p2p_handle: int = 0
    config: int | CudaTPSPConfig | None = None


class CudaTPSPBackend(TPSPBackend):
    tpsp_chunk_granularity = 64
    tpsp_max_microchunk_rows = 4_096

    def __init__(self, group_name: str, device: torch.device):
        super().__init__(group_name, device)
        self._open_context_ids: set[int] = set()

    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        compute_engine_count_min: int | Mapping[str, int] | None = None,
        compute_engine_count_max: int | Mapping[str, int] | None = None,
        compute_engine_step_coarse: int = 6,
        compute_engine_step_fine: int = 1,
        chunk_min_max: tuple[int, int] | Mapping[str, tuple[int, int]] | None = None,
        chunk_step_coarse: int = 256,
        chunk_step_fine: int = 64,
    ) -> TPSPContext | None:
        from vllm.distributed.parallel_state import get_tp_group

        tp_size = get_tp_group().world_size
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        if compute_engine_count_min is None and compute_engine_count_max is None:
            compute_engine_count_min = {
                entry.name: _compute_engine_count_min(entry.name, sm_count)
                for entry in ops_groups
            }
            compute_engine_count_max = {entry.name: sm_count for entry in ops_groups}
        if chunk_min_max is None:
            chunk_min_max = self._chunk_rows_range(max_batched_tokens, tp_size)
        return super().profile(
            ops_groups,
            max_batched_tokens,
            compute_engine_count_min=compute_engine_count_min,
            compute_engine_count_max=compute_engine_count_max,
            compute_engine_step_coarse=compute_engine_step_coarse,
            compute_engine_step_fine=compute_engine_step_fine,
            chunk_min_max=chunk_min_max,
            chunk_step_coarse=chunk_step_coarse,
            chunk_step_fine=chunk_step_fine,
        )

    def _chunk_rows_range(
        self, max_batched_tokens: int, tp_size: int
    ) -> tuple[int, int]:
        shard_rows = math.ceil(max_batched_tokens / tp_size)
        max_chunk_rows = math.ceil(shard_rows / self.tpsp_chunk_granularity)
        max_chunk_rows *= self.tpsp_chunk_granularity
        cap_rows = self.get_max_microchunk_rows(tp_size)
        if cap_rows is not None:
            if cap_rows >= self.tpsp_chunk_granularity:
                cap_rows = (
                    cap_rows // self.tpsp_chunk_granularity
                ) * self.tpsp_chunk_granularity
            max_chunk_rows = min(max_chunk_rows, cap_rows)
        return min(self.tpsp_chunk_granularity, max_chunk_rows), max_chunk_rows

    def get_compute_engine_count(self, config: object) -> int | None:
        if isinstance(config, CudaTPSPConfig):
            return config.sm_count_target
        return None

    def get_default_compute_engine_count(self) -> int:
        return 0

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> CudaTPSPContext | None:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if (
            device != self.device
            or group_name != self.group_name
            or device.type != "cuda"
        ):
            return None
        if (
            dtype != torch.bfloat16
            or tp_size < 2
            or hidden_size <= 0
            or hidden_size % tp_size
            or max_batched_tokens <= 0
        ):
            logger.warning(
                "TPSP unavailable for dtype=%s tp_size=%s hidden_size=%s",
                dtype,
                tp_size,
                hidden_size,
            )
            return None
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::fused_add_rms_norm", device.type.upper()
        ):
            logger.warning("TPSP unavailable: %s fused_add_rms_norm is missing", device)
            return None
        if (
            device.index is not None
            and device.index >= torch.accelerator.device_count()
        ):
            logger.warning("CUDA TPSP device index is unavailable: %s", device.index)
            return None
        try:
            group = c10d._resolve_process_group(group_name)
        except (RuntimeError, ValueError) as exc:
            logger.warning("CUDA TPSP process group is unavailable: %s", exc)
            return None
        if dist.get_world_size(group) != tp_size:
            logger.warning("TPSP unavailable: group_name and tp_size disagree")
            return None
        if not all(
            hasattr(torch.ops._C, name)
            for name in (
                "tpsp_fused_matmul_reduce_scatter_norm_all_gather",
                "init_tpsp_p2p",
                "destroy_tpsp_p2p",
            )
        ):
            logger.warning("CUDA TPSP native operator is not built")
            return None
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.device_group is not group:
            logger.warning("CUDA TPSP group is not the active TP group")
            return None
        comm = getattr(tp_group.device_communicator, "pynccl_comm", None)
        nccl_ready = (
            isinstance(comm, PyNcclCommunicator)
            and comm.available
            and not comm.disabled
            and comm.comm.value is not None
        )
        if nccl_ready:
            assert isinstance(comm, PyNcclCommunicator)
            comm_address = comm.comm.value
            assert comm_address is not None
        else:
            comm_address = 0
        rank = dist.get_rank(group)
        max_chunk_rows = math.ceil(max_batched_tokens / tp_size)
        cap_rows = self.get_max_microchunk_rows(tp_size)
        if cap_rows is not None:
            max_chunk_rows = min(max_chunk_rows, cap_rows)
        context = CudaTPSPContext(device, tp_size, rank, comm_address, max_chunk_rows)
        if not self._all_ranks_ready(nccl_ready, group, device):
            logger.warning("CUDA TPSP default NCCL transport is unavailable")
            return None
        context.p2p_handle = torch.ops._C.init_tpsp_p2p(
            device.index if device.index is not None else torch.cuda.current_device(),
            comm_address,
            tp_size,
            rank,
            max_chunk_rows,
            hidden_size,
            max_batched_tokens,
        )
        if rank == 0:
            logger.info(
                "TPSP available transports=%s with direct P2P gather for "
                "%s rows per rank",
                "P2P,NCCL" if context.p2p_handle else "NCCL",
                max_chunk_rows,
            )
        self._open_context_ids.add(id(context))
        return context

    @staticmethod
    def _all_ranks_ready(
        ready: bool, group: dist.ProcessGroup, device: torch.device
    ) -> bool:
        supported = torch.tensor(int(ready), device=device)
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=group)
        return bool(supported.item())

    def _profile_context(self, context: object) -> CudaTPSPContext:
        if context is None:
            raise ValueError("CUDA TPSP requires a projection context")
        if (
            not isinstance(context, CudaTPSPContext)
            or id(context) not in self._open_context_ids
        ):
            raise ValueError("CUDA TPSP context belongs to another backend")
        return context

    def set_config(self, handle: object, config: object) -> None:
        context = self._profile_context(handle)
        context.config = self._resolve_config(context, config)

    def _resolve_config(
        self, context: CudaTPSPContext, config: object
    ) -> CudaTPSPConfig:
        if type(config) is int:
            config = CudaTPSPConfig(config, bool(context.p2p_handle))
        if (
            not isinstance(config, CudaTPSPConfig)
            or type(config.chunk_rows) is not int
            or config.chunk_rows <= 0
            or type(config.use_p2p) is not bool
            or type(config.sm_count_target) is not int
            or config.sm_count_target < 0
        ):
            raise ValueError("Invalid CUDA TPSP tuning configuration")
        if config.use_p2p and not context.p2p_handle:
            raise ValueError("CUDA TPSP P2P transport is unavailable")
        return config

    def get_tuning_configs(
        self,
        handle: object,
        chunk: int,
        group_name: str | None = None,
        *,
        compute_engine_counts: Sequence[int] | None = None,
        compute_engine_step: int = 6,
    ) -> tuple[CudaTPSPConfig, ...]:
        context = self._profile_context(handle)
        if compute_engine_step <= 0:
            raise ValueError("Compute-engine scan step must be positive")
        sm_count = torch.cuda.get_device_properties(
            context.device
        ).multi_processor_count
        targets: tuple[int, ...]
        if compute_engine_counts is not None:
            targets = tuple(dict.fromkeys(compute_engine_counts))
        else:
            start = _compute_engine_count_min(group_name, sm_count)
            values = list(range(start, sm_count + 1, compute_engine_step))
            if values[-1] != sm_count:
                values.append(sm_count)
            targets = tuple(dict.fromkeys((0, *values)))
        if not targets or any(
            type(target) is not int or not 0 <= target <= sm_count for target in targets
        ):
            raise ValueError("Invalid TPSP SM count targets")
        use_p2p = bool(context.p2p_handle)
        return tuple(CudaTPSPConfig(chunk, use_p2p, target) for target in targets)

    def fused_gemm_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: torch.nn.Module,
        residual: torch.Tensor,
        norm: torch.nn.Module,
        *,
        config: object | None = None,
        norm_type: str = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        context = self._profile_context(projection_context)
        hidden_size = norm.weight.numel()
        rows = (x.size(0) + context.tp_size - 1) // context.tp_size
        if residual.shape != (rows, hidden_size):
            raise RuntimeError("TPSP residual shard has an unexpected shape")
        local_residual = residual

        weight = projection.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(projection, "_tpsp_transposed_weight", None)
        if cached is None or cached[0] != key:
            cached = (key, weight.T.contiguous())
            projection._tpsp_transposed_weight = cached
        a, b = x.contiguous(), cached[1]
        norm_weight = norm.weight
        eps = norm.eps if norm_type == "layer_norm" else norm.variance_epsilon
        projection_bias = projection.bias
        norm_bias = getattr(norm, "bias", None)
        selected = self._resolve_config(
            context, context.config if config is None else config
        )
        chunk = selected.chunk_rows
        p2p_handle = context.p2p_handle if selected.use_p2p else 0
        if (
            p2p_handle
            and min((a.size(0) + context.tp_size - 1) // context.tp_size, chunk)
            > context.max_chunk_rows
        ):
            raise ValueError("CUDA TPSP microchunk exceeds P2P workspace capacity")
        if (
            norm_type not in ("rms_norm", "layer_norm")
            or eps <= 0
            or (norm_bias is not None and norm_type != "layer_norm")
        ):
            raise ValueError("Invalid CUDA TPSP normalization or chunk configuration")
        if (
            any(
                t.dtype != torch.bfloat16 or t.device != context.device
                for t in (a, b, norm_weight, local_residual)
            )
            or a.ndim != 2
            or b.ndim != 2
            or norm_weight.ndim != 1
            or local_residual.ndim != 2
            or a.shape[1] != b.shape[0]
            or b.shape[1] != norm_weight.numel()
        ):
            raise ValueError("CUDA TPSP requires compatible CUDA BF16 inputs")
        for bias in (projection_bias, norm_bias):
            if bias is not None and (
                bias.dtype != torch.bfloat16
                or bias.device != context.device
                or bias.shape != norm_weight.shape
                or not bias.is_contiguous()
            ):
                raise ValueError("CUDA TPSP bias must match the BF16 norm weight")
        reduced, _, gathered = (
            torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather(
                a,
                b,
                norm_weight,
                local_residual,
                projection_bias,
                norm_bias,
                eps,
                1 if norm_type == "layer_norm" else 0,
                chunk,
                context.comm_address,
                context.tp_size,
                p2p_handle,
                selected.sm_count_target,
            )
        )
        return gathered, reduced

    def close(self, context: CudaTPSPContext | None = None) -> None:
        if context is not None:
            context = self._profile_context(context)
            if context.p2p_handle:
                torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
                context.p2p_handle = 0
            self._open_context_ids.remove(id(context))
        else:
            self._closed = True
