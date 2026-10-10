# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of the TPSP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from vllm.logger import init_logger
from vllm.model_executor.tpsp import TPSPBackend

logger = init_logger(__name__)


@dataclass
class CudaTPSPContext:
    device: torch.device
    tp_size: int
    rank: int
    comm_address: int
    max_chunk_rows: int
    p2p_handle: int = 0
    config: int | None = None


class CudaTPSPBackend(TPSPBackend):
    tpsp_chunk_granularity = 64
    tpsp_max_microchunk_tokens = 16_384

    def __init__(self, group_name: str, device: torch.device):
        super().__init__(group_name, device)
        self._open_context_ids: set[int] = set()

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
        max_chunk_rows = (
            min(max_batched_tokens, self.tpsp_max_microchunk_tokens) + tp_size - 1
        ) // tp_size
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
        )
        if rank == 0:
            logger.info(
                "TPSP transport=%s for %s rows per rank",
                "P2P" if context.p2p_handle else "NCCL",
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
        if type(config) is not int or config <= 0:
            raise ValueError("CUDA TPSP requires a positive microchunk size")
        context.config = config

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
        chunk = context.config if config is None else config
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("CUDA TPSP requires a positive microchunk size")
        if (
            context.p2p_handle
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
                context.p2p_handle,
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
