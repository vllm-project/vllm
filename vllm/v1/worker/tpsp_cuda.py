# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 CUDA TPSP projection with optional bias and fused RMSNorm/LayerNorm."""

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d


class CudaTPSPOps:
    tpsp_chunk_granularity = 64

    def __init__(self, group: dist.ProcessGroup, device: torch.device, group_name: str):
        self.group = group
        self.device = device
        self.group_name = group_name
        self.tp_size = dist.get_world_size(group)
        self._closed = False

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        comm = getattr(tp_group.device_communicator, "pynccl_comm", None)
        if (
            tp_group.device_group is not group
            or not isinstance(comm, PyNcclCommunicator)
            or not comm.available
            or comm.disabled
            or comm.comm.value is None
        ):
            raise RuntimeError("CUDA TPSP requires the active TP PyNccl communicator")
        self.comm_address = comm.comm.value

    @classmethod
    def open(cls, group_name: str, device: torch.device) -> "CudaTPSPOps":
        if device.type != "cuda":
            raise ValueError("CUDA TPSP requires a CUDA device")
        group = c10d._resolve_process_group(group_name)
        if dist.get_backend(group) != "nccl":
            raise ValueError("CUDA TPSP requires an NCCL process group")
        if not hasattr(
            torch.ops._C, "tpsp_fused_matmul_reduce_scatter_norm_all_gather"
        ):
            raise RuntimeError("CUDA TPSP native operator is not built")
        return cls(group, device, group_name)

    def fused_matmul_reduce_scatter_norm_all_gather(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        _unused: None,
        group_name: str,
        *,
        eps: float,
        norm_type: str,
        residual: torch.Tensor,
        microchunk_tokens: int,
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("CUDA TPSP backend is closed")
        if group_name != self.group_name:
            raise ValueError("CUDA TPSP process group changed")
        if (
            norm_type not in ("rms_norm", "layer_norm")
            or microchunk_tokens <= 0
            or eps <= 0
            or (norm_bias is not None and norm_type != "layer_norm")
        ):
            raise ValueError("Invalid CUDA TPSP normalization or chunk configuration")
        if (
            any(
                t.dtype != torch.bfloat16 or t.device != self.device
                for t in (a, b, weight, residual)
            )
            or a.ndim != 2
            or b.ndim != 2
            or weight.ndim != 1
            or residual.ndim != 2
            or a.shape[1] != b.shape[0]
            or b.shape[1] != weight.numel()
        ):
            raise ValueError("CUDA TPSP requires compatible CUDA BF16 inputs")
        for bias in (projection_bias, norm_bias):
            if bias is not None and (
                bias.dtype != torch.bfloat16
                or bias.device != self.device
                or bias.shape != weight.shape
                or not bias.is_contiguous()
            ):
                raise ValueError("CUDA TPSP bias must match the BF16 norm weight")

        return torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather(
            a,
            b,
            weight,
            residual,
            projection_bias,
            norm_bias,
            eps,
            1 if norm_type == "layer_norm" else 0,
            microchunk_tokens,
            self.comm_address,
            self.tp_size,
        )

    def close_tpsp(self, group_name: str) -> None:
        if group_name != self.group_name:
            raise ValueError("CUDA TPSP process group changed")
        self._closed = True
