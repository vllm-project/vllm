# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.nvlink_fabric import (
    SymmetricMemoryTopology,
    get_symmetric_memory_topology,
)
from vllm.logger import init_logger
from vllm.models.common.ops.sequence_parallel import sp_reduce_scatter

from ..quantization import bf16_mxfp8_linear_method_cls, mxfp8_linear_as_bf16

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear import LinearBase

logger = init_logger(__name__)

_MIN_TOKENS = 1024


class HopperWoBGemmRs:
    """Pipeline the BF16 projection with the TP reduce-scatter on SM90."""

    def __init__(self, group_name: str, tp_size: int) -> None:
        self.group_name = group_name
        self.tp_size = tp_size

    @classmethod
    def try_init(cls, config: "VllmConfig") -> "HopperWoBGemmRs | None":
        if (
            config.model_config.dtype != torch.bfloat16
            or config.parallel_config.use_ubatching
            or envs.VLLM_BATCH_INVARIANT
            or not envs.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD
            or not mxfp8_linear_as_bf16()
        ):
            return None

        from torch.distributed import _symmetric_memory as symm_mem

        if not hasattr(torch.ops.symm_mem, "fused_matmul_reduce_scatter"):
            return None
        tp_group = get_tp_group()
        if (
            get_symmetric_memory_topology(tp_group.cpu_group)
            is not SymmetricMemoryTopology.INTRA_NODE
        ):
            return None
        group_name = tp_group.device_group.group_name
        try:
            # Probe the workspace collectively during model initialization.
            symm_mem.get_symm_mem_workspace(group_name, min_size=1)
        except RuntimeError as exc:
            logger.warning_once("Hopper wo_b GEMM-RS is unavailable: %s", exc)
            return None
        logger.info_once("Hopper wo_b GEMM-RS is enabled.")
        return cls(group_name, tp_group.world_size)

    def can_run(self, layer: "LinearBase") -> bool:
        return (
            layer.bias is None
            and type(layer.quant_method) is bf16_mxfp8_linear_method_cls()
        )

    def warn_incompatible_projection(self) -> None:
        logger.warning_once("Hopper wo_b GEMM-RS requires load-time BF16 weights.")

    def should_run(self, x: torch.Tensor) -> bool:
        return (
            x.dtype == torch.bfloat16
            and x.shape[0] >= _MIN_TOKENS
            and x.shape[0] % self.tp_size == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def apply(self, x: torch.Tensor, layer: "LinearBase") -> torch.Tensor:
        if layer.weight.dtype != torch.bfloat16:
            return sp_reduce_scatter(layer(x))
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x, layer.weight.t(), "sum", 0, self.group_name
        )
