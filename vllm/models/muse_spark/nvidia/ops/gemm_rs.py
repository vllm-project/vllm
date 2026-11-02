# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic Muse Spark attention GEMM + reduce-scatter dispatch."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.common.ops.gemm_rs_ar import GemmRsAr

logger = init_logger(__name__)


class MuseSparkGemmRs:
    """One model-owned workspace, shared by sequential attention layers."""

    def __init__(self, kernel: "GemmRsAr") -> None:
        self.kernel = kernel

    @staticmethod
    def supports_projection(linear: LinearBase) -> bool:
        if not isinstance(linear.quant_method, UnquantizedLinearMethod):
            return False
        weight = linear.weight
        return (
            linear.bias is None
            and weight.dtype == torch.bfloat16
            and weight.is_cuda
            and weight.ndim == 2
            and weight.shape[0] == 8192
            and weight.shape[1] in (1024, 2048)
            and weight.is_contiguous()
        )

    def should_run(self, x: torch.Tensor) -> bool:
        # B300 TP4/TP8: separate GEMM + Lamport RS wins below M=512 and on
        # partial tiles (TP4 128 rows, TP8 256; TP8 M=640 also loses). Padded
        # shards are excluded because the kernel does not initialize them.
        return (
            x.ndim == 2
            and 512 <= x.shape[0] <= self.kernel.max_M
            and x.shape[0] % (128 if self.kernel.world_size == 4 else 256) == 0
            and x.shape[1] == 8192 // self.kernel.world_size
            and x.dtype == torch.bfloat16
            and x.device == self.kernel.device
            and x.is_contiguous()
        )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, weight)


def maybe_create_gemm_rs(
    vllm_config: VllmConfig, projections: list[LinearBase]
) -> MuseSparkGemmRs | None:
    """Collectively create a workspace for the measured B300 TP4/TP8 deployments."""
    pc = vllm_config.parallel_config
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability(103)
        or pc.tensor_parallel_size not in (4, 8)
        or pc.pipeline_parallel_size != 1
        or pc.use_ubatching
        or vllm_config.lora_config is not None
        or vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.scheduler_config.max_num_batched_tokens < 512
    ):
        return None
    tp_group = get_tp_group()
    eligible = torch.tensor(
        int(any(MuseSparkGemmRs.supports_projection(p) for p in projections))
    )
    dist.all_reduce(eligible, op=dist.ReduceOp.MIN, group=tp_group.cpu_group)
    if not eligible.item():
        return None
    if not all(in_the_same_node_as(tp_group.cpu_group, source_rank=0)):
        return None
    physical_id = current_platform.visible_device_id_to_physical_device_id(
        torch.accelerator.current_device_index()
    )
    physical_ids = [None] * tp_group.world_size
    dist.all_gather_object(physical_ids, physical_id, group=tp_group.cpu_group)
    if not current_platform.is_fully_connected(physical_ids):
        return None

    from vllm.models.common.ops.gemm_rs_ar import (
        GemmRsAr,
        Sm100GemmRsArBF16,
    )

    try:
        kernel = GemmRsAr(
            max_M=vllm_config.scheduler_config.max_num_batched_tokens, N=8192
        )
    except RuntimeError as exc:
        logger.warning_once(
            "Muse Spark GEMM-RS workspace unavailable; using separate GEMM "
            "and reduce-scatter: %s",
            exc,
        )
        return None
    # JIT-compile before CUDA graph capture, bypassing the shared singleton.
    for bn, cta_group in ((128, 1), (128, 2), (256, 2)):
        Sm100GemmRsArBF16.compile(kernel.rank, kernel.world_size, bn, cta_group)
    logger.info_once(
        "Muse Spark attention GEMM-RS enabled for TP%d BF16, aligned M >= 512.",
        pc.tensor_parallel_size,
    )
    return MuseSparkGemmRs(kernel)
