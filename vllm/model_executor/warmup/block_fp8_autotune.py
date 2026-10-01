# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 block FP8 Triton GEMM launch configs as a TunableConfigTable."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.quantization.utils.fp8_block_tuning import (
    BlockFp8TuningCase,
    block_fp8_scratch_bytes,
    tune_block_fp8_case,
    valid_m_buckets,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_w8a8_block_fp8_configs,
    load_block_fp8_autotune_configs,
    save_block_fp8_configs,
)
from vllm.model_executor.warmup.triton_autotune import (
    Config,
    TunableConfigTable,
    TuningItem,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlockFp8Shape:
    N: int
    K: int
    block_n: int
    block_k: int
    out_dtype: torch.dtype

    @property
    def file_key(self) -> tuple[int, int, int, int]:
        return self.N, self.K, self.block_n, self.block_k


def _triton_block_fp8_kernel(
    module: torch.nn.Module,
) -> TritonFp8BlockScaledMMKernel | None:
    # Fp8LinearMethod and most schemes keep the kernel on the quant method;
    # compressed-tensors keeps it on the layer's scheme.
    for owner_name in ("quant_method", "scheme"):
        kernel = getattr(getattr(module, owner_name, None), "fp8_linear", None)
        if isinstance(kernel, TritonFp8BlockScaledMMKernel):
            return kernel
    return None


def discover_block_fp8_shapes(worker: Worker) -> set[BlockFp8Shape]:
    shapes: set[BlockFp8Shape] = set()
    for module in worker.get_model().modules():
        kernel = _triton_block_fp8_kernel(module)
        if kernel is None:
            continue
        N, K = module.weight.shape
        block_n, block_k = kernel.weight_group_shape
        shapes.add(BlockFp8Shape(N, K, block_n, block_k, kernel.config.out_dtype))
    return shapes


class BlockFp8ConfigTable(TunableConfigTable):
    name = "block_fp8"

    def pending_items(self, worker: Worker) -> list[TuningItem]:
        if not current_platform.is_cuda_alike():
            return []
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1250

            # gfx1250 runs a torch reference instead of the Triton kernel.
            if on_gfx1250():
                return []
        buckets = valid_m_buckets(worker.scheduler_config.max_num_batched_tokens)
        items = []
        for shape in discover_block_fp8_shapes(worker):
            has_config = get_w8a8_block_fp8_configs(*shape.file_key) is not None
            if has_config and not envs.VLLM_TRITON_AUTOTUNE_FORCE:
                continue
            items.extend(TuningItem(self.name, shape, m) for m in buckets)
        return items

    def tune(self, item: TuningItem) -> Config | None:
        shape = item.shape
        case = BlockFp8TuningCase(
            M=item.bucket,
            N=shape.N,
            K=shape.K,
            block_n=shape.block_n,
            block_k=shape.block_k,
            out_dtype=shape.out_dtype,
            device=torch.device(current_platform.device_type),
        )
        free_bytes, _ = torch.accelerator.get_memory_info()
        if block_fp8_scratch_bytes(case) > free_bytes // 2:
            logger.warning("Skipping block FP8 tuning for %s: low free memory.", item)
            return None
        return tune_block_fp8_case(case)

    def commit(self, results: dict[TuningItem, Config]) -> None:
        by_file: dict[tuple[int, int, int, int], dict[int, Config]] = defaultdict(dict)
        for item, config in results.items():
            by_file[item.shape.file_key][item.bucket] = config
        for file_key, configs in by_file.items():
            merged = {**load_block_fp8_autotune_configs(*file_key), **configs}
            path = save_block_fp8_configs(*file_key, merged)
            logger.info("Saved %d block FP8 configs to %s.", len(configs), path)
