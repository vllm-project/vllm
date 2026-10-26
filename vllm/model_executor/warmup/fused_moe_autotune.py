# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton launch configs as a TunableConfigTable."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import (
    MOE_CONFIGS_DIR,
    get_config_file_name,
    load_moe_autotune_configs,
    save_moe_configs,
)
from vllm.model_executor.layers.fused_moe.moe_tuning import (
    FusedMoETuningCase,
    moe_scratch_bytes,
    tune_fused_moe_case,
    valid_m_buckets,
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

FileKey = tuple[int, int, str | None, tuple[int, int] | None]


@dataclass(frozen=True)
class FusedMoEShape:
    w1_shape: tuple[int, int, int]
    w2_shape: tuple[int, int, int]
    w1_stride: tuple[int, int, int]
    w2_stride: tuple[int, int, int]
    top_k: int
    dtype: torch.dtype
    use_fp8_w8a8: bool
    per_act_token_quant: bool
    block_shape: tuple[int, int] | None
    activation: MoEActivation

    @property
    def file_key(self) -> FileKey:
        """(E, N, dtype, block_shape), as try_get_optimal_moe_config looks it up."""
        dtype = "fp8_w8a8" if self.use_fp8_w8a8 else None
        return self.w2_shape[0], self.w2_shape[2], dtype, self.block_shape


def _block_shape_list(block_shape: tuple[int, int] | None) -> list[int] | None:
    # get_config_file_name formats the list; a tuple would print as (128,128).
    return list(block_shape) if block_shape is not None else None


def _plain_triton_experts(module: torch.nn.Module) -> TritonExperts | None:
    moe_kernel = getattr(getattr(module, "quant_method", None), "moe_kernel", None)
    experts = getattr(moe_kernel, "fused_experts", None)
    # Subclasses dequantize, emulate, or launch other kernels.
    return experts if type(experts) is TritonExperts else None


def discover_fused_moe_shapes(worker: Worker) -> list[FusedMoEShape]:
    """MoE shapes the model runs on the plain TritonExperts kernel, one per
    config file since lookups cannot tell shapes in the same file apart."""
    dtype = worker.model_config.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        return []
    shapes: dict[FileKey, FusedMoEShape] = {}
    for module in worker.get_model().modules():
        experts = _plain_triton_experts(module)
        w1 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if experts is None or w1 is None or w2 is None:
            continue
        quant_config = experts.quant_config
        unquantized = (
            quant_config.quant_dtype is None and quant_config.weight_quant_dtype is None
        )
        if not (unquantized or quant_config.use_fp8_w8a8):
            continue
        block_shape = quant_config.block_shape
        shape = FusedMoEShape(
            w1_shape=tuple(w1.shape),
            w2_shape=tuple(w2.shape),
            w1_stride=tuple(w1.stride()),
            w2_stride=tuple(w2.stride()),
            top_k=module.top_k,
            dtype=dtype,
            use_fp8_w8a8=quant_config.use_fp8_w8a8,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=tuple(block_shape) if block_shape is not None else None,
            activation=module.activation,
        )
        shapes.setdefault(shape.file_key, shape)
    return list(shapes.values())


def _has_fixed_config_file(file_key: FileKey) -> bool:
    E, N, dtype, block_shape = file_key
    file_name = get_config_file_name(E, N, dtype, _block_shape_list(block_shape))
    folders = [MOE_CONFIGS_DIR]
    if envs.VLLM_TUNED_CONFIG_FOLDER is not None:
        folders.append(envs.VLLM_TUNED_CONFIG_FOLDER)
    return any(os.path.exists(os.path.join(d, file_name)) for d in folders)


class FusedMoEConfigTable(TunableConfigTable):
    name = "fused_moe"

    def __init__(self) -> None:
        # Winners tuned on this rank, as starting points for nearby buckets.
        self._tuned: dict[FileKey, dict[int, Config]] = defaultdict(dict)

    def pending_items(self, worker: Worker) -> list[TuningItem]:
        if not current_platform.is_cuda_alike():
            return []
        # Batch-invariant mode always runs the default configs.
        if envs.VLLM_BATCH_INVARIANT:
            return []
        buckets = valid_m_buckets(worker.scheduler_config.max_num_batched_tokens)
        items = []
        for shape in discover_fused_moe_shapes(worker):
            E, N, dtype, block_shape = shape.file_key
            if envs.VLLM_TRITON_AUTOTUNE_FORCE:
                missing = buckets
            elif _has_fixed_config_file(shape.file_key):
                # Shapes with a user or bundled table keep it unless forced.
                continue
            else:
                tuned = load_moe_autotune_configs(
                    E, N, dtype, _block_shape_list(block_shape)
                )
                missing = [m for m in buckets if m not in tuned]
            items.extend(TuningItem(self.name, shape, m) for m in missing)
        return items

    def tune(self, item: TuningItem) -> Config | None:
        shape = item.shape
        case = FusedMoETuningCase(
            M=item.bucket,
            w1_shape=shape.w1_shape,
            w2_shape=shape.w2_shape,
            w1_stride=shape.w1_stride,
            w2_stride=shape.w2_stride,
            top_k=shape.top_k,
            dtype=shape.dtype,
            use_fp8_w8a8=shape.use_fp8_w8a8,
            per_act_token_quant=shape.per_act_token_quant,
            block_shape=shape.block_shape,
            activation=shape.activation,
            device=torch.device(current_platform.device_type),
        )
        free_bytes, _ = torch.accelerator.get_memory_info()
        if moe_scratch_bytes(case) > free_bytes // 2:
            logger.warning("Skipping fused MoE tuning for %s: low free memory.", item)
            return None
        tuned = self._tuned[shape.file_key]
        nearby = sorted(tuned, key=lambda m: abs(math.log(m / item.bucket)))[:2]
        config = tune_fused_moe_case(case, seeds=[tuned[m] for m in nearby])
        if config is not None:
            tuned[item.bucket] = config
        return config

    def commit(self, results: dict[TuningItem, Config]) -> None:
        by_file: dict[FileKey, dict[int, Config]] = defaultdict(dict)
        for item, config in results.items():
            by_file[item.shape.file_key][item.bucket] = config
        for (E, N, dtype, block_shape), configs in by_file.items():
            block_list = _block_shape_list(block_shape)
            merged = {**load_moe_autotune_configs(E, N, dtype, block_list), **configs}
            path = save_moe_configs(E, N, dtype, block_list, merged)
            logger.info("Saved %d fused MoE configs to %s.", len(configs), path)
