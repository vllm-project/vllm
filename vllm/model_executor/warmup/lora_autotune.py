# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA Triton autotuning table."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.lora.layers import LogitsProcessorWithLoRA, MergedColumnParallelLinearWithLoRA
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.ops.triton_ops.lora_tuning import (
    LoRATuningCase,
    lora_m_buckets,
    lora_scratch_bytes,
    tune_lora_case,
)
from vllm.lora.ops.triton_ops.utils import (
    get_lora_op_config_file_name,
    load_lora_autotune_configs,
    save_lora_configs,
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

TableKeys = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class LoRAShape:
    op_type: str  # "shrink" or "expand"
    max_loras: int
    rank: int
    slice_sizes: tuple[int, ...]
    add_inputs: bool | None  # expand only
    dtype: torch.dtype

    @property
    def file_key(self) -> tuple[str, bool | None]:
        return self.op_type, self.add_inputs

    def table_keys(self, m: int) -> TableKeys:
        """Return this shape's nested config-table keys."""
        hidden_size = max(self.slice_sizes)
        k, n = (
            (hidden_size, self.rank)
            if self.op_type == "shrink"
            else (self.rank, hidden_size)
        )
        return (
            str(self.max_loras + 1),
            str(len(self.slice_sizes)),
            str(m),
            str(k),
            str(n),
        )


def _expand_add_inputs(module: BaseLinearLayerWithLoRA) -> bool:
    # Shared dual-stream paths expand into a separate buffer.
    shared_apply = type(module).apply in (
        BaseLinearLayerWithLoRA.apply,
        MergedColumnParallelLinearWithLoRA.apply,
    )
    return not (module._enable_aux_cuda_stream and shared_apply)


def discover_lora_shapes(worker: Worker) -> dict[LoRAShape, int]:
    """Return LoRA kernel shapes and maximum token counts."""
    max_loras = worker.vllm_config.lora_config.max_loras
    max_tokens = worker.scheduler_config.max_num_batched_tokens
    spec_config = worker.vllm_config.speculative_config
    num_spec_tokens = (spec_config.num_speculative_tokens or 0) if spec_config else 0
    max_logits = worker.scheduler_config.max_num_seqs * (1 + num_spec_tokens)

    shapes: dict[LoRAShape, int] = {}
    for module in worker.get_model().modules():
        if isinstance(module, BaseLinearLayerWithLoRA):
            lora_a, lora_b = module.lora_a_stacked, module.lora_b_stacked
            limit, add_inputs = max_tokens, _expand_add_inputs(module)
        elif isinstance(module, LogitsProcessorWithLoRA):
            lora_a, lora_b = (module.lora_a_stacked,), (module.lora_b_stacked,)
            limit, add_inputs = max_logits, True
        else:
            continue
        dtype = lora_a[0].dtype
        shrink = LoRAShape(
            "shrink",
            max_loras,
            rank=lora_a[0].shape[-2],
            slice_sizes=tuple(a.shape[-1] for a in lora_a),
            add_inputs=None,
            dtype=dtype,
        )
        expand = LoRAShape(
            "expand",
            max_loras,
            rank=lora_b[0].shape[-1],
            slice_sizes=tuple(b.shape[-2] for b in lora_b),
            add_inputs=add_inputs,
            dtype=dtype,
        )
        for shape in (shrink, expand):
            shapes[shape] = max(shapes.get(shape, 0), limit)
    return shapes


def _get_entry(table: dict[str, Any], keys: TableKeys) -> Any:
    for key in keys:
        if not isinstance(table, dict) or key not in table:
            return None
        table = table[key]
    return table


class LoRAConfigTable(TunableConfigTable):
    name = "lora"

    def __init__(self) -> None:
        # Starting points from buckets tuned earlier.
        self._tuned: dict[tuple[LoRAShape, int], Config] = {}

    def pending_items(self, worker: Worker) -> list[TuningItem]:
        if not current_platform.is_cuda_alike():
            return []
        # Batch-invariant mode always runs the default configs.
        if worker.vllm_config.lora_config is None or envs.VLLM_BATCH_INVARIANT:
            return []
        user_folder = envs.VLLM_TUNED_CONFIG_FOLDER
        force = envs.VLLM_TRITON_AUTOTUNE_FORCE
        cached: dict[tuple[str, bool | None], dict[str, Any]] = {}
        items: list[TuningItem] = []
        for shape, max_tokens in discover_lora_shapes(worker).items():
            file_key = shape.file_key
            # A user-supplied file replaces the autotune cache for its op.
            has_user_file = user_folder is not None and os.path.exists(
                os.path.join(user_folder, get_lora_op_config_file_name(*file_key))
            )
            if file_key not in cached:
                cached[file_key] = load_lora_autotune_configs(*file_key)
            configs = cached[file_key]
            for m in lora_m_buckets(max_tokens):
                if not force and (
                    has_user_file
                    or _get_entry(configs, shape.table_keys(m)) is not None
                ):
                    continue
                items.append(TuningItem(self.name, shape, m))
        return items

    def tune(self, item: TuningItem) -> Config | None:
        shape = item.shape
        case = LoRATuningCase(
            op_type=shape.op_type,
            M=item.bucket,
            max_loras=shape.max_loras,
            rank=shape.rank,
            slice_sizes=shape.slice_sizes,
            add_inputs=shape.add_inputs,
            dtype=shape.dtype,
            device=torch.device(current_platform.device_type),
        )
        free_bytes, _ = torch.accelerator.get_memory_info()
        if lora_scratch_bytes(case) > free_bytes // 2:
            logger.warning("Skipping LoRA tuning for %s: low free memory.", item)
            return None
        # Transfer configs from nearby buckets with the same search axes.
        num_slices = len(shape.slice_sizes)
        nearby = sorted(
            (
                key
                for key in self._tuned
                if key[0].op_type == shape.op_type
                and len(key[0].slice_sizes) == num_slices
            ),
            key=lambda key: (abs(math.log(key[1] / item.bucket)), key[0] != shape),
        )[:2]
        config = tune_lora_case(case, seeds=[self._tuned[key] for key in nearby])
        if config is not None:
            self._tuned[shape, item.bucket] = config
        return config

    def commit(self, results: dict[TuningItem, Config]) -> None:
        by_file: dict[tuple[str, bool | None], dict[TableKeys, Config]] = defaultdict(
            dict
        )
        for item, config in results.items():
            shape = item.shape
            by_file[shape.file_key][shape.table_keys(item.bucket)] = config
        for file_key, configs in by_file.items():
            merged = load_lora_autotune_configs(*file_key)
            for keys, config in configs.items():
                node = merged
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = config
            path = save_lora_configs(*file_key, merged)
            logger.info("Saved %d LoRA configs to %s.", len(configs), path)
