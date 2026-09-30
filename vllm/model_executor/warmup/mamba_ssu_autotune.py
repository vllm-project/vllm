# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba selective_state_update launch configs as a TunableConfigTable."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    get_ssm_configs,
    load_ssm_autotune_configs,
    save_ssm_configs,
)
from vllm.model_executor.layers.mamba.ops.ssu_tuning import (
    SSUTuningCase,
    make_active_cases,
    ssu_scratch_bytes,
    tune_ssu_case,
    valid_request_batches,
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
class SSUShape:
    headdim: int
    dstate: int
    nheads: int
    ngroups: int
    dtype: torch.dtype
    state_dtype: torch.dtype

    @property
    def cache_dtype(self) -> str:
        return str(self.state_dtype).removeprefix("torch.")


def discover_ssu_shapes(worker: Worker) -> set[SSUShape]:
    dtype = worker.model_config.dtype
    shapes: set[SSUShape] = set()
    for module in worker.get_model().modules():
        if isinstance(module, MambaMixer2):
            if module.use_replayssm:
                continue
            nheads, headdim, dstate = module.get_state_shape()[1]
            ngroups = module.n_groups // module.tp_size
        elif isinstance(module, MambaMixer):
            # Mamba-1 state is (dim, dstate); the SSU wrapper treats it as one head.
            headdim, dstate = module.get_state_shape()[1]
            nheads, ngroups = 1, 1
        else:
            continue
        state_dtype = module.get_state_dtype()[1]
        shapes.add(SSUShape(headdim, dstate, nheads, ngroups, dtype, state_dtype))
    return shapes


class MambaSSUConfigTable(TunableConfigTable):
    name = "mamba_ssu"

    def pending_items(self, worker: Worker) -> list[TuningItem]:
        if not current_platform.is_cuda_alike():
            return []
        if worker.vllm_config.mamba_config.backend != MambaBackendEnum.TRITON:
            return []
        request_batches = valid_request_batches(worker.scheduler_config.max_num_seqs)
        items = []
        for shape in discover_ssu_shapes(worker):
            has_config = (
                get_ssm_configs(shape.headdim, shape.dstate, shape.cache_dtype)
                is not None
            )
            if has_config and not envs.VLLM_TRITON_AUTOTUNE_FORCE:
                continue
            for _, batch, _ in make_active_cases(
                request_batches, shape.nheads, shape.ngroups
            ):
                items.append(TuningItem(self.name, shape, batch))
        return items

    def tune(self, item: TuningItem) -> Config | None:
        shape = item.shape
        case = SSUTuningCase(
            batch=item.bucket,
            nheads=shape.nheads,
            headdim=shape.headdim,
            dstate=shape.dstate,
            ngroups=shape.ngroups,
            dtype=shape.dtype,
            state_dtype=shape.state_dtype,
            device=torch.device(current_platform.device_type),
            is_blackwell=current_platform.is_device_capability_family(100),
        )
        free_bytes, _ = torch.accelerator.get_memory_info()
        if ssu_scratch_bytes(case) > free_bytes // 2:
            logger.warning("Skipping Mamba SSU tuning for %s: low free memory.", item)
            return None
        return tune_ssu_case(case)

    def commit(self, results: dict[TuningItem, Config]) -> None:
        by_file: dict[tuple[int, int, str], dict[int, Config]] = defaultdict(dict)
        for item, config in results.items():
            shape = item.shape
            key = (shape.headdim, shape.dstate, shape.cache_dtype)
            by_file[key][item.bucket * shape.nheads] = config
        for (headdim, dstate, cache_dtype), configs in by_file.items():
            merged = {**load_ssm_autotune_configs(headdim, dstate, cache_dtype), **configs}
            path = save_ssm_configs(headdim, dstate, cache_dtype, merged)
            logger.info("Saved %d Mamba SSU configs to %s.", len(configs), path)