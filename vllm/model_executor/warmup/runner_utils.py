# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner accessors shared by warmups that support both V1 and V2."""

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker


def draft_model(worker: "Worker") -> torch.nn.Module | None:
    """The drafter's model: V1 ``runner.drafter``, V2 ``runner.speculator``."""
    runner = worker.model_runner
    owner_name = "speculator" if worker.use_v2_model_runner else "drafter"
    model = getattr(getattr(runner, owner_name, None), "model", None)
    return model if isinstance(model, torch.nn.Module) else None


def target_and_draft_models(worker: "Worker") -> list[torch.nn.Module]:
    drafter = draft_model(worker)
    return [worker.get_model()] + ([drafter] if drafter is not None else [])


def block_table(worker: "Worker", group_id: int, num_reqs: int) -> torch.Tensor:
    """The runner's device block table for ``group_id``, first ``num_reqs`` rows."""
    if worker.use_v2_model_runner:
        runner_v2 = cast("GPUModelRunnerV2", worker.model_runner)
        return runner_v2.block_tables.input_block_tables[group_id][:num_reqs]
    return worker.model_runner.input_batch.block_table[group_id].get_device_tensor(
        num_reqs
    )


def kv_cache_group_ids(worker: "Worker") -> dict[str, int]:
    """Map each attention layer name to its KV cache group index."""
    return {
        name: group_id
        for group_id, group in enumerate(
            worker.model_runner.kv_cache_config.kv_cache_groups
        )
        for name in group.layer_names
    }
