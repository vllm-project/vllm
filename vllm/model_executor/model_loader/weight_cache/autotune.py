# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the weight cache daemon's in-daemon FlashInfer autotune."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    write_unified_block_size,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

# Block size resolution is memory-independent (memory only feeds num_blocks);
# a generous probe keeps the admission check and auto-fit inert. Auto-fit only
# engages with --max-model-len -1, whose engine value depends on real free
# memory and is out of scope for key fidelity.
_KV_CACHE_PROBE_MEMORY = 1 << 40  # 1 TiB


def build_tuning_runner(
    vllm_config: VllmConfig,
    local_rank: int,
    *,
    is_draft: bool,
    model_config: "ModelConfig",
) -> "GPUModelRunner":
    """Create model runner for the autotune dummy runs"""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if is_draft:
        # The draft model is what this daemon holds: tune it as the runner's
        # own model, not as another model's drafter. replace() is a shallow
        # copy, so the runner's config shares compilation_config with the
        # config the model was built under -- the layers' static registries
        # stay visible.
        vllm_config = replace(
            vllm_config, model_config=model_config, speculative_config=None
        )
    device = torch.device(current_platform.device_type, local_rank)
    with set_current_vllm_config(vllm_config):
        runner = GPUModelRunner(vllm_config, device)
    return runner


def replicate_engine_cache_config(
    vllm_config: VllmConfig, runner: "GPUModelRunner"
) -> None:
    """Replicate the engine's cache-config mutations up to kernel warmup."""
    # Same call the executor makes right after load_model.
    current_platform.update_block_size_for_backend(vllm_config)
    # Engine core merges the per-worker specs (all TP x PP ranks); the daemon
    # world group has the same shape on a single node.
    world = get_world_group()
    if world.world_size > 1:
        gathered: list[dict[str, KVCacheSpec]] = [None] * world.world_size  # type: ignore[list-item]
        torch.distributed.all_gather_object(
            gathered, runner.get_kv_cache_spec(), group=world.cpu_group
        )
        kv_cache_specs = gathered
    else:
        kv_cache_specs = [runner.get_kv_cache_spec()]
    # The engine resolves the KV cache layout before grouping and records it
    # on the cache config.
    backends = get_current_attn_backends(vllm_config)
    layouts = [layout.name for layout in get_supported_kv_cache_layouts(backends)]
    resolve_kv_cache_layout(
        vllm_config,
        [layouts],
        [spec for specs in kv_cache_specs for spec in specs.values()],
    )
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, [_KV_CACHE_PROBE_MEMORY] * len(kv_cache_specs)
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    write_unified_block_size(vllm_config, scheduler_kv_cache_config)
    vllm_config.validate_block_size()
