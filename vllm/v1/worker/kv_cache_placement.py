# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover cache replication and bundle relationships from loaded modules."""

from typing import Any

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.kv_cache_placement import KVCacheBundle, KVCachePlacement


def get_kv_cache_placement(
    vllm_config: VllmConfig, model_runner: Any
) -> KVCachePlacement:
    if not current_platform.is_cuda():
        raise ValueError("KVPP broadcast currently requires NVIDIA CUDA.")
    parallel = vllm_config.parallel_config
    if parallel.decode_context_parallel_size != 1:
        raise ValueError("KVPP requires replicated KV; DCP is not supported.")
    if parallel.prefill_context_parallel_size != 1:
        raise ValueError("KVPP with PCP requires a verified replica-domain backend.")
    if parallel.enable_dbo:
        raise ValueError("KVPP does not yet support overlapping microbatches.")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError("KVPP currently requires explicit eager execution.")
    if vllm_config.attention_config.hisparse_config is not None:
        raise ValueError("KVPP cannot yet be combined with HiSparse storage.")

    if vllm_config.kv_transfer_config is not None:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        connector = KVConnectorFactory.get_connector_class(
            vllm_config.kv_transfer_config
        )
        if not connector.supports_layer_sharded_kv_cache():
            raise ValueError(
                f"{connector.__name__} does not support persistent KVPP views."
            )

    specs = model_runner.get_kv_cache_spec()
    modules = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    if any(
        getattr(module, "kv_sharing_target_layer_name", None) is not None
        for module in modules.values()
    ):
        raise ValueError("Shared-layer KVPP requires explicit shared lifetimes.")
    names_by_id = {id(module): name for name, module in modules.items()}
    proposer = getattr(model_runner, "speculator", None)
    draft_names = getattr(proposer, "draft_attn_layer_names", ())
    draft_names = set(draft_names or ())

    targets = []
    covered: set[str] = set()
    for name, module in modules.items():
        if name not in specs or name in draft_names:
            continue
        components = module.get_kv_cache_bundle()
        if components is None:
            continue
        bundle = tuple(names_by_id[id(component)] for component in components)
        if any(n not in specs or n in covered or n in draft_names for n in bundle):
            raise ValueError(f"Invalid or shared KVPP bundle at {name}.")
        targets.append(bundle)
        covered.update(bundle)
    if covered | draft_names != set(specs):
        raise ValueError(
            "KVPP requires explicit replicated bundles for every target cache: "
            f"{set(specs) - covered - draft_names}"
        )
    group = get_tp_group()
    if group.world_size <= 1:
        raise ValueError("KVPP requires at least two KV replica ranks.")
    if len(targets) < group.world_size:
        raise ValueError("KVPP requires at least one target bundle per replica rank.")
    base, extra = divmod(len(targets), group.world_size)
    bundles = []
    index = 0
    for owner in range(group.world_size):
        for _ in range(base + (owner < extra)):
            bundles.append(KVCacheBundle(targets[index], owner))
            index += 1
    bundles.extend(KVCacheBundle((name,), None) for name in sorted(draft_names))
    return KVCachePlacement(
        group.rank_in_group,
        group.world_size,
        tuple(bundles),
        alignment=256,
    )
