# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct the optional control runtime from the initialized engine profile."""

import hashlib
import json
import uuid

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_none_hash_seed
from vllm.v1.kv_hints.api import KVControlService
from vllm.v1.kv_hints.control import OperationRegistry
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.resources import KVResourceController
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex


def create_control_service(
    config: VllmConfig, manager: KVCacheManager, weight_version: str
) -> KVControlService:
    settings = config.kv_control_config
    assert settings is not None
    if not manager.enable_caching:
        raise ValueError("Programmable KV control requires prefix caching")
    engine_id = settings.engine_id or uuid.uuid4().hex
    if config.parallel_config.data_parallel_size > 1:
        engine_id += f":dp{config.parallel_config.data_parallel_rank}"
    profile = {
        "model": config.model_config.compute_hash(),
        "cache": config.cache_config.compute_hash(),
        "hash_algorithm": config.cache_config.prefix_caching_hash_algo,
        "hash_block_size": manager.block_pool.hash_block_size,
        "hash_seed": get_none_hash_seed(),
        "layout": config.cache_config.kv_cache_layout,
        "parallel": (
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.pipeline_parallel_size,
            config.parallel_config.decode_context_parallel_size,
            config.parallel_config.prefill_context_parallel_size,
        ),
        "groups": [
            (
                tuple(group.layer_names),
                repr(group.kv_cache_spec),
                group_manager.block_size,
            )
            for group, group_manager in zip(
                manager.kv_cache_config.kv_cache_groups,
                manager.coordinator.single_type_managers,
                strict=True,
            )
        ],
        "weight_version": weight_version,
    }
    namespace = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    registry = OperationRegistry(
        engine_id,
        max_records=settings.max_records,
        max_actions=settings.max_actions,
        max_envelope_bytes=settings.max_envelope_bytes,
        max_events=settings.max_events,
        result_ttl=settings.result_ttl,
    )
    index = SessionPrefixIndex(
        namespace,
        registry.engine_epoch,
        max_sessions=settings.max_sessions,
        max_continuations=settings.max_continuations,
        max_associations=settings.max_associations,
        max_events=settings.max_events,
        max_tombstones=settings.max_tombstones,
    )
    declarations = RetentionDeclarations(
        index,
        max_declarations=settings.max_declarations,
        max_snapshot_entries=settings.max_snapshot_entries,
    )
    executor = KVResourceController(manager, registry, declarations)
    cache_profile = {
        "hash_algorithm": profile["hash_algorithm"],
        "hash_block_size": profile["hash_block_size"],
        "hash_seed": profile["hash_seed"],
        "groups": [
            {
                "group_id": group_id,
                "block_size": group_manager.block_size,
                "prefix_cacheable": group.kv_cache_spec.prefix_cacheable,
            }
            for group_id, (group, group_manager) in enumerate(
                zip(
                    manager.kv_cache_config.kv_cache_groups,
                    manager.coordinator.single_type_managers,
                    strict=True,
                )
            )
        ],
    }
    return KVControlService(
        NativeKVController(
            registry,
            index,
            declarations,
            executor=executor,
            cache_profile=cache_profile,
        )
    )
