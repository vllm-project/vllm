# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the weight cache daemon, the IPC loader and the engine.

These live outside ``protocol`` so callers that only need to know whether a
draft is cached, or how a daemon group is named, do not have to import the
wire format.
"""

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import SpeculativeConfig, VllmConfig, replace, set_current_vllm_config
from vllm.model_executor.models.interfaces import SupportsEagleBase
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        TensorEntry,
        TensorKind,
        WeightCacheKey,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

DaemonRole = Literal["target", "draft"]
"""Which daemon group serves a model: the target or the speculative draft."""


# Speculative methods whose draft model the daemon caches in its own group.
# Other drafts keep loading from disk in the engine.
WEIGHT_CACHE_DRAFT_METHODS = frozenset({"mtp", "eagle", "eagle3"})

# Python-side flags that weight loading sets on EAGLE-style drafts; the engine
# never runs load_weights for cached models, so the daemon ships them.
EXPORTED_MODEL_ATTRS = tuple(SupportsEagleBase.__annotations__)


def export_model_attrs(model: Any) -> dict[str, bool]:
    return {
        name: bool(getattr(model, name))
        for name in EXPORTED_MODEL_ATTRS
        if hasattr(model, name)
    }


def is_draft_model_cacheable(speculative_config: SpeculativeConfig | None) -> bool:
    """Whether the daemon serves the speculative draft as a separate role."""
    return (
        speculative_config is not None
        and speculative_config.method in WEIGHT_CACHE_DRAFT_METHODS
        and speculative_config.draft_model_config is not None
    )


def format_daemon_role(is_draft: bool) -> DaemonRole:
    """Name of a daemon group: the target model or the draft."""
    return "draft" if is_draft else "target"


def format_socket_role_suffix(is_draft: bool) -> str:
    """Socket-name suffix keeping the draft group distinct from the target."""
    return "_draft" if is_draft else ""


def get_engine_dp_placement(vllm_config: VllmConfig, dp_rank: int) -> tuple[int, int]:
    """``(dp_size, dp_rank)`` as the engine fingerprints this DP rank.

    The engine runs the DP ranks of a non-MoE target as independent DP=1
    replicas, so their workers all report ``(1, 0)``.
    """
    if vllm_config.model_config.is_moe:
        return vllm_config.parallel_config.data_parallel_size, dp_rank
    return 1, 0


def build_weight_cache_key(
    vllm_config: VllmConfig,
    model_config: "ModelConfig",
    *,
    is_draft: bool = False,
) -> "WeightCacheKey":
    """Fingerprint this rank's shard exactly as ``WeightCacheDaemon`` does.

    Shared with the daemon so the peer loader requests the shard its own
    daemon fingerprinted: TP/PP ranks come from the rank the daemon (or
    engine worker) stamped on ``parallel_config.rank``, and the DP placement
    follows the engine's per-worker view.
    """
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        WeightCacheKey,
    )

    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    rank = parallel_config.rank
    dp_size, dp_rank = get_engine_dp_placement(
        vllm_config, parallel_config.data_parallel_rank
    )
    return WeightCacheKey.from_model_config(
        model_config,
        cache_config=vllm_config.cache_config,
        tp_size=tp_size,
        tp_rank=rank % tp_size,
        pp_size=parallel_config.pipeline_parallel_size,
        pp_rank=rank // tp_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        is_draft=is_draft,
        enable_expert_parallel=parallel_config.enable_expert_parallel,
        expert_placement_strategy=parallel_config.expert_placement_strategy,
        kernel_config=vllm_config.kernel_config,
    )


def build_warmup_runner(
    vllm_config: VllmConfig,
    local_rank: int,
    *,
    is_draft: bool,
    model_config: "ModelConfig",
) -> "GPUModelRunner":
    """Create a model runner for the daemon's warmup dummy runs."""
    if is_draft:
        vllm_config = replace(
            vllm_config, model_config=model_config, speculative_config=None
        )
    device = torch.device(current_platform.device_type, local_rank)
    with set_current_vllm_config(vllm_config):
        if vllm_config.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner as Runner
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner as Runner
        runner = Runner(vllm_config, device)
    # The two runner classes share the warmup surface; the nominal type
    # stays V1's GPUModelRunner.
    return cast("GPUModelRunner", runner)


def collect_tensors(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, "TensorKind"], dict[str, str]]:
    """Collect a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is kept once; every additional name that refers to the same tensor
    object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.

    Returns:
        A ``(tensors, kinds, aliases)`` triple: ``tensors`` maps a canonical
        name to its tensor, ``kinds`` maps it to "param" or "buffer", and
        ``aliases`` maps each duplicate name to its canonical name.

    """
    tensors: dict[str, torch.Tensor] = {}
    kinds: dict[str, TensorKind] = {}
    aliases: dict[str, str] = {}
    canonical_by_id: dict[int, str] = {}

    def _add(name: str, tensor: torch.Tensor, kind: "TensorKind") -> None:
        canonical = canonical_by_id.get(id(tensor))
        if canonical is not None:
            aliases[name] = canonical
            return
        canonical_by_id[id(tensor)] = name
        tensors[name] = tensor
        kinds[name] = kind

    for name, param in model.named_parameters(remove_duplicate=False):
        _add(name, param, "param")
    # named_buffers includes non-persistent buffers (e.g. rotary embedding
    # caches) that state_dict would miss.
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in tensors or name in aliases:
            continue
        _add(name, buffer, "buffer")
    return tensors, kinds, aliases


def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, "TensorEntry"], dict[str, str]]:
    """Export a model's tensors as ``TensorEntry``s, preserving aliases.

    CUDA reduction arguments must be exported separately for each consumer so
    that PyTorch registers a reference for each IPC mapping's lifetime.

    Returns:
        A ``(entries, aliases)`` pair, as ``collect_tensors`` but with each
        tensor exported.

    """
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        TensorEntry,
    )

    tensors, kinds, aliases = collect_tensors(model)
    entries = {
        name: TensorEntry.from_tensor(tensor, kinds[name])
        for name, tensor in tensors.items()
    }
    return entries, aliases


def _unique_cuda_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[int, int] = {}
    for tensor in tensors:
        if tensor.is_cuda:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def exported_weight_bytes(model: torch.nn.Module | None) -> int:
    """Bytes of the distinct CUDA storages an engine maps from this model.

    This is the memory the engine would otherwise allocate for the weights
    itself. ``memory_allocated()`` also counts daemon-only state the engine
    never maps (communication buffers, warmup workspaces, tensors outside
    parameters/buffers that the engine rebuilds itself), so reporting it
    would shrink the engine's KV cache below that of a disk load.
    """
    if model is None:
        return 0
    return _unique_cuda_storage_bytes(
        itertools.chain(model.parameters(), model.buffers())
    )
