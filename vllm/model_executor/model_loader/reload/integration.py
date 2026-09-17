# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold-load registration and checkpoint transport lifecycle for reload tracing."""

import inspect
from dataclasses import dataclass, field
from functools import partial

import torch

from .trace import ModelReloadTracer, ReloadError, ReloadState


@dataclass
class CopyReloadPolicy:
    """Reload parameters whose checkpoint and runtime representations coincide."""

    aliases: dict[str, torch.Tensor] = field(default_factory=dict)

    def bind(self, state: ReloadState) -> None:
        for role, expected in self.aliases.items():
            if getattr(state.module, role) is not expected:
                raise ReloadError(f"{state.key}/{role}: tied parameter was replaced")
            state.bind_target(role, partial(getattr, state.module, role))
        for role in state.roles:
            meta = state.metadata[role]
            target = state.targets[role].tensor
            if (
                meta.shape != target.shape
                or meta.dtype != target.dtype
                or meta.stride() != target.stride()
            ):
                raise ReloadError(f"{state.key}/{role}: requires a conversion policy")

    def validate(self, state: ReloadState) -> None:
        pass

    def destination(
        self, state: ReloadState, role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        return state.source(role, alias_runtime=True)

    def finish(self, state: ReloadState) -> None:
        for role in state.roles:
            state.copy_(role, state.work(role))


def create_model_reload_tracer(model: torch.nn.Module) -> ModelReloadTracer:
    """Register supported layers before cold loading; never silently fall back."""
    from vllm.model_executor.layers.attention import (
        Attention,
        is_deferred_attention_layer,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.utils.torch_utils import is_quantized_kv_cache

    if hasattr(model, "process_weights_after_loading"):
        raise NotImplementedError("Trace reload needs a model-level post-load policy")
    trace = ModelReloadTracer()
    copy_owners: dict[int, str] = {}
    for key, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if is_deferred_attention_layer(module):
            if type(module) is not Attention or is_quantized_kv_cache(
                module.kv_cache_dtype
            ):
                raise NotImplementedError(
                    f"{key}: trace reload needs an attention post-load policy"
                )
            if method is None or isinstance(method, BaseKVCacheMethod):
                # Non-quantized KV caches always use unit scales, independent
                # of checkpoint scale placeholders consumed during cold load.
                continue
        builder = getattr(method, "create_reload_state", None)
        if builder is not None:
            trace.register_state(builder(module, key))
            continue
        if callable(getattr(module, "process_weights_after_loading", None)):
            raise NotImplementedError(f"{key}: trace reload needs a post-load policy")
        if method is not None and type(method) not in (
            UnquantizedLinearMethod,
            UnquantizedEmbeddingMethod,
        ):
            raise NotImplementedError(
                f"{key}: {type(method).__name__} has no trace reload policy"
            )
        roles = []
        aliases = {}
        dependencies = set()
        for role, param in module.named_parameters(recurse=False):
            owner = copy_owners.get(id(param))
            if owner is not None:
                aliases[role] = param
                dependencies.add(owner)
            else:
                roles.append(role)
                copy_owners[id(param)] = key
        if roles or aliases:
            trace.register_state(
                ReloadState(
                    key,
                    module,
                    tuple(roles),
                    CopyReloadPolicy(aliases),
                    dependencies=tuple(sorted(dependencies)),
                )
            )
    return trace


def get_model_reload_tracer(model: torch.nn.Module) -> ModelReloadTracer:
    trace = getattr(model, "_reload_tracer", None)
    if not isinstance(trace, ModelReloadTracer) or not trace.bound:
        raise ReloadError(
            "Trace reload requires cold loading with reload_mode='trace'; "
            "this loader or update target did not initialize a tracer"
        )
    return trace
