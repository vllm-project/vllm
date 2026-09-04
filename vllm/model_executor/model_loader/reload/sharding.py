# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe the rank-local checkpoint fragments consumed by weight loaders."""

import inspect
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch

from vllm.model_executor.load_receipt import LoadFragment, LoadReceipt

from .source import get_current_source_name
from .units import TRACKER_ATTR

_FRAGMENT_ARGUMENTS = ("loaded_shard_id", "shard_id", "expert_id", "weight_name")
OBSERVED_SHARDS_ATTR = "_reload_observed_shards"
_ACTIVE_MODEL: ContextVar[torch.nn.Module | None] = ContextVar(
    "rank_sharding_model", default=None
)
_ACTIVE_SHARDS: ContextVar[set["RankShard"] | None] = ContextVar(
    "rank_sharding_records", default=None
)


@dataclass(frozen=True, order=True)
class RankShard:
    source_name: str
    target_name: str
    fragment: LoadFragment = LoadFragment()


@dataclass(frozen=True)
class RankShardingManifest:
    scope: "RankScope"
    shards: tuple[RankShard, ...]
    state: str = "exact"
    reason: str | None = None

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(sorted({shard.source_name for shard in self.shards}))


@dataclass(frozen=True)
class GroupRank:
    rank: int
    world_size: int


@dataclass(frozen=True)
class RankScope:
    global_rank: int | None = None
    global_world_size: int | None = None
    tp: GroupRank | None = None
    pp: GroupRank | None = None
    dp: GroupRank | None = None
    ep: GroupRank | None = None


_MANIFESTS: WeakKeyDictionary[torch.nn.Module, set[RankShard]] = WeakKeyDictionary()
_CAPTURED: WeakKeyDictionary[torch.nn.Module, bool] = WeakKeyDictionary()


class ReloadAwareWeightLoader:
    """Record successful parameter-loader calls during checkpoint loading.

    ``install_sharding_recorders`` replaces each parameter's ``weight_loader``
    with this wrapper. The wrapper keeps the original loader contract intact:
    it forwards all arguments, then records the source-to-target mapping only
    when the original loader consumed the tensor and a ``capture_rank_sharding``
    context is active. The recorded fragment fields (for example
    ``expert_id`` and ``shard_id``) describe the rank-local shard needed by a
    later streaming reload.
    """

    def __init__(self, model: torch.nn.Module, target_name: str, inner: Callable):
        self.model = weakref.ref(model)
        self.target_name = target_name
        self.inner = inner
        self.original = inner
        self.__wrapped__ = inner
        self.__name__ = getattr(inner, "__name__", "sharding_loader")
        self.supports_moe_loading = getattr(inner, "supports_moe_loading", False)
        try:
            self.signature = inspect.signature(inner)
        except (TypeError, ValueError):
            self.signature = None

    def __call__(self, *args, **kwargs):
        """Invoke the original loader and record a successfully consumed shard.

        The active source name is supplied by the checkpoint-loading source
        context, while loader arguments provide optional expert/shard metadata.
        Calls outside ``capture_rank_sharding`` are intentionally transparent;
        this allows the wrapper to be installed for the whole load lifecycle.
        """
        bound = None
        if self.signature is not None:
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
        model = self.model()
        source_name = get_current_source_name()
        routed_tracker = None
        routed_key = None
        if model is not None and self.signature is not None:
            fragment = {
                name: bound.arguments[name]
                for name in _FRAGMENT_ARGUMENTS
                if name in bound.arguments
                and isinstance(bound.arguments[name], (str, int, float, bool))
            }
            if "expert_id" in fragment and "shard_id" in fragment:
                module_name, _, parameter_name = self.target_name.rpartition(".")
                module = dict(model.named_modules()).get(module_name)
                tracker = (
                    getattr(module, TRACKER_ATTR, None)
                    if module is not None
                    else None
                )
                if tracker is not None:
                    expert_id = int(fragment["expert_id"])
                    mapper = getattr(
                        module, "_map_global_expert_id_to_local_expert_id", None
                    )
                    local_expert = mapper(expert_id) if mapper is not None else -1
                    key = (parameter_name, local_expert, str(fragment["shard_id"]))
                    routed_tracker = tracker
                    routed_key = key
                    parameter = bound.arguments.get("param")
                    if parameter is not None:
                        bound.arguments["param"] = tracker.target(key, parameter)
                        args = bound.args
                        kwargs = bound.kwargs

        result = self.inner(*args, **kwargs)
        if model is None:
            return result
        fields = {}
        if isinstance(result, LoadReceipt):
            consumed = result.consumed
            fragment = result.fragment
            fields = dict(fragment.items)
        else:
            consumed = result is True if (
                bound is not None
                and bound.arguments.get("return_success") is True
            ) else True
            if bound is not None:
                fields = {
                    name: bound.arguments[name]
                    for name in _FRAGMENT_ARGUMENTS
                    if name in bound.arguments
                    and isinstance(bound.arguments[name], (str, int, float, bool))
                }
            fragment = LoadFragment.from_fields(**fields)
        if consumed:
            if routed_tracker is not None and routed_key is not None:
                routed_tracker.record(routed_key)
            if "expert_id" in fields and "shard_id" in fields:
                module_name, _, parameter_name = self.target_name.rpartition(".")
                module = dict(model.named_modules()).get(module_name)
                if module is not None:
                    mapper = getattr(
                        module, "_map_global_expert_id_to_local_expert_id", None
                    )
                    expert_id = int(fields["expert_id"])
                    local_expert = mapper(expert_id) if mapper else expert_id
                    if local_expert >= 0:
                        observed = getattr(module, OBSERVED_SHARDS_ATTR, None)
                        if observed is None:
                            observed = set()
                            setattr(module, OBSERVED_SHARDS_ATTR, observed)
                        observed.add(
                            (parameter_name, local_expert, str(fields["shard_id"]))
                        )
            records = _ACTIVE_SHARDS.get()
            if records is not None and source_name is not None:
                records.add(RankShard(source_name, self.target_name, fragment))
        return result


def install_sharding_recorders(model: torch.nn.Module) -> None:
    """Wrap direct parameter loaders without changing their call contract."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for name, param in module._parameters.items():
            if param is None or id(param) in seen:
                continue
            seen.add(id(param))
            loader = getattr(param, "weight_loader", None)
            if isinstance(loader, ReloadAwareWeightLoader):
                continue
            if loader is None:
                loader = default_weight_loader
            target_name = f"{module_name}.{name}" if module_name else name
            param.weight_loader = ReloadAwareWeightLoader(model, target_name, loader)


def uninstall_sharding_recorders(model: torch.nn.Module) -> None:
    """Restore loaders after a baseline observation pass."""
    for module in model.modules():
        for param in module._parameters.values():
            if param is None:
                continue
            loader = getattr(param, "weight_loader", None)
            if isinstance(loader, ReloadAwareWeightLoader):
                param.weight_loader = loader.original


@contextmanager
def capture_rank_sharding(
    model: torch.nn.Module, *, reset: bool = False
) -> Iterator[set[RankShard]]:
    records: set[RankShard] = set()
    model_token = _ACTIVE_MODEL.set(model)
    records_token = _ACTIVE_SHARDS.set(records)
    try:
        yield records
    finally:
        _ACTIVE_SHARDS.reset(records_token)
        _ACTIVE_MODEL.reset(model_token)
        if reset:
            _MANIFESTS[model] = records
            _CAPTURED[model] = True


def get_rank_sharding_manifest(model: torch.nn.Module) -> RankShardingManifest:
    shards = tuple(sorted(_MANIFESTS.get(model, set())))
    if not _CAPTURED.get(model, False):
        return RankShardingManifest(
            _get_rank_scope(),
            shards,
            state="legacy",
            reason="No initial-load sharding capture was requested for this model.",
        )
    if shards or not any(True for _ in model.parameters()):
        return RankShardingManifest(_get_rank_scope(), shards)
    return RankShardingManifest(
        _get_rank_scope(),
        shards,
        state="unavailable",
        reason=(
            "The initial model load did not expose any checkpoint source-to-target "
            "events; its model loader needs a source-name adapter."
        ),
    )


def _get_rank_scope() -> RankScope:
    try:
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            get_pp_group,
            get_tp_group,
            get_world_group,
        )

        def group(getter: Callable) -> GroupRank | None:
            try:
                value = getter()
            except (AssertionError, RuntimeError):
                return None
            return GroupRank(value.rank_in_group, value.world_size)

        world = get_world_group()
        return RankScope(
            global_rank=world.rank,
            global_world_size=world.world_size,
            tp=group(get_tp_group),
            pp=group(get_pp_group),
            dp=group(get_dp_group),
            ep=group(get_ep_group),
        )
    except (AssertionError, RuntimeError):
        return RankScope()
