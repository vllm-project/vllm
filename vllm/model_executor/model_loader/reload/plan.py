# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Records the initial checkpoint load as the set of ``weight_loader``
applications every later reload must repeat."""

import inspect
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from weakref import WeakKeyDictionary

import torch

from .utils import get_loadable_layer_tensors

# (canonical_source_name, tensor_name, selectors), e.g.
# ("model.layers.0.q_proj.weight", "weight", (("loaded_shard_id", "q"),))
LoadKey = tuple[str, str, tuple[tuple[str, Hashable], ...]]

# A multiset, because a quant config may load one key more than once
# (CT_WNA16 loads `weight_shape` per QKV partition).
LoadPlan = Counter[LoadKey]

_NON_SELECTOR_ARGS = frozenset({"param", "self", "return_success"})

_PLANS: WeakKeyDictionary[torch.nn.Module, LoadPlan] = WeakKeyDictionary()
# Separate from _PLANS so a layer that arms itself mid-load (online
# quantization does) cannot mistake an in-progress recording for a contract.
_RECORDING: WeakKeyDictionary[torch.nn.Module, LoadPlan] = WeakKeyDictionary()
_CURRENT_LOAD_SOURCE: ContextVar[str | None] = ContextVar(
    "vllm_current_load_source", default=None
)


@contextmanager
def load_source(source_name: str) -> Iterator[None]:
    """Associate loader applications with one canonical checkpoint tensor."""
    token = _CURRENT_LOAD_SOURCE.set(source_name)
    try:
        yield
    finally:
        _CURRENT_LOAD_SOURCE.reset(token)


def get_current_load_source() -> str | None:
    """Return the checkpoint source currently being consumed."""
    return _CURRENT_LOAD_SOURCE.get()


def install_load_source_dispatch(model: torch.nn.Module) -> None:
    """Keep each checkpoint source active while the model consumes it."""
    load_weights = getattr(model, "load_weights", None)
    if not callable(load_weights):
        return

    @wraps(load_weights)
    def dispatch(weights, *args, **kwargs):
        return load_weights(_with_load_sources(weights), *args, **kwargs)

    model.load_weights = dispatch


def _with_load_sources(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    for source_name, weight in weights:
        with load_source(source_name):
            yield source_name, weight


def make_load_key(
    tensor_name: str,
    bound_args: inspect.BoundArguments,
) -> LoadKey | None:
    """Identify one loader application by its source, destination, and every
    non-tensor argument, which is what selects the fragment being filled."""
    source_name = _CURRENT_LOAD_SOURCE.get()
    if source_name is None:
        # Unkeyable, because this caller did not route through `load_weights`.
        return None

    selectors = tuple(
        (name, value)
        for name, value in bound_args.arguments.items()
        if name not in _NON_SELECTOR_ARGS and not isinstance(value, torch.Tensor)
    )
    return source_name, tensor_name, selectors


def get_load_plan(layer: torch.nn.Module) -> LoadPlan | None:
    return _PLANS.get(layer)


def get_recorded_load_plan(layer: torch.nn.Module) -> LoadPlan | None:
    """Return the in-progress initial-load recording for probe validation."""
    return _RECORDING.get(layer)


def install_load_recorder(model: torch.nn.Module) -> None:
    """Instrument loaders so the initial load becomes the contract, until
    `freeze_load_plan` removes them again."""
    for layer in model.modules():
        _record_layer(layer)


def _record_layer(layer: torch.nn.Module) -> None:
    for name, tensor in get_loadable_layer_tensors(layer).items():
        loader = getattr(tensor, "weight_loader", None)
        if loader is None or not getattr(loader, "_is_load_recorder", False):
            tensor.weight_loader = _make_recorder(layer, name, loader)


def _make_recorder(
    layer: torch.nn.Module, tensor_name: str, original: Callable | None
) -> Callable:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    inner = default_weight_loader if original is None else original
    signature = inspect.signature(inner)

    @wraps(inner, assigned=("__doc__", "__annotations__"))
    def load_recorder(*args, **kwargs):
        # Wrap parameters registered mid-load, e.g. a quant method adding `bias`.
        _record_layer(layer)

        ret = inner(*args, **kwargs)
        if ret is False:
            # Declined, e.g. an expert this rank does not own, so it wrote
            # nothing and has not earned a place on the contract.
            return ret

        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        if (key := make_load_key(tensor_name, bound_args)) is not None:
            _RECORDING.setdefault(layer, Counter())[key] += 1
        return ret

    load_recorder._is_load_recorder = True  # type: ignore[attr-defined]
    # `None` where the tensor had no loader, so unwrapping restores the absence
    # rather than leaving `default_weight_loader` behind.
    load_recorder._original_loader = original  # type: ignore[attr-defined]
    return load_recorder


def freeze_load_plan(model: torch.nn.Module) -> None:
    """Freeze the recording as the contract and remove the recorders,
    idempotently and safely on a model that was never instrumented."""
    for layer in model.modules():
        _unwrap_recorders(layer)
        if plan := _RECORDING.pop(layer, None):
            _PLANS[layer] = plan


def _unwrap_recorders(layer: torch.nn.Module) -> None:
    for tensor in get_loadable_layer_tensors(layer).values():
        loader = getattr(tensor, "weight_loader", None)
        if not getattr(loader, "_is_load_recorder", False):
            continue
        if loader._original_loader is None:
            del tensor.weight_loader
        else:
            tensor.weight_loader = loader._original_loader
