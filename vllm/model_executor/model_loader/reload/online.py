# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffered loading for online-quantized layers.

An online quantization method creates its checkpoint-format parameters on the
meta device and calls `install_online_buffers`. Each such parameter's weight
loader then writes into a zero-filled buffer of the checkpoint shape, which
exists only while the layer loads. When every expected shard has arrived, the
layer's `process_weights_after_loading` (PWAL) runs on the buffers, exactly as
at cold start, and the buffers are dropped.

Cold start and weight updates take the same path. On an update,
`begin_online_reload` remembers the tensors the layer holds (what captured
CUDA graphs read), and after PWAL the results are copied back into them.
These layers run PWAL when their last shard arrives instead of at the end of
the update; every other module still runs it at the end, after them.

Completion is detected from shard keys, not element counts: a stacked linear
expects each of its `shard_ids` with rows on this rank, an MoE layer expects
`(expert, w1|w3)` for `w13_*` and `(expert, w2)` for `w2_*` per local expert
slot, and other layers expect one unsharded load per parameter. A load whose
key cannot be derived defers the layer to `finalize_online`.
"""

import inspect
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps
from weakref import WeakSet

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)

__all__ = [
    "begin_online_reload",
    "drop_online_buffers",
    "finalize_online",
    "has_online_buffers",
    "install_online_buffers",
]

_STATE_ATTR = "_online_buffer_state"
_GUARD = "_already_called_process_weights_after_loading"

# Layers holding buffers right now, to warn when loads interleave many layers.
_LOADING: WeakSet[torch.nn.Module] = WeakSet()

Key = tuple


@dataclass
class _State:
    # checkpoint-format parameters as created, on meta; never registered again
    ckpt: dict[str, torch.nn.Parameter]
    loaders: dict[str, Callable]  # the original weight loaders
    wrappers: dict[str, Callable]  # what each parameter's weight_loader is
    device: torch.device
    expected: set[Key] | None = None  # computed on first load
    seen: set[Key] = field(default_factory=set)
    loaded: set[str] = field(default_factory=set)  # params holding a buffer
    unknown_key: bool = False  # a load whose shard key could not be derived
    defer: bool = False  # process only in `finalize_online`
    # (params, buffers, non-persistent names) held when the update began, each
    # with the tensor it held: PWAL may rebind a kept parameter's `.data`
    kernel: tuple[dict, dict, set[str]] | None = None


def _state(module: torch.nn.Module) -> _State | None:
    return module.__dict__.get(_STATE_ATTR)


def has_online_buffers(module: torch.nn.Module) -> bool:
    """Whether `module` loads through online buffers."""
    return _state(module) is not None


def _online_layers(model: torch.nn.Module) -> Iterable[tuple[torch.nn.Module, _State]]:
    for module in model.modules():
        state = _state(module)
        if state is not None:
            yield module, state


def install_online_buffers(layer: torch.nn.Module) -> None:
    """Route the loads of `layer`'s meta parameters through buffers. Called by
    online quantization methods at the end of `create_weights`."""
    state = _State(ckpt={}, loaders={}, wrappers={}, device=torch.get_default_device())
    for name, param in layer.named_parameters(recurse=False):
        if not param.is_meta:
            continue  # e.g. a bias the layer adds later, loaded in place
        original = getattr(param, "weight_loader", default_weight_loader)
        wrapper = _make_loader(layer, name, original)
        param.weight_loader = wrapper
        state.ckpt[name] = param
        state.loaders[name] = original
        state.wrappers[name] = wrapper
    layer.__dict__[_STATE_ATTR] = state


def drop_online_buffers(layer: torch.nn.Module) -> None:
    """Stop routing `layer`'s loads through buffers (a derived copy that is
    never loaded)."""
    layer.__dict__.pop(_STATE_ATTR, None)


def _materialize(meta: torch.nn.Parameter, device: torch.device) -> torch.Tensor:
    """A zero-filled tensor like `meta`, keeping its class and attributes
    (`BasevLLMParameter` loaders are methods of the parameter)."""
    with device:
        tensor = torch.zeros_like(meta, device=device)
    tensor.__class__ = meta.__class__
    tensor.__dict__ = meta.__dict__.copy()
    return tensor


def _expected_keys(layer: torch.nn.Module, names: Iterable[str]) -> set[Key]:
    keys: set[Key] = set()
    moe_config = getattr(layer, "moe_config", None)
    shard_ids = getattr(layer, "shard_ids", None)
    for name in names:
        if moe_config is not None and hasattr(layer, "local_num_experts"):
            if name.startswith("w13"):
                shards = ("w1", "w3") if moe_config.is_act_and_mul else ("w1",)
            else:
                shards = ("w2",)
            keys |= {
                (name, e, s) for e in range(layer.local_num_experts) for s in shards
            }
        elif shard_ids is not None:
            # a shard with no rows on this rank is never loaded
            size = layer._get_shard_size_mapping
            keys |= {(name, s) for s in shard_ids if size(s)}
        else:
            keys.add((name, None))
    return keys


def _load_keys(
    layer: torch.nn.Module, name: str, args: inspect.BoundArguments
) -> set[Key] | None:
    """The keys one load fills; `None` if they cannot be derived."""
    arguments = args.arguments
    values = list(arguments.values())
    if getattr(layer, "moe_config", None) is not None and hasattr(
        layer, "local_num_experts"
    ):
        shard, expert = arguments.get("shard_id"), arguments.get("expert_id")
        if shard is None or expert is None or len(values) < 2:
            return None
        local = layer._map_global_expert_id_to_local_expert_id(expert)
        if local < 0:
            return set()  # not on this rank; the loader skips it too
        if values[1].ndim == 3:  # all experts in one tensor
            return {(name, e, shard) for e in range(layer.local_num_experts)}
        return {(name, local, shard)}
    shard_ids = getattr(layer, "shard_ids", None)
    if shard_ids is not None:
        if "loaded_shard_id" not in arguments:
            return None
        shard = arguments["loaded_shard_id"]
        shards = shard_ids if shard is None else shard
        shards = shards if isinstance(shards, tuple) else (shards,)
        return {(name, s) for s in shards}
    return {(name, None)}


def _make_loader(layer: torch.nn.Module, name: str, original: Callable) -> Callable:
    signature = inspect.signature(original)

    # inspect.signature follows __wrapped__, so callers that look for
    # `expert_id` in the loader's signature still find it
    @wraps(original)
    def online_buffer_loader(*args, **kwargs):
        from .trace import TraceTensor

        state = _state(layer)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()  # e.g. `loaded_shard_id=None`: a fused tensor
        first, loaded_weight = list(bound.arguments)[:2]
        if state is None:  # dropped: load as the layer was built
            return original(*args, **kwargs)
        if isinstance(bound.arguments[loaded_weight], TraceTensor):
            # Load plan tracing: these bytes land in a buffer, never in a
            # checkpoint tensor, so the name goes through `load_weights`.
            trace = bound.arguments[loaded_weight]
            trace._sink.mark(trace._name, "loads into an online quantization buffer")
            trace._sink.wrote.add(trace._name)
            return True if bound.arguments.get("return_success") else None

        if name not in state.loaded:
            target = _materialize(state.ckpt[name], state.device)
            layer.register_parameter(name, target)  # type: ignore[arg-type]
            state.loaded.add(name)
            _LOADING.add(layer)
            if len(_LOADING) > 2:
                logger.warning_once(
                    "Online quantization is buffering %d layers at once; order "
                    "the weights by layer to load one at a time.",
                    len(_LOADING),
                )
        bound.arguments[first] = getattr(layer, name)
        ret = original(*bound.args, **bound.kwargs)

        keys = _load_keys(layer, name, bound)
        if keys is None:
            if not state.unknown_key:
                logger.debug(
                    "%s: no shard key for %s, deferring its processing",
                    type(layer).__name__,
                    name,
                )
            state.unknown_key = True
            return ret
        state.seen |= keys
        if state.expected is None:
            state.expected = _expected_keys(layer, state.ckpt)
        if not state.defer and not state.unknown_key and state.seen >= state.expected:
            _process(layer, state)
        return ret

    online_buffer_loader.loads_into_buffer = True  # type: ignore[attr-defined]
    return online_buffer_loader


def begin_online_reload(model: torch.nn.Module, defer: bool = False) -> None:
    """Start a weight update of `model`'s online layers: remember the tensors
    each holds, which receive the update once it is processed. With `defer`,
    layers are processed only by `finalize_online`."""
    for layer, state in _online_layers(model):
        state.kernel = (
            {n: (p, p.data) for n, p in layer._parameters.items() if p is not None},
            {n: (b, b) for n, b in layer._buffers.items() if b is not None},
            set(layer._non_persistent_buffers_set),
        )
        # until processed, the layer shows its checkpoint format, as other
        # layers do during an update, and as its PWAL saw it at cold start:
        # parameters that PWAL created are hidden (`_restore` brings them back)
        for name in [n for n in layer._parameters if n not in state.ckpt]:
            delattr(layer, name)
        for name, meta in state.ckpt.items():
            layer.register_parameter(name, meta)
        state.seen.clear()
        state.loaded.clear()
        state.unknown_key = False
        state.defer = defer


def finalize_online(model: torch.nn.Module, strict: bool) -> None:
    """Process `model`'s online layers that did not complete while loading.

    At cold start, a layer with missing weights is processed with zeros, as
    before. During an update, a layer that received nothing keeps its weights;
    one that received part of its shards raises with `strict` and is
    processed with zeros otherwise."""
    for layer, state in _online_layers(model):
        if state.kernel is None:  # cold start
            if getattr(layer, _GUARD, False):
                continue
        elif not state.loaded:  # not part of this update
            _restore(layer, state.kernel)
            state.kernel = None
            continue
        missing = sorted(map(str, (state.expected or set()) - state.seen))
        if missing and not state.unknown_key:
            if strict and state.kernel is not None:
                raise RuntimeError(
                    f"online quantization: {type(layer).__name__} received only "
                    f"part of its weights in this update (missing {missing[:4]}); "
                    "send every tensor of the layer"
                )
            if state.loaded:
                logger.warning(
                    "%s: missing weights %s, processed as zeros",
                    type(layer).__name__,
                    missing[:4],
                )
        _process(layer, state)


@torch.no_grad()
def _process(layer: torch.nn.Module, state: _State) -> None:
    """Run the layer's PWAL on its buffers; on an update, copy the result into
    the tensors the layer held when the update began."""
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )
    from vllm.model_executor.model_loader.utils import (
        online_quantization_timing_context,
    )

    for name, meta in state.ckpt.items():
        if name not in state.loaded:  # nothing arrived for it
            layer.register_parameter(name, _materialize(meta, state.device))  # type: ignore[arg-type]
    layer.__dict__.pop(_GUARD, None)
    quant_method = layer.quant_method
    timing = (
        online_quantization_timing_context(layer, quant_method)
        if isinstance(quant_method, (OnlineLinearBase, OnlineMoEMethodBase))
        else nullcontext()
    )
    with timing:
        quant_method.process_weights_after_loading(layer)
    if hasattr(layer, "update_param_tp_status"):
        layer.update_param_tp_status()
    if state.kernel is not None:
        _copy_back(layer, state.kernel)
    for name, wrapper in state.wrappers.items():  # undo an override (EPLB)
        state.ckpt[name].weight_loader = wrapper
        param = layer._parameters.get(name)
        if param is not None:
            param.weight_loader = wrapper
    state.kernel = None
    state.seen.clear()
    state.loaded.clear()
    state.unknown_key = False
    state.defer = False
    _LOADING.discard(layer)


def _layout(t: torch.Tensor) -> tuple:
    return t.data_ptr(), t.shape, t.stride(), t.dtype


def _copy_back(layer: torch.nn.Module, kernel: tuple[dict, dict, set[str]]) -> None:
    params, buffers, _ = kernel
    for held, live in ((params, layer._parameters), (buffers, layer._buffers)):
        for name, (_, data) in held.items():
            new = live.get(name)
            if new is None or _layout(new) == _layout(data):
                continue  # dropped, or written in place
            if new.shape != data.shape or new.dtype != data.dtype:
                raise RuntimeError(
                    f"online quantization: reprocessing {type(layer).__name__}."
                    f"{name} gave {tuple(new.shape)} {new.dtype}, but it holds "
                    f"{tuple(data.shape)} {data.dtype}"
                )
            data.copy_(new.data)
    _restore(layer, kernel)


def _restore(layer: torch.nn.Module, kernel: tuple[dict, dict, set[str]]) -> None:
    params, buffers, non_persistent = kernel
    for live in (layer._parameters, layer._buffers):
        for name in [n for n, t in live.items() if t is not None]:
            delattr(layer, name)
    for name, (param, data) in params.items():
        param.data = data
        layer.register_parameter(name, param)
    for name, (buffer, _) in buffers.items():
        layer.register_buffer(name, buffer, persistent=name not in non_persistent)
