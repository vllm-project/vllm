# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload.

Instead of moving layers to meta and rebuilding them, the loader writes
checkpoint-format tensors straight into the live tensors and the cold-start
post-load processing runs once at the end:

- cold start records each tensor's checkpoint layout (before post-load
  processing);
- ``begin`` views tensors whose layout post-load processing changed back to
  their checkpoint layout, on the same storage;
- ``finish`` reruns ``process_weights_after_loading`` for every module, as
  cold start does. Post-load processing has to refill derived tensors in
  place (captured CUDA graphs keep their startup addresses); nothing moves
  them back.

A reload is exactly as correct as a cold start on the delivered weights: no
module is skipped because it "received nothing" (a loader may bypass the
parameter's ``weight_loader``, and post-processing may read other modules'
tensors).

Delivery contract: an update must deliver every checkpoint tensor of each
module whose post-load processing changes the tensor's layout or is not
idempotent (it would convert the kept runtime bytes again). A tensor without
such processing (e.g. an embedding table the trainer keeps frozen) may be left
out and keeps its bytes. Nothing checks this contract here.

Post-load processing must keep every checkpoint tensor registered under its
checkpoint name (`replace_parameter`, `retire_parameter`): a reload loads it
by that name.

Between ``begin`` and a successful ``finish`` the model holds checkpoint-layout
bytes and must not run; `direct_reload_pending` reports this. There is no
rollback: after a failed update, the next full update repairs the model.

Selected with ``VLLM_RELOAD_MODE=direct``; `start_reload` and `finish_reload`
(``lifecycle.py``) dispatch on it.
"""

from collections.abc import Iterator

import torch

import vllm.envs as envs
from vllm.config import ModelConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import _extent, is_weights_pre_processed

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "direct_reload_pending",
    "finish_direct_reload",
    "keep_addresses",
    "record_checkpoint_layout",
]


def direct_reload_enabled() -> bool:
    return envs.VLLM_RELOAD_MODE == "direct"


_finishing = False  # a direct finish is rerunning post-load processing


def keep_addresses() -> bool:
    """Whether `replace_parameter` writes into existing storage: at a direct
    cold start, or during a direct finish whatever the env says by then."""
    return _finishing or direct_reload_enabled()


def direct_reload_pending(model: torch.nn.Module) -> bool:
    """True between ``begin`` and a successful ``finish``: the model holds
    checkpoint-layout bytes and must not run."""
    return model.__dict__.get("_direct_reload_pending", False)


def _own_tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """A module's own parameters and persistent buffers."""
    for n, p in module._parameters.items():
        if p is not None:
            yield n, p
    for n, b in module._buffers.items():
        if b is not None and n not in module._non_persistent_buffers_set:
            yield n, b


def _loadable_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters and persistent buffers, each object once."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for name, tensor in _own_tensors(module):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield (f"{module_name}.{name}" if module_name else name), tensor


def _check_supported(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    config = get_current_vllm_config_or_none()
    spec = config.speculative_config if config is not None else None
    if spec is not None and spec.draft_lm_head_quantization is not None:
        # after `begin` the draft's quantized lm_head is a view of the target
        # lm_head's storage: post-load processing would write it there
        raise RuntimeError("direct reload: draft_lm_head_quantization is not supported")
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase)):
            # `finish` clears the guard that keeps online quantization from
            # quantizing its already-quantized bytes again
            raise RuntimeError(
                f"direct reload: online quantization is not supported "
                f"({name}: {type(method).__name__})"
            )


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout, and which parameter names hold checkpoint tensors.
    Runs once; reloads reuse the cold-start record."""
    if getattr(model, "_direct_layout_recorded", False) or is_weights_pre_processed():
        return  # pre-processed weights are already in runtime layout
    _check_supported(model)
    model._direct_layout_recorded = True
    for _, t in _loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt_layout"):
            t._ckpt_layout = (tuple(t.shape), t.stride(), t.dtype, t.storage_offset())
    # every name, aliases included: post-load processing may replace either
    model._ckpt_param_names = [
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if hasattr(p, "_ckpt_layout")
    ]


def _check_params_kept(model: torch.nn.Module) -> None:
    """Raise if post-load processing registered a new parameter under a
    checkpoint name: `begin` cannot view it back, and the loader would write
    checkpoint bytes into its runtime layout."""
    replaced = []
    for name in model._ckpt_param_names:
        module_name, _, local = name.rpartition(".")
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue
        param = module._parameters.get(local)
        if param is not None and not hasattr(param, "_ckpt_layout"):
            replaced.append(name)
    if replaced:
        raise RuntimeError(
            f"direct reload: post-load processing replaced the parameter object "
            f"of {len(replaced)} checkpoint tensors (e.g. {replaced[:3]}); keep "
            "it with `replace_parameter` / `retire_parameter` instead"
        )


def _checkpoint_views(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(tensor, view of its storage in its checkpoint layout) for every
    tensor post-load processing re-laid out; raises if a storage cannot
    hold its checkpoint layout."""
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, t in _loadable_tensors(model):
        layout = getattr(t, "_ckpt_layout", None)
        if layout is None:
            continue  # no layout: created by post-load processing, derived
        if (tuple(t.shape), t.stride(), t.dtype, t.storage_offset()) == layout:
            continue
        shape, stride, dtype, offset = layout
        storage = t.untyped_storage()
        span = _extent(torch.empty_strided(shape, stride, device="meta"))
        need = (offset + span) * dtype.itemsize
        if need > storage.nbytes():
            raise RuntimeError(
                f"direct reload: {name} storage has {storage.nbytes()} bytes, "
                f"its checkpoint layout needs {need}"
            )
        view = torch.empty(0, dtype=dtype, device=t.device)
        view.set_(storage, offset, shape, stride)
        views.append((t, view))
    return views


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader.

    Everything that can refuse the reload is checked first and raises with
    the model untouched. Once this returns, the model holds checkpoint-layout
    bytes until a finish succeeds."""
    if not getattr(model, "_direct_layout_recorded", False):
        raise RuntimeError("direct reload: no checkpoint layout was recorded")
    _check_params_kept(model)
    views = _checkpoint_views(model)

    # nothing above modified the model
    model._direct_reload_pending = True
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


@torch.no_grad()
def finish_direct_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Rerun cold-start post-load processing."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    if device.type != "cpu":
        torch.accelerator.synchronize()  # engines may write on other streams

    # One-shot "already processed" guards would make the rerun below a no-op
    # and leave the freshly written checkpoint layout in place. Only quant
    # methods' guards: other modules set it for other reasons.
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), QuantizeMethodBase):
            module.__dict__.pop("_already_called_process_weights_after_loading", None)

    global _finishing
    _finishing = True
    try:
        process_weights_after_loading(model, model_config, device)
    finally:
        _finishing = False
    model._direct_reload_pending = False
