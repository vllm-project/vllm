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

Selected with ``VLLM_RELOAD_MODE=direct``.
"""

from collections.abc import Iterator

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "finish_direct_reload",
    "record_checkpoint_layout",
]


def direct_reload_enabled() -> bool:
    return envs.VLLM_RELOAD_MODE == "direct"


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


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout. Runs once; reloads reuse the cold-start record."""
    if getattr(model, "_direct_layout_recorded", False):
        return
    model._direct_layout_recorded = True
    config = get_current_vllm_config_or_none()
    spec = config.speculative_config if config is not None else None
    model._direct_draft_lm_head_quantization = (
        spec.draft_lm_head_quantization if spec is not None else None
    )
    for _, t in _loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt_layout"):
            t._ckpt_layout = (tuple(t.shape), t.stride(), t.dtype, t.storage_offset())


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
        span = 1 + sum((s - 1) * st for s, st in zip(shape, stride)) if shape else 1
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
    if model._direct_draft_lm_head_quantization is not None:
        # the draft's quantized lm_head is a view of the target lm_head's
        # storage after `begin`: post-load processing would write it there
        raise RuntimeError("direct reload: draft_lm_head_quantization is not supported")
    views = _checkpoint_views(model)

    # nothing above modified the model
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


@torch.no_grad()
def finish_direct_reload(model: torch.nn.Module, model_config) -> None:
    """Rerun cold-start post-load processing."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    torch.accelerator.synchronize()  # engines may write on other streams

    # One-shot "already processed" guards (R7) would make the rerun below a
    # no-op and leave the freshly written checkpoint layout in place.
    cleared = 0
    for module in model.modules():
        if module.__dict__.pop("_already_called_process_weights_after_loading", None):
            cleared += 1
    if cleared:
        logger.info("direct reload: cleared %d one-shot post-load guards", cleared)

    device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"),
        torch.accelerator.current_accelerator(),
    )
    process_weights_after_loading(model, model_config, device)
