# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload (``VLLM_RELOAD_MODE=direct``).

Cold start records each tensor's checkpoint layout before post-load processing
(PWAL). ``begin`` views re-laid-out tensors back to that layout on their
storage, the loaders write the checkpoint in place, and ``finish`` reruns PWAL,
which must keep addresses (captured CUDA graphs) and refill derived tensors.
Between ``begin`` and a successful ``finish`` the model must not run; there is
no rollback, the next full update repairs it.

Delivery contract (not checked here): an update delivers every checkpoint
tensor of each module whose PWAL changes its layout or is not idempotent.
"""

from collections.abc import Iterator

import torch

import vllm.envs as envs
from vllm.config import ModelConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    storage_extent,
    storage_view,
)

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
    "direct_reload_pending",
    "finish_direct_reload",
    "record_checkpoint_layout",
]


# Loader metadata post-load processing may retag on a kept parameter (AWQ
# repacks along another dim); the checkpoint loaders shard with it.
_LOADER_ATTRS = (
    "_input_dim",
    "_output_dim",
    "_packed_dim",
    "_packed_factor",
    "_marlin_tile_size",
    "input_dim",
    "output_dim",
    "packed_dim",
    "packed_factor",
    "pack_factor",
)


def direct_reload_enabled() -> bool:
    return envs.VLLM_RELOAD_MODE == "direct"


def direct_reload_pending(model: torch.nn.Module) -> bool:
    """True between ``begin`` and a successful ``finish``: the model holds
    checkpoint-layout bytes and must not run."""
    return model.__dict__.get("_direct_reload_pending", False)


def _layout(t: torch.Tensor) -> tuple:
    return tuple(t.shape), t.stride(), t.dtype, t.storage_offset()


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
    from vllm.model_executor.layers.quantization.humming import HummingMoEMethod
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )
    from vllm.model_executor.offloader import NoopOffloader, get_offloader

    offloader = get_offloader()
    if not isinstance(offloader, NoopOffloader):
        # CPU/UVA offload gives parameters new storage on every post-load
        # processing; prefetch offload shares one buffer between layers
        raise RuntimeError(
            f"direct reload: weight offloading is not supported "
            f"({type(offloader).__name__})"
        )
    config = get_current_vllm_config_or_none()
    model_config = config.model_config if config is not None else None
    if model_config is not None and model_config.quantization == "torchao":
        # its load_weights runs a layerwise reload, and its post-load
        # processing replaces the weight parameter
        raise RuntimeError("direct reload: torchao quantization is not supported")
    quant_config = config.quant_config if config is not None else None
    if isinstance(quant_config, OnlineQuantizationConfig) or (
        getattr(quant_config, "online_quantization_config", None) is not None
    ):
        # `finish` clears the guard that keeps online quantization from
        # quantizing its already-quantized bytes again
        raise RuntimeError("direct reload: online quantization is not supported")
    if config is not None and config.lora_config is not None:
        # LoRA wraps layers after cold start; their wrappers forward
        # `quant_method`, so a rerun would process each base layer twice
        raise RuntimeError("direct reload: LoRA is not supported")
    spec = config.speculative_config if config is not None else None
    if spec is not None and spec.draft_lm_head_quantization is not None:
        # after `begin` the draft's quantized lm_head is a view of the target
        # lm_head's storage: post-load processing would write it there
        raise RuntimeError("direct reload: draft_lm_head_quantization is not supported")
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        # also without an online config (experts_int8); fused shared experts
        # quantize staging parameters that post-load processing deletes
        if isinstance(method, (OnlineLinearBase, OnlineMoEMethodBase)) or (
            getattr(module, "shared_expert_online_loader", None) is not None
        ):
            raise RuntimeError(
                f"direct reload: online quantization is not supported "
                f"({name}: {type(method).__name__})"
            )
        if isinstance(method, HummingMoEMethod):
            # its post-load processing runs once per quant method (a guard on
            # the method, not the module): a reload would keep checkpoint bytes
            raise RuntimeError(f"direct reload: humming MoE is not supported ({name})")


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout and loader metadata, which tensors this model owns (those
    no other model recorded first), and which parameter names hold checkpoint
    tensors.
    Runs once; reloads reuse the cold-start record."""
    if hasattr(model, "_ckpt_param_names") or is_weights_pre_processed():
        return  # pre-processed weights are already in runtime layout
    _check_supported(model)
    # a model owns the tensors it records; another model's (already recorded)
    # are refreshed by that model's updates
    model._ckpt_tensor_ids = set()
    for _, t in _loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt_layout"):
            model._ckpt_tensor_ids.add(id(t))
            t._ckpt_layout = _layout(t)
            t._ckpt_loader_attrs = {
                k: t.__dict__[k] for k in _LOADER_ATTRS if k in t.__dict__
            }
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
    names = set(model._ckpt_param_names)
    replaced = [
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if name in names and not hasattr(p, "_ckpt_layout")
    ]
    if replaced:
        raise RuntimeError(
            f"direct reload: post-load processing replaced the parameter object "
            f"of {len(replaced)} checkpoint tensors (e.g. {replaced[:3]}); keep "
            "it with `replace_parameter` / `retire_parameter` instead"
        )


def _foreign_modules(model: torch.nn.Module) -> set[torch.nn.Module]:
    """Modules all of whose own tensors another model owns: that model's
    updates refresh them, so this model's finish must not process them."""
    owned = model._ckpt_tensor_ids
    return {
        module
        for module in model.modules()
        if (ids := {id(t) for _, t in _own_tensors(module)}) and not ids & owned
    }


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
        if id(t) not in model._ckpt_tensor_ids:
            continue  # another model's (a target lm_head shared into a draft)
        if _layout(t) == layout:
            continue
        shape, stride, dtype, offset = layout
        storage = t.untyped_storage()
        span = storage_extent(torch.empty_strided(shape, stride, device="meta"))
        need = (offset + span) * dtype.itemsize
        if need > storage.nbytes():
            raise RuntimeError(
                f"direct reload: {name} storage has {storage.nbytes()} bytes, "
                f"its checkpoint layout needs {need}"
            )
        views.append((t, storage_view(storage, offset, shape, stride, dtype)))
    return views


@torch.no_grad()
def begin_direct_reload(model: torch.nn.Module) -> None:
    """Make every tensor writable by its checkpoint weight loader.

    Everything that can refuse the reload is checked first and raises with
    the model untouched. Once this returns, the model holds checkpoint-layout
    bytes until a finish succeeds."""
    if not hasattr(model, "_ckpt_param_names"):
        raise RuntimeError("direct reload: no checkpoint layout was recorded")
    _check_params_kept(model)
    views = _checkpoint_views(model)

    # nothing above modified the model
    model._direct_reload_pending = True
    model._direct_reload_begun = True  # one finish per begin
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    for _, t in _loadable_tensors(model):
        attrs = getattr(t, "_ckpt_loader_attrs", None)
        if attrs is None or id(t) not in model._ckpt_tensor_ids:
            continue
        for k in _LOADER_ATTRS:
            if k in attrs:
                t.__dict__[k] = attrs[k]
            else:
                t.__dict__.pop(k, None)
    logger.info(
        "direct reload: %d tensors viewed back to checkpoint layout", len(views)
    )


@torch.no_grad()
def finish_direct_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Rerun cold-start post-load processing."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    if not model.__dict__.pop("_direct_reload_begun", False):
        # e.g. a retry after a failed finish, which may have processed some
        # modules already: only a new full update repairs the model
        raise RuntimeError(
            "direct reload: finish without a begin; start a new full update"
        )
    if device.type != "cpu":
        torch.accelerator.synchronize()  # engines may write on other streams

    # One-shot "already processed" guards would make the rerun below a no-op
    # and leave the freshly written checkpoint layout in place. Only quant
    # methods' guards: other modules set it for other reasons.
    foreign = _foreign_modules(model)
    for module in model.modules():
        if module not in foreign and isinstance(
            getattr(module, "quant_method", None), QuantizeMethodBase
        ):
            module.__dict__.pop("_already_called_process_weights_after_loading", None)

    process_weights_after_loading(model, model_config, device, foreign)
    model._direct_reload_pending = False
