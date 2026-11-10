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
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    storage_view,
)

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "direct_reload_enabled",
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
    """Whether weight updates reload in place (``VLLM_RELOAD_MODE=direct``)."""
    return envs.VLLM_RELOAD_MODE == "direct"


def _layout(t: torch.Tensor) -> tuple:
    return tuple(t.shape), t.stride(), t.dtype, t.storage_offset()


def _loadable_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters and persistent buffers, each object once."""
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        persistent = (
            (n, b)
            for n, b in module._buffers.items()
            if n not in module._non_persistent_buffers_set
        )
        for name, tensor in [*module._parameters.items(), *persistent]:
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            yield (f"{module_name}.{name}" if module_name else name), tensor


def _check_supported_modules(model: torch.nn.Module) -> None:
    # config-level refusals: VllmConfig._verify_direct_reload_compat
    from vllm.model_executor.layers.quantization.humming import (
        HummingLinearMethod,
        HummingMoEMethod,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

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
        if isinstance(method, (HummingLinearMethod, HummingMoEMethod)):
            # its post-load processing runs once per quant method (a guard on
            # the method, not the module): a reload would keep checkpoint bytes
            raise RuntimeError(f"direct reload: humming is not supported ({name})")


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout and loader metadata, which tensors this model owns (those
    no other model recorded first), and which parameter names hold checkpoint
    tensors.
    Runs once; reloads reuse the cold-start record."""
    if hasattr(model, "_ckpt_param_names") or is_weights_pre_processed():
        return  # pre-processed weights are already in runtime layout
    _check_supported_modules(model)
    # a model owns the tensors it records first; `begin` refuses a shared one
    # (a target lm_head in a draft) whose post-load processing re-laid it out
    model._ckpt_owner = object()  # tags owned tensors (ids of freed ones recur)
    for _, t in _loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt"):
            attrs = {k: t.__dict__[k] for k in _LOADER_ATTRS if k in t.__dict__}
            t._ckpt = (model._ckpt_owner, _layout(t), attrs)
    # every name, aliases included: post-load processing may replace either
    model._ckpt_param_names = frozenset(
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if hasattr(p, "_ckpt")
    )


def _check_params_kept(model: torch.nn.Module) -> None:
    """Raise if post-load processing registered a new parameter under a
    checkpoint name: `begin` cannot view it back, and the loader would write
    checkpoint bytes into its runtime layout."""
    replaced = [
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if name in model._ckpt_param_names and not hasattr(p, "_ckpt")
    ]
    if replaced:
        raise RuntimeError(
            f"direct reload: post-load processing replaced the parameter object "
            f"of {len(replaced)} checkpoint tensors (e.g. {replaced[:3]}); keep "
            "it with `replace_parameter` / `retire_parameter` instead"
        )


def _owned(model: torch.nn.Module, t: torch.Tensor) -> bool:
    ckpt = getattr(t, "_ckpt", None)
    return ckpt is not None and ckpt[0] is model._ckpt_owner


def _checkpoint_views(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(tensor, view of its storage in its checkpoint layout) for every
    tensor post-load processing re-laid out; raises if a storage cannot
    hold its checkpoint layout."""
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, t in _loadable_tensors(model):
        ckpt = getattr(t, "_ckpt", None)
        if ckpt is None:
            continue  # no layout: created by post-load processing, derived
        if _layout(t) == ckpt[1]:
            continue
        if ckpt[0] is not model._ckpt_owner:
            # shared from another model (a target lm_head in a draft): this
            # finish would reprocess bytes a draft update does not resend
            raise RuntimeError(
                f"direct reload: {name} is shared from another model and its "
                "post-load processing changed its layout"
            )
        shape, stride, dtype, offset = ckpt[1]
        storage = t.untyped_storage()
        span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
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
    model._direct_reload_begun = True  # one finish per begin
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    for _, t in _loadable_tensors(model):
        if not _owned(model, t):
            continue
        for k in _LOADER_ATTRS:
            t.__dict__.pop(k, None)
        t.__dict__.update(t._ckpt[2])
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
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), QuantizeMethodBase):
            module.__dict__.pop("_already_called_process_weights_after_loading", None)

    process_weights_after_loading(model, model_config, device)
