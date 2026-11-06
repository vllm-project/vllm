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
tensor of each module whose PWAL changes its layout or is not idempotent, and
of every ``padding_fill_params()`` parameter, which ``begin`` zero-fills whole.
"""

from collections.abc import Iterator

import torch

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    storage_view,
)

logger = init_logger(__name__)

__all__ = [
    "begin_direct_reload",
    "check_direct_reload_support",
    "direct_reload_active",
    "direct_reload_enabled",
    "finish_direct_reload",
    "loadable_tensors",
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


def direct_reload_active(model: torch.nn.Module) -> bool:
    """Whether ``model`` is between ``begin_direct_reload`` and its finish."""
    return getattr(model, "_direct_reload_begun", False)


def _layout(t: torch.Tensor) -> tuple:
    return tuple(t.shape), t.stride(), t.dtype, t.storage_offset()


def loadable_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
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


def _config_refusal(vllm_config: VllmConfig) -> str | None:
    """Why ``vllm_config`` cannot reload directly. Checked with the model, so
    a server that never updates its weights starts with any of these."""
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    offload = vllm_config.offload_config
    if (
        offload.offload_backend != "auto"
        or offload.prefetch.offload_group_size > 0
        or offload.uva.cpu_offload_gb > 0
    ):
        # CPU/UVA offload gives parameters new storage on every post-load
        # processing; prefetch offload shares one buffer between layers
        return "weight offloading is not supported"
    model_config = vllm_config.model_config
    if model_config is not None and model_config.quantization == "torchao":
        # its load_weights runs a layerwise reload, and its post-load
        # processing replaces the weight parameter
        return "torchao quantization is not supported"
    quant_config = vllm_config.quant_config
    if isinstance(quant_config, OnlineQuantizationConfig) or (
        getattr(quant_config, "online_quantization_config", None) is not None
    ):
        # `finish` clears the guard that keeps online quantization from
        # quantizing its already-quantized bytes again
        return "online quantization is not supported"
    if vllm_config.lora_config is not None:
        # LoRA wraps layers after cold start; their wrappers forward
        # `quant_method`, so a rerun would process each base layer twice
        return "LoRA is not supported"
    spec = vllm_config.speculative_config
    if spec is not None and spec.draft_lm_head_quantization is not None:
        # after `begin` the draft's quantized lm_head is a view of the target
        # lm_head's storage: post-load processing would write it there
        return "draft_lm_head_quantization is not supported"
    return None


def check_direct_reload_support(
    model: torch.nn.Module, vllm_config: VllmConfig | None = None
) -> None:
    """Raise if ``model`` (or ``vllm_config``, by default the current one) is
    known to reload incorrectly with direct reload; warn once per class if the
    model is not validated (`SupportsDirectReload`).

    Called when the weight-transfer engine is created and in
    `begin_direct_reload`. A refusal is read fresh each time: runtime flags
    may change after load (specialized decode built later, mono decode turned
    off). Config-time refusal: sharded_rdt, in VllmConfig."""
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
        FusedMoEModularMethod,
    )
    from vllm.model_executor.models.interfaces import supports_direct_reload

    vllm_config = vllm_config or get_current_vllm_config_or_none()
    if vllm_config is not None and (reason := _config_refusal(vllm_config)):
        raise RuntimeError(f"direct reload: {reason}")
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, FusedMoEModularMethod):  # elastic EP wraps it
            method = method.old_quant_method
        # declared broken (SupportsDirectReload / QuantizeMethodBase; may change
        # at runtime), e.g. a one-shot guard
        for obj in (module, method):
            if getattr(obj, "supports_direct_reload", None) is False:
                reason = getattr(obj, "direct_reload_unsupported_reason", None)
                raise RuntimeError(
                    f"direct reload: {type(obj).__name__} is not supported "
                    f"({name or 'model'}): {reason or 'no reason given'}"
                )
        # fused shared experts quantized online under a pre-quantized method:
        # post-load processing deletes the staging parameters a reload loads
        if getattr(module, "shared_expert_online_loader", None) is not None:
            raise RuntimeError(
                f"direct reload: online quantization is not supported ({name})"
            )
    if not supports_direct_reload(type(model)):
        logger.warning_once(
            "direct reload: %s is not validated with direct reload (it does "
            "not declare SupportsDirectReload); correctness after a weight "
            "update is not guaranteed. Use VLLM_RELOAD_MODE=layerwise if "
            "outputs look wrong.",
            type(model).__name__,
        )


def record_checkpoint_layout(model: torch.nn.Module) -> None:
    """Cold start, before post-load processing: remember each tensor's
    checkpoint layout and loader metadata, which tensors this model owns (those
    no other model recorded first), and which parameter names hold checkpoint
    tensors.
    Runs once; reloads reuse the cold-start record."""
    if hasattr(model, "_ckpt_param_names") or is_weights_pre_processed():
        return  # pre-processed weights are already in runtime layout
    # a model owns the tensors it records first; `begin` refuses a shared one
    # (a target lm_head in a draft) whose post-load processing re-laid it out
    model._ckpt_owner = object()  # tags owned tensors (ids of freed ones recur)
    for _, t in loadable_tensors(model):
        if not t.is_meta and not hasattr(t, "_ckpt"):
            attrs = {k: t.__dict__[k] for k in _LOADER_ATTRS if k in t.__dict__}
            storage = t.untyped_storage().data_ptr()
            t._ckpt = (model._ckpt_owner, _layout(t), attrs, storage)
    # Before PWAL, which may pad moe_config; by name, so a dropped param isn't kept.
    model._ckpt_padding_fill = [
        (module, name)
        for module in model.modules()
        if hasattr(module, "padding_fill_params")
        for name in module.padding_fill_params()
    ]
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
    for name, t in loadable_tensors(model):
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
        if storage.data_ptr() != ckpt[3]:  # moved, e.g. `set_` into a fused buffer
            offset = t.storage_offset()
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
    if not model.__dict__.get("_direct_reload_supported"):
        # cache a pass only: a refusal may clear (mono decode turned off)
        check_direct_reload_support(model)
        model._direct_reload_supported = True

    # nothing above modified the model
    model._direct_reload_begun = True  # one finish per begin
    for t, view in views:
        t.data = view  # no zeroing: the full sync rewrites the checkpoint region
    for _, t in loadable_tensors(model):
        if not _owned(model, t):
            continue
        for k in _LOADER_ATTRS:
            t.__dict__.pop(k, None)
        t.__dict__.update(t._ckpt[2])
    # Reset padding the loaders skip; erases the checkpoint region, so all is resent.
    for module, name in model._ckpt_padding_fill:
        t = module._parameters.get(name)
        if t is not None and _owned(model, t):
            t.data.zero_()
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
