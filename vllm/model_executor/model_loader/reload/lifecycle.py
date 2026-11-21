# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Begin and complete a checkpoint-format weight update in the reload mode
the model's cold start fixed."""

import math

import torch

from vllm.config import ModelConfig

from .direct import begin_direct_reload, finish_direct_reload, reloads_directly
from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload

__all__ = ["finish_reload", "start_reload", "weight_update_active"]


_weight_update_active = False


def weight_update_active() -> bool:
    """Whether a weight update is running, when one `load_weights` call may hold
    only part of the checkpoint. Process-wide, so a module nested in the model
    passed to `start_reload` sees it too."""
    return _weight_update_active


def _kv_scales(model: torch.nn.Module) -> dict[str, tuple[float, float, float]]:
    """Host q/k/v scales of the attention layers with a quantized KV cache."""
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache

    return {
        name: (module._q_scale_float, module._k_scale_float, module._v_scale_float)
        for name, module in model.named_modules()
        if (impl := getattr(module, "impl", None)) is not None
        and is_quantized_kv_cache(getattr(impl, "kv_cache_dtype", "auto"))
    }


def _check_frozen_kv_scales(model: torch.nn.Module) -> None:
    """Refuse an update that changed KV-cache scales a decode kernel keeps from
    startup (see `AttentionImplBase.frozen_kv_scales`)."""
    from vllm.config import get_current_vllm_config_or_none

    startup = model.__dict__.get("_startup_kv_scales", {})
    config = get_current_vllm_config_or_none()
    full_graphs = (
        config is not None
        and config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    )
    for name, scales in _kv_scales(model).items():
        impls = model.get_submodule(name).impl.get_impl_variants()
        frozen = [
            impl
            for impl in impls
            if impl.frozen_kv_scales == "always"
            or (impl.frozen_kv_scales == "full_graph" and full_graphs)
        ]
        if name not in startup or not frozen:
            continue
        names = set().union(*(impl.frozen_kv_scale_names or "qkv" for impl in frozen))
        # Layerwise reload rounds scales to BF16; ignore that.
        if all(
            math.isclose(new, old, rel_tol=2**-8)
            for n, new, old in zip("qkv", scales, startup[name])
            if n in names
        ):
            continue
        raise RuntimeError(
            f"This weight update changed the FP8 KV-cache scales of {name} "
            f"(q/k/v {startup[name]} -> {scales}), but its decode keeps the "
            "scales it had at startup (as host values that FULL CUDA graphs "
            "captured, or in tables built once). Restart the engine on the new "
            "checkpoint, or start it with cudagraph_mode=PIECEWISE or with "
            "weight_transfer_config set (some backends then read device-side "
            "scales)."
        )


def start_reload(model: torch.nn.Module) -> None:
    """Begin a weight update in the configured mode."""
    global _weight_update_active
    # The KV scales FULL CUDA graphs captured at startup, kept across updates.
    if "_startup_kv_scales" not in model.__dict__:
        model.__dict__["_startup_kv_scales"] = _kv_scales(model)
    if reloads_directly(model):
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)
    _weight_update_active = True


def finish_reload(
    model: torch.nn.Module, model_config: ModelConfig, device: torch.device
) -> None:
    """Complete the weight update `start_reload` began."""
    global _weight_update_active
    _weight_update_active = False
    if reloads_directly(model):
        finish_direct_reload(model, model_config, device)
    else:
        finalize_layerwise_reload(model, model_config)
    _check_frozen_kv_scales(model)
