# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the INT4 shadow checkpoint and attach it to a LoRA-wrapped model.

Residency model: the INT4 model is loaded through the regular model loader
(inside the runner's ``weights`` CuMem pool, so it sleeps and wakes with the
BF16 weights), its GPTQ-packed linears are matched by module name onto the
BF16 model's LoRA wrappers, and only the attached linears are kept alive in an
:class:`Int4ShadowLayerStore` registered as a submodule of the model. The
temporary INT4 model is then dropped.

Supported shadow formats are the GPTQ packings -- Intel AutoRound
``auto_round:auto_gptq`` and compressed-tensors ``pack-quantized`` -- and
ModelOpt NVFP4. AWQ is refused: its activation-aware scales live in the norms
next to the linears, so a W4 linear against the BF16 norm is a different model.

Nothing in the residency or binding path is format-aware: a binding holds two
``LinearBase`` objects and the forward runs through whichever is active, using
that layer's own ``quant_method``. Adding NVFP4 is therefore a matter of letting
it past the format gate and recognising its parameters, not of new dispatch.
"""

from __future__ import annotations

import gc
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.dual_precision.binding import (
    DualPrecisionBinding,
    DualPrecisionState,
    find_lora_wrappers,
    install_binding,
    mark_lifecycle_event,
    set_dual_precision_state,
)
from vllm.model_executor.dual_precision.policy_layers import (
    format_layer_indices,
    is_quantized_shadow_layer,
    resolve_bf16_layer_indices,
    should_attach_int4_shadow,
    transformer_layer_index,
)
from vllm.model_executor.dual_precision.validation import (
    MAX_LIFECYCLE_PROBES,
    LifecycleProbe,
    ShadowValidation,
    compare_shadow_numerics,
    log_shadow_validation,
    record_lifecycle_probe,
    sanity_probe_shadow,
)
from vllm.model_executor.layers.linear import LinearBase

logger = init_logger(__name__)

SHADOW_MODULE_NAME = "_vllm_dual_precision_int4_model"
"""Attribute under which the shadow store hangs on the model.

Contract with verl's ``_hide_dual_precision_shadow_model`` (weight sync pops
this submodule while it re-runs ``process_weights_after_loading``).
"""

_GPTQ_RESOLVED_METHODS = frozenset(
    {"gptq", "gptq_marlin", "auto_gptq", "inc", "compressed-tensors"}
)
_AWQ_RESOLVED_METHODS = frozenset({"awq", "awq_marlin"})
_MODELOPT_RESOLVED_METHODS = frozenset({"modelopt", "modelopt_fp4"})


class Int4ShadowLayerStore(nn.Module):
    """Strongly owns the attached INT4 linears so they share the model's
    lifecycle (CuMem ``weights`` pool, sleep level 1 offload/restore)."""

    def __init__(self, named_layers: list[tuple[str, LinearBase]]) -> None:
        super().__init__()
        self.layer_names = [name for name, _ in named_layers]
        self.layers = nn.ModuleList(layer for _, layer in named_layers)

    def __len__(self) -> int:
        return len(self.layers)


def dual_precision_rollout_enabled() -> bool:
    """Feature gate: ``VLLM_DUAL_PRECISION_ROLLOUT=1``."""
    return bool(envs.VLLM_DUAL_PRECISION_ROLLOUT)


def check_dual_precision_model_runner(vllm_config: VllmConfig) -> None:
    """Refuse engine configurations dual precision does not support.

    Only the V1 ``GPUModelRunner`` attaches the shadow, binds the base
    precision before every forward and captures precision-keyed CUDA graphs.
    The V2 runner (``VLLM_USE_V2_MODEL_RUNNER=1``, or the default for
    unquantized ``Qwen3ForCausalLM``) would silently serve BF16 for every
    step; fail at worker init instead. Speculative decoding, the KV-sharing
    fast-prefill path and data parallelism dispatch extra forwards with
    descriptors that carry no precision (implicitly BF16) or choose the
    precision per DP rank, so they are refused as well.

    With the feature off, a configured ``VLLM_DUAL_PRECISION_POLICY`` on a
    LoRA-enabled engine is refused too: the scheduler would publish ``int4``
    with no shadow attached and every post-switch step would run eager.
    Without a LoRA config nothing could ever bind, so the scheduler-only
    smokes (C4, C7) may set a policy alone.
    """
    if not dual_precision_rollout_enabled():
        if envs.VLLM_DUAL_PRECISION_POLICY and vllm_config.lora_config is not None:
            raise NotImplementedError(
                "VLLM_DUAL_PRECISION_POLICY requires VLLM_DUAL_PRECISION_ROLLOUT=1 "
                "on a LoRA-enabled engine: without the INT4 shadow the "
                "scheduler's int4 steps would have no graph and run eager."
            )
        return
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with the V2 model "
            "runner: the INT4 shadow is attached, bound and graph-captured by "
            "the V1 GPUModelRunner only. Set VLLM_USE_V2_MODEL_RUNNER=0."
        )
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with speculative "
            "decoding: draft forwards carry no base precision."
        )
    if vllm_config.cache_config.kv_sharing_fast_prefill:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with "
            "kv_sharing_fast_prefill: the decoder-portion dispatch carries no "
            "base precision."
        )
    if vllm_config.parallel_config.data_parallel_size > 1:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with data "
            "parallelism: the precision is chosen per DP rank's scheduler and "
            "is not synchronised across ranks."
        )


# --------------------------------------------------------------------------- #
# Config cloning and format validation                                         #
# --------------------------------------------------------------------------- #


def clone_init_dataclass(config_obj: Any, **overrides: Any) -> Any:
    """Re-construct ``config_obj`` from its init fields with overrides."""
    config_cls = type(config_obj)
    kwargs = {
        field.name: getattr(config_obj, field.name)
        for field in fields(config_obj)
        if field.init
    }
    kwargs.update(overrides)
    return config_cls(**kwargs)


def validate_shadow_quantization(
    hf_quant_config: dict[str, Any] | None, resolved_method: str | None
) -> str:
    """Accept GPTQ packings and ModelOpt NVFP4; reject the rest clearly.

    Returns a short label of the accepted format for logging.
    """
    if not hf_quant_config:
        raise ValueError(
            "VLLM_DUAL_PRECISION_INT4_MODEL must point at a quantized checkpoint "
            "(no quantization_config found)."
        )
    quant_method = str(hf_quant_config.get("quant_method", "")).lower()
    resolved = (resolved_method or "").lower()

    if quant_method in _AWQ_RESOLVED_METHODS or resolved in _AWQ_RESOLVED_METHODS:
        raise ValueError(
            "Dual precision shadow checkpoints must be GPTQ-packed; AWQ "
            f"(quant_method={quant_method!r}, resolved={resolved!r}) is not "
            "supported because its activation scales are folded into the "
            "norms of the quantized checkpoint."
        )
    if quant_method == "auto-round":
        packing = str(hf_quant_config.get("packing_format", "auto_round:auto_gptq"))
        backend = str(
            hf_quant_config.get("backend", hf_quant_config.get("vllm_backend", "auto"))
        )
        if "awq" in packing.lower() or "awq" in backend.lower():
            raise ValueError(
                "Dual precision shadow checkpoints must be GPTQ-packed; "
                f"AutoRound packing_format={packing!r} backend={backend!r} is AWQ."
            )
        return f"auto-round:{packing}"
    if quant_method == "compressed-tensors":
        fmt = str(hf_quant_config.get("format", "")).lower()
        groups = hf_quant_config.get("config_groups") or {}
        group_formats = {
            str(group.get("format", "")).lower()
            for group in groups.values()
            if isinstance(group, dict)
        }
        if fmt != "pack-quantized" and "pack-quantized" not in group_formats:
            raise ValueError(
                "Dual precision shadow checkpoints must be compressed-tensors "
                f"pack-quantized (GPTQ); got format={fmt!r}."
            )
        return "compressed-tensors:pack-quantized"
    if (
        quant_method in _MODELOPT_RESOLVED_METHODS
        or resolved in _MODELOPT_RESOLVED_METHODS
    ):
        return validate_modelopt_shadow(hf_quant_config, quant_method, resolved)
    if quant_method in ("gptq", "gptq_marlin") or resolved in _GPTQ_RESOLVED_METHODS:
        return f"gptq:{resolved or quant_method}"
    raise ValueError(
        "Dual precision shadow checkpoints must be GPTQ-packed (AutoRound "
        "auto_round:auto_gptq or compressed-tensors pack-quantized) or "
        "ModelOpt NVFP4; got "
        f"quant_method={quant_method!r} (resolved {resolved!r})."
    )


def modelopt_weight_spec(hf_quant_config: dict[str, Any]) -> dict[str, Any]:
    """The ``weights`` spec of a ModelOpt config, from whichever shape it uses."""
    groups = hf_quant_config.get("config_groups") or {}
    for group in groups.values():
        if isinstance(group, dict) and isinstance(group.get("weights"), dict):
            return group["weights"]
    quantization = hf_quant_config.get("quantization")
    if isinstance(quantization, dict):
        return quantization
    return {}


def validate_modelopt_shadow(
    hf_quant_config: dict[str, Any], quant_method: str, resolved: str
) -> str:
    """Accept a ModelOpt NVFP4 shadow; refuse the FP8 ModelOpt checkpoints.

    ModelOpt covers both FP8 and NVFP4 under one ``quant_method``, and only the
    4-bit float one is a W4 shadow. vLLM resolves NVFP4 to ``modelopt_fp4``;
    older exports only say so in the weight spec (``num_bits`` 4, ``type``
    ``float``) or in ``quant_algo``.
    """
    algo = str(hf_quant_config.get("quant_algo", "")).upper()
    weights = modelopt_weight_spec(hf_quant_config)
    num_bits = weights.get("num_bits")
    weight_type = str(weights.get("type", "")).lower()

    is_nvfp4 = (
        resolved == "modelopt_fp4"
        or quant_method == "modelopt_fp4"
        or algo.startswith("NVFP4")
        or (num_bits == 4 and weight_type == "float")
    )
    if not is_nvfp4:
        raise ValueError(
            "Dual precision ModelOpt shadow checkpoints must be NVFP4 "
            f"(4-bit float); got quant_algo={algo or None!r}, "
            f"num_bits={num_bits!r}, type={weight_type or None!r} "
            f"(resolved {resolved!r}). ModelOpt FP8 is not a W4 shadow."
        )
    return "modelopt:nvfp4"


SHADOW_LOAD_FORMAT = "auto"
"""The shadow is always loaded from its checkpoint, whatever the engine does."""


def make_shadow_load_config(load_config: Any) -> Any:
    """Clone the engine's ``LoadConfig`` for the shadow with ``load_format``
    forced to :data:`SHADOW_LOAD_FORMAT`.

    verl's rollout default (``rollout.load_format: dummy``, the trainer syncs
    the real base weights later) used to reach the shadow through the plain
    config clone, so the INT4 store was ``DummyModelLoader`` noise and every
    INT4-phase token was garbage (integration defect 1). The shadow checkpoint
    is real weights by definition, so its loader is always ``auto``; a dummy
    engine is announced at WARNING so the log explains the extra load time.
    """
    engine_format = str(load_config.load_format).lower()
    if engine_format == "dummy":
        logger.warning(
            "Dual precision: engine load_format=dummy, but the INT4 shadow is "
            "loaded with load_format=%s regardless (the shadow checkpoint is "
            "real weights; a dummy shadow would serve random INT4 tokens).",
            SHADOW_LOAD_FORMAT,
        )
    elif engine_format != SHADOW_LOAD_FORMAT:
        logger.info(
            "Dual precision: engine load_format=%s; the INT4 shadow uses "
            "load_format=%s.",
            engine_format,
            SHADOW_LOAD_FORMAT,
        )
    return clone_init_dataclass(load_config, load_format=SHADOW_LOAD_FORMAT)


def make_int4_vllm_config(vllm_config: VllmConfig, int4_model: str) -> VllmConfig:
    """Clone ``vllm_config`` so it loads ``int4_model`` with auto-detected
    quantization, its own compilation config (own static forward context)
    and its own load config (:func:`make_shadow_load_config`).
    """
    if not int4_model:
        raise ValueError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 requires "
            "VLLM_DUAL_PRECISION_INT4_MODEL to point at the INT4 checkpoint."
        )

    int4_model_config = clone_init_dataclass(
        vllm_config.model_config,
        model=int4_model,
        model_weights="",
        hf_config_path=int4_model,
        quantization=None,
    )
    shadow_format = validate_shadow_quantization(
        int4_model_config.model_arch_config.quantization_config,
        int4_model_config.quantization,
    )
    logger.info("Dual precision shadow checkpoint format: %s.", shadow_format)
    int4_compilation_config = clone_init_dataclass(vllm_config.compilation_config)
    int4_load_config = make_shadow_load_config(vllm_config.load_config)
    return clone_init_dataclass(
        vllm_config,
        model_config=int4_model_config,
        compilation_config=int4_compilation_config,
        load_config=int4_load_config,
    )


def load_int4_shadow_model(int4_vllm_config: VllmConfig) -> nn.Module:
    from vllm.model_executor.model_loader import get_model_loader

    load_format = str(int4_vllm_config.load_config.load_format).lower()
    if load_format == "dummy":
        raise RuntimeError(
            "Dual precision INT4 shadow must be loaded from its checkpoint; "
            "load_format=dummy would serve random INT4 weights. Build the "
            "shadow config with make_int4_vllm_config."
        )
    logger.info(
        "Loading dual precision INT4 shadow model from %s (load_format=%s)...",
        int4_vllm_config.model_config.model,
        load_format,
    )
    loader = get_model_loader(int4_vllm_config.load_config)
    int4_model = loader.load_model(
        vllm_config=int4_vllm_config, model_config=int4_vllm_config.model_config
    )
    int4_model.eval()
    for param in int4_model.parameters():
        param.requires_grad_(False)
    return int4_model


# --------------------------------------------------------------------------- #
# Attach                                                                       #
# --------------------------------------------------------------------------- #


def module_tensor_bytes(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1 << 30):.2f} GiB"


def register_shadow_store(model: nn.Module, store: Int4ShadowLayerStore) -> None:
    """Hang the store on the model (after LoRA wrapping, so the LoRA manager
    never sees the shadow linears) under :data:`SHADOW_MODULE_NAME`."""
    if SHADOW_MODULE_NAME in model._modules:
        raise RuntimeError("Dual precision shadow store is already registered.")
    model.add_module(SHADOW_MODULE_NAME, store)


def attach_shadow_layers(
    model: nn.Module,
    int4_model: nn.Module,
    *,
    bf16_layer_policy: str,
    module_policy: str,
    num_layers: int,
    static_forward_context: dict[str, Any],
    dtype: torch.dtype,
    validate_shadow: bool = False,
    validate_lifecycle: bool = False,
    engine_load_format: str = SHADOW_LOAD_FORMAT,
    shadow_load_format: str = SHADOW_LOAD_FORMAT,
) -> DualPrecisionState:
    """Match, bind and store. Pure of env access; the runner entry point
    :func:`attach_dual_precision` supplies the knobs.

    The always-on sanity probe (:func:`sanity_probe_shadow`) runs here on the
    first attached layer before anything is installed, unless
    ``engine_load_format`` is ``dummy``: then the BF16 twin is noise until the
    trainer syncs weights, and the probe is deferred to the first INT4 bind
    after a weight-load lifecycle event.
    """
    bf16_layer_indices = resolve_bf16_layer_indices(bf16_layer_policy, num_layers)
    logger.info(
        "Dual precision BF16 layer policy %r resolved to transformer blocks %s of %d.",
        bf16_layer_policy,
        format_layer_indices(bf16_layer_indices),
        num_layers,
    )
    logger.info("Dual precision INT4 module policy: %s.", module_policy)

    int4_linears = {
        name: module
        for name, module in int4_model.named_modules()
        if isinstance(module, LinearBase)
    }
    quantized = {
        name: layer
        for name, layer in int4_linears.items()
        if is_quantized_shadow_layer(layer)
    }
    wrappers = find_lora_wrappers(model)
    wrapped_base_names = {f"{name}.base_layer" for name in wrappers}

    # Counts follow the archived log line and cover every BF16 ``LinearBase``
    # (wrapped or not): ``fallback`` has no quantized peer, ``policy_bf16`` is
    # excluded by the layer/module policy, ``attached`` joins the store, and
    # ``unwrapped`` would attach but has no LoRA wrapper to switch through.
    attached = policy_bf16 = fallback = unwrapped = 0
    bare_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LinearBase) and name not in wrapped_base_names
    ]
    for name in list(wrappers) + bare_linears:
        if name not in quantized:
            fallback += 1
        elif not should_attach_int4_shadow(name, bf16_layer_indices, module_policy):
            policy_bf16 += 1
        elif name in wrappers:
            attached += 1
        else:
            unwrapped += 1

    if attached == 0:
        # Checked before anything is installed so a failed attach leaves the
        # model exactly as it was (no overrides, no bindings, no store).
        raise RuntimeError(
            "Dual precision INT4 shadow model loaded, but no quantized LinearBase "
            "layers were attached to a LoRA wrapper."
        )

    plan: list[tuple[str, nn.Module, LinearBase | None]] = []
    for name, wrapper in wrappers.items():
        int4_layer = quantized.get(name)
        if int4_layer is None or not should_attach_int4_shadow(
            name, bf16_layer_indices, module_policy
        ):
            int4_layer = None
        plan.append((name, wrapper, int4_layer))

    # Always-on sanity probe on the first attached layer, still before any
    # override is installed so a failure leaves the model untouched.
    deferred = str(engine_load_format).lower() == "dummy"
    first_name, first_wrapper, first_int4 = next(
        item for item in plan if item[2] is not None
    )
    if deferred:
        logger.warning(
            "Dual precision: engine load_format=dummy, so the INT4 shadow "
            "sanity probe on %s%s is deferred to the first INT4 bind after the "
            "base weights are loaded.",
            first_name,
            " (and the numerical shadow validation)" if validate_shadow else "",
        )
    else:
        sanity_probe_shadow(
            first_name,
            first_wrapper.base_layer,
            first_int4,
            dtype,
            shadow_load_format=shadow_load_format,
            when="at attach",
        )

    bindings: list[DualPrecisionBinding] = []
    attached_layers: list[tuple[str, LinearBase]] = []
    validations: list[ShadowValidation] = []
    probes: list[LifecycleProbe] = []
    for name, wrapper, int4_layer in plan:
        layer_index = transformer_layer_index(name)
        if int4_layer is not None:
            attached_layers.append((name, int4_layer))
            if validate_shadow and not deferred:
                validations.append(
                    compare_shadow_numerics(name, wrapper.base_layer, int4_layer, dtype)
                )
            if validate_lifecycle and len(probes) < MAX_LIFECYCLE_PROBES:
                probes.append(record_lifecycle_probe(name, int4_layer, dtype))
        bindings.append(
            install_binding(
                wrapper, name, int4_layer, layer_index, static_forward_context
            )
        )

    store = Int4ShadowLayerStore(attached_layers)
    state = DualPrecisionState(
        shadow_store=store,
        bindings=bindings,
        num_shadow_linears=len(int4_linears),
        attached=attached,
        policy_bf16=policy_bf16,
        fallback=fallback,
        unwrapped=unwrapped,
        shadow_bytes=module_tensor_bytes(store),
        lifecycle_probes=probes,
        probe_dtype=dtype,
        shadow_load_format=str(shadow_load_format),
        sanity_probe_pending=deferred,
        shadow_validation_pending=deferred and validate_shadow,
    )
    set_dual_precision_state(model, state)
    register_shadow_store(model, store)

    logger.info(
        "Loaded %d quantized shadow linear layers; attached %d of them, "
        "kept %d quantized layers in BF16 by policy, and left %d layers in "
        "BF16 because no shadow was available.",
        state.num_shadow_linears,
        attached,
        policy_bf16,
        fallback,
    )
    if unwrapped:
        logger.warning(
            "Dual precision: %d quantized linears have no LoRA wrapper and stay BF16.",
            unwrapped,
        )
    logger.info(
        "Dual precision steady-state INT4 shadow store holds %d layers and %s "
        "of parameters/buffers.",
        attached,
        format_gib(state.shadow_bytes),
    )
    log_shadow_validation(validations)
    return state


LOAD_WEIGHTS_ATTR = "load_weights"


def wrap_load_weights_for_lifecycle(model: nn.Module) -> None:
    """Make every ``model.load_weights(...)`` call mark a ``load_weights``
    lifecycle event.

    verl's colocated worker extension streams the trainer's base weights with
    ``model.load_weights`` straight on the model (not through the runner), so
    this instance-level wrapper is the only vLLM-side hook that sees it. It is
    a plain ``__dict__`` entry (never ``_modules``); ``torch.compile`` only
    traces ``forward``.
    """
    original = getattr(model, LOAD_WEIGHTS_ATTR, None)
    if original is None or LOAD_WEIGHTS_ATTR in model.__dict__:
        return

    def load_weights(*args: Any, **kwargs: Any) -> Any:
        mark_lifecycle_event(model, "load_weights")
        return original(*args, **kwargs)

    load_weights.__wrapped__ = original  # type: ignore[attr-defined]
    object.__setattr__(model, LOAD_WEIGHTS_ATTR, load_weights)


def attach_dual_precision(
    model: nn.Module, vllm_config: VllmConfig
) -> DualPrecisionState | None:
    """Runner entry point; call once, after ``load_lora_model``.

    Returns ``None`` (and changes nothing) when dual precision is off or the
    engine has no LoRA config.
    """
    if not dual_precision_rollout_enabled():
        return None
    if vllm_config.lora_config is None:
        logger.warning(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 but LoRA is not enabled; the INT4 "
            "shadow model is not loaded."
        )
        return None

    int4_vllm_config = make_int4_vllm_config(
        vllm_config, envs.VLLM_DUAL_PRECISION_INT4_MODEL
    )
    int4_model = load_int4_shadow_model(int4_vllm_config)
    try:
        state = attach_shadow_layers(
            model,
            int4_model,
            bf16_layer_policy=envs.VLLM_DUAL_PRECISION_BF16_LAYERS,
            module_policy=envs.VLLM_DUAL_PRECISION_INT4_MODULES,
            num_layers=vllm_config.model_config.get_total_num_hidden_layers(),
            static_forward_context=vllm_config.compilation_config.static_forward_context,
            dtype=vllm_config.model_config.dtype,
            validate_shadow=envs.VLLM_DUAL_PRECISION_VALIDATE_SHADOW,
            validate_lifecycle=envs.VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE,
            engine_load_format=str(vllm_config.load_config.load_format),
            shadow_load_format=str(int4_vllm_config.load_config.load_format),
        )
        wrap_load_weights_for_lifecycle(model)
    finally:
        # Only the attached linears survive, owned by the shadow store.
        del int4_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state
