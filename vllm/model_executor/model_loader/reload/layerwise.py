# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .meta import (
    SKIP_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
)
from .source import get_current_source_key
from .types import (
    LayerReloadingInfo,
    LoadManifestGroupScope,
    LoadManifestReport,
    LoadManifestScope,
)
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
    has_device_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
    "record_load_consumption",
    "finalize_load_recording",
    "get_layer_completion_findings",
    "get_load_manifest_report",
    "record_dummy_load_manifest",
    "record_direct_load_consumption",
    "record_external_tensor_manifest",
    "has_dummy_load_manifest",
]


# Global dict storing information used for layerwise restoring, loading, and processing.
# For more information regarding what info is stored when, see `LayerReloadingInfo`
#
# Use a weak ref dictionary so that modules can be freed when the model is freed.
# Values are sanitized from references to the layer key in order to avoid circular refs
LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)

# Global set used to track loading for logging purposes only
LOADING_LAYERS: WeakSet[torch.nn.Module] = WeakSet()

# Per-layer disagreements between the numel completion criterion and the
# observed-key-set criterion, found during the current reload. Cleared at
# reload start, read after finalize. Dual-run signal only -- numel stays
# authoritative until the set criterion has a release of agreement.
_LAYER_COMPLETION_FINDINGS: list[str] = []
_DUMMY_BASELINE_FAILURES: list[str] = []
# The per-layer receipt sets are transient and are cleared at finalize. Keep
# one aggregate per model so a controller can inspect it afterwards without
# mixing staged/replaced models that temporarily share a worker process.
_LAST_RELOAD_RECEIVED_EVENT_COUNTS: WeakKeyDictionary[torch.nn.Module, int] = (
    WeakKeyDictionary()
)


def get_layer_completion_findings() -> list[str]:
    """Where the numel and observed-key-set completion criteria disagreed
    during the last reload."""
    return list(_LAYER_COMPLETION_FINDINGS)


def _get_load_manifest_scope() -> LoadManifestScope:
    """Best-effort distributed coordinates without requiring distributed init."""
    try:
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            get_pp_group,
            get_tp_group,
            get_world_group,
        )

        world = get_world_group()

        def group_scope(get_group: Callable) -> LoadManifestGroupScope | None:
            try:
                group = get_group()
            except (AssertionError, RuntimeError):
                return None
            return LoadManifestGroupScope(
                rank=group.rank_in_group,
                world_size=group.world_size,
            )

        return LoadManifestScope(
            global_rank=world.rank,
            global_world_size=world.world_size,
            tp=group_scope(get_tp_group),
            pp=group_scope(get_pp_group),
            dp=group_scope(get_dp_group),
            ep=group_scope(get_ep_group),
        )
    except (AssertionError, RuntimeError):
        return LoadManifestScope()


def get_load_manifest_report(
    model: torch.nn.Module | None = None,
) -> LoadManifestReport:
    """Return this worker's rank-local manifest/reconciliation result.

    The event sets stay process-local by design. Combining them before
    reconciliation would hide a missing TP/PP/EP shard on one rank behind an
    event received by another rank. Executors/controllers should collect one
    report per worker and require every report to be ``ok``.
    """
    # Supplying the model avoids mixing manifests if a process temporarily
    # retains more than one model (tests and staged model replacement can do
    # this). Worker RPC callers should always provide their active model.
    infos = (
        [get_layerwise_info(module) for module in model.modules()]
        if model is not None
        else list(LAYERWISE_INFO.values())
    )
    return LoadManifestReport(
        scope=_get_load_manifest_scope(),
        required_event_count=sum(
            len(info.required_keys or ()) for info in infos
        ),
        received_event_count=(
            _LAST_RELOAD_RECEIVED_EVENT_COUNTS.get(model, 0)
            if model is not None
            else sum(_LAST_RELOAD_RECEIVED_EVENT_COUNTS.values())
        ),
        completion_findings=tuple(_LAYER_COMPLETION_FINDINGS),
    )


def record_load_consumption(model: torch.nn.Module) -> None:
    """Observe the required key set from the first load.

    Wrap every parameter/buffer weight_loader so it records its name into the
    owning layer's ``required_keys`` when it fires. Install this BEFORE the
    first ``load_weights``; call ``finalize_load_recording`` after.

    The observed set is the ground truth of what a correct load consumes, with
    no prediction from layer structure: a tensor that is never routed through a
    loader (EP bookkeeping such as ``_expert_map``, a shared bias copy with no
    checkpoint key) simply never fires, so it is absent from ``required_keys``.
    That is what removes the need for a hand-maintained ``SKIP_TENSORS``
    allowlist -- and it is exactly the "capturing how many weights are loaded
    on first pass" that make_online_process_loader's own comment anticipated.

    Note: parameters must exist when this is called. Quantization weights and
    biases are created at model init (before the first load), so the common
    case is covered; a parameter registered during load itself would be
    missed (a known edge, shared with the reload path's late-registration
    handling).
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        if info.required_keys is None:
            info.required_keys = set()
        for name, tensor in get_layer_tensors(layer).items():
            # Wrap the effective loader (falling back to default_weight_loader,
            # same as the reload path), so params that rely on the default are
            # observed too. A tensor the checkpoint never drives -- a
            # bookkeeping buffer with no key -- simply never has its loader
            # called, so it stays out of required_keys.
            current = getattr(tensor, "weight_loader", None)
            if current is not None and getattr(current, "_vllm_recording",
                                               False):
                continue
            tensor.weight_loader = _make_recording_loader(
                info, name, _get_weight_loader(tensor))


def record_dummy_load_manifest(model: torch.nn.Module) -> None:
    """Establish a rank-local target baseline after dummy initialization.

    Dummy weights have no checkpoint source keys, so inventing source events
    would make the first real RL transfer impossible to reconcile. Instead,
    require every unique local parameter target to be consumed. A successful
    first real load promotes its exact observed source/fragment receipts to the
    normal manifest used by subsequent reloads.
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        # Restore metadata is the authoritative pre-PWAL shape that the
        # layerwise transfer will materialize. Current parameters may already
        # have been packed/replaced by quantization.
        required = {
            name
            for name, param in info.restore_metadata[0].items()
            if param is not None and name not in SKIP_TENSORS
        }
        if required:
            info.required_target_keys = required
            # Empty means "not established from a real source stream yet".
            info.required_keys = set()


def has_dummy_load_manifest(model: torch.nn.Module) -> bool:
    """Whether this model still awaits its first real source-bearing load."""
    return any(
        get_layerwise_info(layer).required_target_keys is not None
        for layer in model.modules()
    )


def record_direct_load_consumption(
    model: torch.nn.Module,
    target_name: str,
    source_key: str,
) -> None:
    """Record a source-to-target event for loaders that write state_dict directly."""
    if "." in target_name:
        module_name, param_name = target_name.rsplit(".", 1)
        layer = model.get_submodule(module_name)
    else:
        layer, param_name = model, target_name
    info = get_layerwise_info(layer)
    event = f"{source_key}=>{param_name}"
    if info.can_load():
        info.received_keys.add(event)
        info.received_target_keys.add(param_name)
    else:
        if info.required_keys is None:
            info.required_keys = set()
        info.required_keys.add(event)


def record_external_tensor_manifest(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    source_scheme: str,
) -> None:
    """Record a post-PWAL tensor manifest supplied by an external loader.

    Some loaders transfer fully processed tensors and never invoke vLLM
    ``weight_loader`` functions. Match their published tensors back to the
    owning module by identity and retain one canonical event per tensor.
    """
    recorded_names: set[str] = set()
    for source_name in tensors:
        target_name = source_name.removesuffix(".__storage")
        if "." in target_name:
            module_name, local_name = target_name.rsplit(".", 1)
            try:
                layer = model.get_submodule(module_name)
            except AttributeError:
                continue
        else:
            layer, local_name = model, target_name
        info = get_layerwise_info(layer)
        if info.required_keys is None:
            info.required_keys = set()
        info.required_keys.add(
            f"{source_scheme}://{source_name}=>{local_name}"
        )
        recorded_names.add(source_name)

    # External registries may use aliases that are not resolvable as module
    # paths. Match those by storage as a fallback.
    names_by_storage = {
        tensor.untyped_storage().data_ptr(): name
        for name, tensor in tensors.items()
        if name not in recorded_names
    }
    seen: set[int] = set()
    for layer in model.modules():
        info = get_layerwise_info(layer)
        for local_name, tensor in get_layer_tensors(layer).items():
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr in seen or storage_ptr not in names_by_storage:
                continue
            seen.add(storage_ptr)
            if info.required_keys is None:
                info.required_keys = set()
            source_name = names_by_storage[storage_ptr]
            info.required_keys.add(
                f"{source_scheme}://{source_name}=>{local_name}"
            )


def _make_recording_loader(info: LayerReloadingInfo, name: str,
                           original: Callable) -> Callable:
    loader_signature = inspect.signature(original)

    @wraps(original, assigned=("__doc__", "__annotations__"))
    def recording_loader(*args, **kwargs):
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        result = original(*args, **kwargs)
        assert info.required_keys is not None
        if _load_call_consumed(bound_args, result):
            info.required_keys.add(_load_event_key(name, bound_args))
        return result

    # A distinct __name__ so _get_original_loader (which unwraps only
    # "online_process_loader") does not strip it; finalize_load_recording
    # removes it explicitly instead.
    recording_loader.__name__ = "recording_loader"
    recording_loader._vllm_recording = True  # type: ignore[attr-defined]
    recording_loader._vllm_orig = original  # type: ignore[attr-defined]
    return recording_loader


_LOAD_FRAGMENT_ARGUMENTS = (
    "loaded_shard_id",
    "loaded_weight_shard_id",
    "shard_id",
    "expert_id",
    "weight_name",
)


def _load_event_key(param_name: str,
                    args: inspect.BoundArguments) -> str:
    """Return the logical target fragment consumed by one loader call.

    A loader invocation is one event regardless of how many internal writes
    it performs. Packed, merged and expert loaders are distinguished by the
    routing arguments already present in their public loader signature.
    Keeping the no-fragment representation equal to ``param_name`` preserves
    compatibility for ordinary parameters and existing diagnostics.
    """
    fragments = []
    for name in _LOAD_FRAGMENT_ARGUMENTS:
        value = args.arguments.get(name)
        if value is not None:
            fragments.append(f"{name}={value!r}")
    target = (param_name if not fragments else
              f"{param_name}[{','.join(fragments)}]")
    source_key = get_current_source_key()
    if source_key is None:
        return target
    return f"{source_key}=>{target}"


def _load_call_consumed(args: inspect.BoundArguments, result: object) -> bool:
    """Whether a conditional loader reports that it consumed this fragment."""
    if args.arguments.get("return_success") is True:
        return result is True
    return True


def finalize_load_recording(model: torch.nn.Module) -> None:
    """Remove the recording wrappers installed by record_load_consumption,
    restoring the original loaders so the reload path is unaffected."""
    for layer in model.modules():
        for name, tensor in get_layer_tensors(layer).items():
            loader = getattr(tensor, "weight_loader", None)
            if loader is not None and getattr(loader, "_vllm_recording", False):
                tensor.weight_loader = loader._vllm_orig


def _check_set_completion(layer: torch.nn.Module,
                          info: LayerReloadingInfo) -> None:
    """At end of reload, record whether all first-load events were received.

    Reconciliation deliberately happens only after the checkpoint iterator is
    exhausted. Numel-driven processing may finish earlier, while a
    checkpoint-backed tensor excluded from meta restoration can legitimately
    arrive later. Numel stays authoritative; this only records discrepancies.
    """
    if info.required_keys is None:
        return  # no observed baseline (first-load recording did not run)
    missing_targets = (
        (info.required_target_keys or set()) - info.received_target_keys
    )
    if missing_targets:
        finding = (
            f"{layer.__class__.__name__}: reload finalized but "
            f"{len(missing_targets)} dummy-baseline target(s) not received: "
            f"{sorted(missing_targets)[:8]}"
        )
        _LAYER_COMPLETION_FINDINGS.append(finding)
        _DUMMY_BASELINE_FAILURES.append(finding)
    missing = info.required_keys - info.received_keys
    if missing:
        _LAYER_COMPLETION_FINDINGS.append(
            f"{layer.__class__.__name__}: reload finalized but "
            f"{len(missing)} observed key(s) not yet received: "
            f"{sorted(missing)[:8]}")
    if info.required_target_keys is not None and not missing_targets:
        # The first complete real transfer turns the provisional dummy target
        # baseline into the exact source/fragment manifest.
        info.required_keys = set(info.received_keys)
        info.required_target_keys = None


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """
    Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.get_default_device(),
        )

    return LAYERWISE_INFO[layer]


def record_metadata_for_reloading(model: torch.nn.Module):
    """
    Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.
    """
    for layer in model.modules():
        info = get_layerwise_info(layer)
        info.restore_metadata = capture_layer_to_meta(layer)
        info.restore_device = torch.get_default_device()


@torch.no_grad()
def initialize_layerwise_reload(model: torch.nn.Module):
    """
    Set up layerwise weight loading with deferred processing.

    Must be called after `record_metadata_for_reloading`. This function:
    1. Saves current kernel tensors for later copying
    2. Restores layer parameters/buffers from metadata (on meta device)
    3. Wraps weight loaders to defer processing until all weights are loaded

    When all weights for a layer are loaded, the wrapped loaders will:
    1. Materialize the layer onto the target device
    2. Load all cached weights
    3. Run quantization processing if applicable
    4. Copy processed values back to original tensor storage
    """
    # disable torchao reloading to avoid infinite recursion
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    if not any(get_layerwise_info(m).can_load() for m in model.modules()):
        _LAYER_COMPLETION_FINDINGS.clear()
        _DUMMY_BASELINE_FAILURES.clear()
        _LAST_RELOAD_RECEIVED_EVENT_COUNTS[model] = 0

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        if info.can_load():
            continue

        # Save current tensors for later copying
        info.kernel_tensors = get_layer_params_buffers(layer)

        # Restore layer parameters/buffers onto meta device
        restore_layer_on_meta(layer, info)

        # Wrap weight loaders to buffer loading
        initialize_online_processing(layer)


def initialize_online_processing(layer: torch.nn.Module):
    """
    Wrap a layer's weight loaders with online processing loaders.
    Called by either `initialize_layerwise_reload` or an online quantization scheme,
    prevents double wrapping in the case of online quantization + reloading

    Args:
        layer: layer whose parameter weight loaders will be wrapped
    """
    info = get_layerwise_info(layer)

    # Track loading progress to determine when to process/copy
    info.received_keys.clear()
    info.received_target_keys.clear()
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
    _wrap_parameters_weight_loader(layer)


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap each parameter's weight loader."""
    info = get_layerwise_info(layer)
    # Note that nested wrapping will occur for shared tensors
    for name, tensor in get_layer_tensors(layer).items():
        if name in SKIP_TENSORS:
            # Some skipped tensors are nevertheless checkpoint-backed. They
            # stay on device because they are aliases or runtime bookkeeping,
            # but their real loader invocation must still produce a receipt.
            # Keep this lightweight observer installed until the *whole*
            # checkpoint iterator is exhausted: such a tensor may arrive
            # after the other tensors made this layer reach numel completion.
            loader = _get_weight_loader(tensor)
            if not getattr(loader, "_vllm_receipt_observer", False):
                tensor.weight_loader = _make_receipt_observer(
                    info, name, _get_original_loader(tensor))
            continue
        if _get_weight_loader(tensor).__name__ != "online_process_loader":
            tensor.weight_loader = make_online_process_loader(layer, name)


def _make_receipt_observer(info: LayerReloadingInfo, param_name: str,
                           original_loader: Callable) -> Callable:
    """Observe a checkpoint load without deferring or counting the tensor."""
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def receipt_observer(*args, **kwargs):
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        result = original_loader(*args, **kwargs)
        if _load_call_consumed(bound_args, result):
            info.received_keys.add(_load_event_key(param_name, bound_args))
            info.received_target_keys.add(param_name)
        return result

    receipt_observer.__name__ = "reload_receipt_observer"
    receipt_observer._vllm_receipt_observer = True  # type: ignore[attr-defined]
    return receipt_observer


def make_online_process_loader(layer: torch.nn.Module, param_name: str) -> Callable:
    """Create a wrapped weight loader that defers processing."""
    info = get_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        if not info.can_load():
            # Unfortunately, some qconfigs are set up to load the same weight
            # multiple times. For example, CT_WNA16 loads `weight_shape` for
            # each of the qkv partitions. This results in layers loading extra
            # weights (beyond load_numel_total) after it's already processed.
            #
            # Best solution is to ensure that `load_numel_total` reflects the
            # actual number of weights loaded, either by modifying qconfigs to
            # create as many weights as loaded (see padding issue as well)
            # or maybe capturing how many weights are loaded on first pass
            #
            # For now, `load_numel_total` is still safe to use as long as
            # there's no way to reach `load_numel_total` without loading all
            # necessary weights. `weight_shape` is very small, so this is safe.
            # see Limitations(4)
            logger.debug("%s: Excessive loading", layer.__class__.__name__)
            return

        # Re-run on each load: layers may register parameters later (e.g., `bias`).
        # Wrap late parameters and refresh `load_numel_total` so processing waits
        # until all parameters are loaded.
        info.load_numel_total = get_layer_size(layer)
        _wrap_parameters_weight_loader(layer)

        # Bind and normalize arguments
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Buffer loaded weights, track loading progress
        info.loaded_weights.append((param_name, bound_args))
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        info.load_numel += num_loaded

        # Observe the same logical target fragment as the first-load wrapper.
        # This distinguishes Q/K/V shards and MoE expert/shard calls that
        # share one packed destination parameter. Internal copy_ calls remain
        # invisible: the loader invocation, not a storage write, is the event.
        if _load_call_consumed(bound_args, ret):
            info.received_keys.add(_load_event_key(param_name, bound_args))
            info.received_target_keys.add(param_name)

        logger.debug(
            "%s: %d / %d",
            layer.__class__.__name__,
            info.load_numel,
            info.load_numel_total,
        )

        # Do not online process attention layers, must wait until finalize
        if isinstance(layer, (Attention, MLAAttention)):
            return ret

        # Log warnings allocating excessive buffers on device
        if has_device_tensors(bound_args):
            LOADING_LAYERS.add(layer)
            if len(LOADING_LAYERS) >= 2:
                names = sorted([layer.__class__.__name__ for layer in LOADING_LAYERS])
                mem_used = sum(
                    get_info_size(LAYERWISE_INFO[layer]) for layer in LOADING_LAYERS
                )
                logger.warning_once(
                    "Allocating %.1f MB of device memory to buffers to load %s layers. "
                    "This extra memory usage can be avoided by ordering weights "
                    "by their parent layer when reloading.",
                    mem_used / 1e6,
                    str(list(names)),
                )

        # Process and copy when all weights are loaded
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

        return ret

    return online_process_loader


def finalize_layerwise_processing(model: torch.nn.Module, model_config: ModelConfig):
    """
    Apply processing to any layers which were not layerwise processed during loading.
    This includes attention layers and layers which have weight elements which are not
    loaded (due to padding).

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    Args:
        model: model to finalize processing for
        model_config: config needed for applying processing to attention layers
    """
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    received_event_count = 0
    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []

    for layer in model.modules():
        info = get_layerwise_info(layer)
        if not info.can_load():
            received_event_count += len(info.received_keys)
            _check_set_completion(layer, info)
            _unwrap_receipt_observers(layer)
            info.reset()
            continue

        # Attention/MLA layers are processed after all other layers
        if isinstance(layer, (Attention, MLAAttention)):
            deferred_attn.append((layer, info))
            continue

        # No weights were loaded
        if info.load_numel <= 0:
            # first load: checkpoint did not contain weights for this layer
            if info.kernel_tensors is None:
                _layerwise_process(layer, info)
                continue

            # reloading: place kernel tensors back as a fallback
            elif info.load_numel_total > 0:  # type: ignore[operator]
                logger.warning("%s: Failed to load weights", layer.__class__.__name__)
                _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info)

        received_event_count += len(info.received_keys)
        _check_set_completion(layer, info)
        _unwrap_receipt_observers(layer)
        info.reset()

    # Process attention layers after all other layers are done
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        received_event_count += len(info.received_keys)
        _check_set_completion(layer, info)
        _unwrap_receipt_observers(layer)
        info.reset()

    LOADING_LAYERS.clear()
    _LAST_RELOAD_RECEIVED_EVENT_COUNTS[model] = received_event_count
    if _DUMMY_BASELINE_FAILURES:
        raise RuntimeError(
            "First real weight load after dummy initialization was incomplete:\n  "
            + "\n  ".join(_DUMMY_BASELINE_FAILURES[:20])
        )


def finalize_layerwise_reload(*args, **kwargs):
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    if info.load_numel > 0 and info.kernel_tensors is not None:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info)
    elif info.load_numel > 0 or info.kernel_tensors is None:
        raise ValueError(
            "Layerwise loading of attention layers is not supported. "
            "Attention must always process after linears."
        )
    else:
        _place_kernel_tensors(layer, info)
    layer.process_weights_after_loading(model_config.dtype)


def _reload_attention_scales(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return

    # Re-create scale Parameters with sentinel values so unloaded scales
    # are correctly detected by process_weights_after_loading
    quant_method.create_weights(layer)

    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        _get_weight_loader(param)(*args.args, **args.kwargs)

    quant_method.process_weights_after_loading(layer)

    _copy_and_restore_kernel_tensors(layer, info)


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """
    Finalize layer loading after all weights have been buffered.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all buffered weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer tensors onto device
    materialize_layer(layer, info)

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    for name, param in get_layer_tensors(layer).items():
        if (name in SKIP_TENSORS
                and getattr(_get_weight_loader(param),
                            "_vllm_receipt_observer", False)):
            continue
        param.weight_loader = _get_original_loader(param)

    # Load all buffered weights into materialized layer (using original loaders)
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        param.weight_loader(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    if info.kernel_tensors is not None:
        _copy_and_restore_kernel_tensors(layer, info)

    # The checkpoint iterator may still contain checkpoint-backed SKIP_TENSORS
    # (for example DeepSeek's gate.e_score_correction_bias). Keep all receipts
    # until finalize_layerwise_processing can reconcile the complete stream.
    info.reset(preserve_received_keys=True)
    logger.debug("%s: Processed", layer.__class__.__name__)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return the weight loader with any layerwise wrappers removed"""
    loader = _get_weight_loader(tensor)
    while loader.__name__ in ("online_process_loader",
                              "reload_receipt_observer"):
        loader = loader.__wrapped__  # type: ignore[union-attr]

    return loader


def _unwrap_receipt_observers(layer: torch.nn.Module) -> None:
    for tensor in get_layer_tensors(layer).values():
        loader = _get_weight_loader(tensor)
        if getattr(loader, "_vllm_receipt_observer", False):
            tensor.weight_loader = _get_original_loader(tensor)


def _get_weight_loader(tensor: torch.Tensor):
    return getattr(tensor, "weight_loader", default_weight_loader)


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Copy processed values into original kernel tensor storage and restore
    kernel tensor references on the layer. Preserves cudagraph references."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        buffer.data.copy_(getattr(layer, name))

    _place_kernel_tensors(layer, info)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer)
