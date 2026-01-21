# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.modules.module import register_module_module_registration_hook
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.reload import (
    support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import supports_any_eagle
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_accelerator_view_from_cpu_tensor,
)

logger = init_logger(__name__)

WeightsMapping = Mapping[str, str | None]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """
    Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting the
    environment variable `VLLM_LOGGING_LEVEL=DEBUG`.
    """

    # Models trained using early version ColossalAI or quantized by
    # GPTQModel may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(
        self, module: nn.Module, child_params: dict[str, torch.Tensor]
    ):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                desc_param_keys = {
                    base_prefix + k for k, _ in module.named_parameters(recurse=True)
                }
                msg = (
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module._get_name()}. "
                    f"The available parameters belonging to {base_prefix} "
                    f"({module._get_name()}) are: {desc_param_keys}"
                )
                raise ValueError(msg)

    @support_quantized_model_reload_from_hp_weights
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: PretrainedConfig | None = None,
    architectures: list[str] | None = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """
    from vllm.model_executor.model_loader.utils import initialize_model

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = vllm_config.model_config.hf_config

    if hf_config is not None:
        vllm_config = vllm_config.with_hf_config(hf_config, architectures=architectures)

    return initialize_model(vllm_config=vllm_config, prefix=prefix)


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor: ...


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: Literal[True],
) -> torch.Tensor: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor: ...


def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor:
    """
    Flatten the `B` and `N` dimensions of batched multimodal inputs.

    The input tensor should have shape `(B, N, ...)`.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    ranges: list[list[int]] = [[] for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def isin_list(
    elements: torch.Tensor,
    test_elements_list: list[int],
) -> torch.Tensor:
    test_elements = torch.tensor(
        test_elements_list,
        pin_memory=is_pin_memory_available(),
    ).to(device=elements.device, non_blocking=True)

    return torch.isin(elements, test_elements)


class StageMissingLayer(nn.Module):
    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None:
        super().__init__()

        self.stage_name = stage_name

        # Don't register this as a child module in order to
        # avoid missing keys when loading weights
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        return getattr(self.__dict__["module"], name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self} should not be called")

    def extra_repr(self) -> str:
        return f"stage_name={self.stage_name!r}"


@contextmanager
def collect_children(
    module: nn.Module,
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    Within this context, collect all direct child assignments to `module`,
    returning a list of children names that is internally updated until the
    context is exited.

    If `targets` is set, instead collect descendents of `module`
    that are an instance of `targets`, even if they aren't direct children.
    """
    children_names = list[str]()

    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                children_names.append(name)

        with register_module_module_registration_hook(hook):
            yield children_names
    else:
        yield children_names

        for name, module_ in module.named_modules():
            if isinstance(module_, targets):
                children_names.append(name)


@contextmanager
def no_init_weights(
    module: nn.Module,
    placeholder: Callable[[nn.Module], nn.Module],
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    Within this context, prevent weight initialization from using device memory and
    replace direct child assignments to `module` with the result of `placeholder()`.

    If `targets` is set, instead prevent weight initialization and
    replace assignments where the child is an instance of `targets`,
    even if they aren't direct children of `module`.
    """
    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                return placeholder(submodule)

            return submodule

        with register_module_module_registration_hook(hook), torch.device("meta"):
            yield
    else:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if isinstance(module_, targets):
                submodule.to("meta")  # Free memory
            if isinstance(submodule, targets):
                submodule.to("meta")  # Free memory
                return placeholder(submodule)

            return submodule

        # Not all descendents are targeted, so we can't use a blanket
        # `torch.device("meta")` context
        with register_module_module_registration_hook(hook):
            yield


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes


def maybe_offload_to_cpu(module: torch.nn.Module) -> torch.nn.Module:
    if (params := next(module.parameters(), None)) is None:
        return module

    device = params.device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES
    if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()
    uva_available = is_uva_available()

    assert uva_available, "V1 CPU offloading requires uva (pin memory) support"
    uva_offloading = False

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    for p in module.parameters():
        if _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break

        # `torch.empty_like` does not support `pin_memory` argument
        cpu_data = torch.empty_strided(
            size=p.data.size(),
            stride=p.data.stride(),
            dtype=p.data.dtype,
            layout=p.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_data.copy_(p.data)
        if not uva_offloading:
            p.data = cpu_data
        else:
            # keep the cpu data alive
            p._vllm_offloaded_cpu_data = cpu_data
            p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
        _CPU_OFFLOAD_BYTES += p.data.numel() * p.data.element_size()
        offloaded_parameters = True

    if offloaded_parameters and not uva_offloading:
        original_forward = module.forward

        def forward(*args, **kwargs):
            module.forward = original_forward
            device_state = {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }

            # set `tie_weights=False` as tied weights in original model
            # become untied when calling .to(device)
            output = functional_call(module, device_state, args=args, kwargs=kwargs, tie_weights=False)
            module.forward = forward
            return output

        module.forward = forward

    return module


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + ".")
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, (StageMissingLayer, PPMissingLayer)):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def get_draft_quant_config(
    vllm_config: VllmConfig,
) -> QuantizationConfig | None:
    """Get quantization config for Draft models.

    Draft models should use their own quantization config instead of the verifier/target
    model's config. This helper retrieves the draft model's quantization config.

    Args:
        vllm_config: The vLLM configuration object.

    Returns:
        The draft model's config if available, None otherwise.
    """
    draft_model_config = vllm_config.speculative_config.draft_model_config
    draft_load_config = vllm_config.load_config

    return (
        VllmConfig.get_quantization_config(draft_model_config, draft_load_config)
        if draft_model_config
        else None
    )


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    else:
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(
    values: torch.Tensor, topk: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized topk implementation that uses torch.max for k=1 case.

    This function provides better performance for the common case of k=1
    by using torch.max instead of the more general torch.topk.

    Args:
        values: Input tensor to find top-k values from
        topk: Number of top values to return (k). Must be > 0.
        dim: Dimension along which to compute topk

    Returns:
        Tuple of (values, indices) where values are the top-k values
        and indices are their corresponding indices in the input tensor
    """
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.sequence_parallel_chunk_impl(x)


def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)


def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    seq_len = cdiv(x.size(0), tp_size)
    shape = list(x.shape)
    shape[0] = seq_len
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op(
    op_name="sequence_parallel_chunk_impl",
    op_func=sequence_parallel_chunk_impl,
    fake_impl=sequence_parallel_chunk_impl_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def process_eagle_weight(
    model: nn.Module,
    name: str,
) -> None:
    """
    Update EAGLE model flags based on loaded weight name.
    This should be called during weight loading to detect if a model
    has its own lm_head or embed_tokens weight.
    Args:
        model: The model instance (must support EAGLE)
        name: The name of the weight to process
    """
    if not supports_any_eagle(model):
        return

    # To prevent overriding with target model's layers
    if "lm_head" in name:
        model.has_own_lm_head = True
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True


def get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
       needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index
