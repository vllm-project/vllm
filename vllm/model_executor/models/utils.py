# SPDX-License-Identifier: Apache-2.0

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Protocol, Union, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MultiModalPlaceholderMap, NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils import (get_cuda_view_from_cpu_tensor, is_pin_memory_available,
                        is_uva_available)

logger = init_logger(__name__)

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
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
        return ((out_name, data) for name, data in weights
                if (out_name := self._map_name(name)) is not None)


class AutoWeightsLoader:
    """
    Helper class to load weights into a {class}`torch.nn.Module`. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.

    Detailed weight loading information can be viewed by setting the
    environment variable ``VLLM_LOGGING_LEVEL=DEBUG``.
    """

    # Models trained using early version ColossalAI
    # may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: Optional[list[str]] = None,
        skip_substrs: Optional[list[str]] = None,
        ignore_unexpected_prefixes: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts,
                                               key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (("" if len(parts) == 1 else parts[1], weights_data)
                 for parts, weights_data in group),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return (any(qualname.startswith(p) for p in self.skip_prefixes)
                or any(substr in qualname for substr in self.skip_substrs))

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes)

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
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'")

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname,
                         param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(self, module: nn.Module,
                                        child_params: dict[str, torch.Tensor]):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(module, (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
        )):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var",
                              "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters "
                        "for module %s", module)
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

                yield from self._load_module(prefix,
                                             child_modules[child_prefix],
                                             child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
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

                msg = (f"There is no module or parameter named '{prefix}' "
                       f"in {type(self.module).__name__}")
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = ((name, weight) for name, weight in weights
                   if not self._can_skip(name))

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: Optional[PretrainedConfig] = None,
    architectures: Optional[list[str]] = None,
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
        vllm_config = vllm_config.with_hf_config(hf_config,
                                                 architectures=architectures)

    return initialize_model(vllm_config=vllm_config, prefix=prefix)


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor:
    ...


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]:
    ...


@overload
def flatten_bn(
    x: Union[list[torch.Tensor], torch.Tensor],
    *,
    concat: Literal[True],
) -> torch.Tensor:
    ...


@overload
def flatten_bn(
    x: Union[list[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[list[torch.Tensor], torch.Tensor]:
    ...


def flatten_bn(
    x: Union[list[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[list[torch.Tensor], torch.Tensor]:
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, ...)```.
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

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def merge_multimodal_embeddings_from_map(
        inputs_embeds: torch.Tensor, multimodal_embeddings: NestedTensors,
        placeholder_map: MultiModalPlaceholderMap.IndexMap) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` using the provided 
    placeholder map .

    Note:
        This updates ``inputs_embeds`` in place.
    """
    flattened_embeddings = _flatten_embeddings(multimodal_embeddings)
    inputs_embeds[placeholder_map.dest] = flattened_embeddings[
        placeholder_map.src]
    return inputs_embeds


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds


def embed_multimodal(
    input_ids: torch.Tensor,
    multimodal_token_id: int,
    get_text_embeds: Callable[[torch.Tensor], torch.Tensor],
    multimodal_embeds: NestedTensors,
) -> torch.Tensor:
    """
    Embed token IDs and multimodal inputs and combine their embeddings.

    ``multimodal_token_id`` is used to determine whether a token ID should
    be embedded using ``get_text_embeds`` or ``get_multimodal_embeds``.

    Compared to ``merge_multimodal_embeddings`, this avoids running
    ``get_text_embeds`` on ``input_ids[input_ids == multimodal_token_id]``
    which causes issues when the placeholder token ID exceeds the
    vocabulary size of the language model.
    """
    is_multimodal = input_ids == multimodal_token_id
    is_text = ~is_multimodal

    text_embeds = get_text_embeds(input_ids[is_text])
    merged_embeds = torch.empty(
        (input_ids.shape[0], text_embeds.shape[1]),
        dtype=text_embeds.dtype,
        device=text_embeds.device,
    )

    merged_embeds[is_text] = text_embeds

    return _merge_multimodal_embeddings(
        merged_embeds,
        is_multimodal,
        multimodal_embeds,
    )


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: Union[int, list[int]],
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    
    ``placeholder_token_id`` can be a list of token ids (e.g, token ids 
    of img_start, img_break, and img_end tokens) when needed: This means 
    the order of these tokens in the ``input_ids`` MUST MATCH the order of 
    their embeddings in ``multimodal_embeddings`` since we need to 
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.
    
    Then the image embeddings (that correspond to I's) from vision encoder 
    must be padded with embeddings of S, B, and E in the same order of 
    input_ids for a correct embedding merge.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = torch.tensor(placeholder_token_id,
                                            device=input_ids.device)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            torch.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )


class LayerFn(Protocol):

    def __call__(self, prefix: str) -> torch.nn.Module:
        ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input, ) if self.return_tuple else input


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

    if envs.VLLM_USE_V1:
        assert uva_available, ("V1 CPU offloading requires"
                               " uva (pin memory) support")
        uva_offloading = True
    else:
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
        cpu_data = torch.empty_strided(size=p.data.size(),
                                       stride=p.data.stride(),
                                       dtype=p.data.dtype,
                                       layout=p.data.layout,
                                       device='cpu',
                                       pin_memory=pin_memory)
        cpu_data.copy_(p.data)
        if not uva_offloading:
            p.data = cpu_data
        else:
            # keep the cpu data alive
            p._vllm_offloaded_cpu_data = cpu_data
            p.data = get_cuda_view_from_cpu_tensor(cpu_data)
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
            output = functional_call(module,
                                     device_state,
                                     args=args,
                                     kwargs=kwargs)
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
    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
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
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + '.')
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model))


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):

    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors({
            key:
            torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
            for key in keys
        })

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


def extract_layer_index(layer_name: str) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (f"layer name {layer_name} should"
                                " only contain one integer")
    return int_vals[0]


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)
