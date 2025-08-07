# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper around `transformers` models"""
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Union, cast

import regex as re
import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    NestedTensors,
    PlaceholderRange,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors, PromptInputs
from vllm.utils import is_list_of

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP, SupportsQuant
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    default_weight_loader,
    flatten_bn,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)


def vllm_flash_attention_forward(
    # Transformers args
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    # Transformers kwargs
    scaling: Optional[float] = None,
    # vLLM kwargs
    attention_instances: Optional[dict[Attention]] = None,
    **kwargs,
):
    self_attn = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.impl.scale = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(query, key, value), None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def replace_linear_class(
    linear: nn.Linear,
    style: Literal["colwise", "rowwise"],
    quant_config: QuantizationConfig,
) -> Union[ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
    }.get(style, ReplicatedLinear)

    return vllm_linear_cls(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        return_bias=False,
    )


# Copied from `accelerate`
@contextmanager
def init_on_device_without_buffers(device: torch.device):
    """
    A context manager under which models are initialized with all
    parameters on the specified device. However buffers are not
    initialized on specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
    """

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):

        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        for torch_function_name in tensor_constructors_to_patch:
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


class MultiModalProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_max_image_tokens()}

    def get_max_image_tokens(self) -> int:
        width, height = self.get_max_image_size()
        processor = self.get_hf_processor()
        mm_processor_kwargs = self.ctx.model_config.mm_processor_kwargs or {}
        mm_tokens = processor._get_num_multimodal_tokens(
            image_sizes=([height, width],), **mm_processor_kwargs
        )
        image_tokens = mm_tokens["num_image_tokens"][0]
        return image_tokens

    def get_max_image_size(self):
        return 10_000, 10_000  # hardcode for arbitrary very large size


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        if "gemma3" in processor.__class__.__name__.lower():
            image_token = processor.boi_token
        else:
            image_token = getattr(processor, "image_token", "")
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_max_image_size()

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            ),
        }


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ):
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct  :class:`~vllm-multimodal.input.PlaceholderRange`
        for each multi-modal item.
        """
        return None

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
        num_image_patches: torch.Tensor = None,
    ):
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)
        mm_fields = {
            key: MultiModalFieldConfig.flat_from_sizes("image", num_image_patches)
            for key in hf_inputs
        }
        mm_fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
            "image", num_image_patches
        )
        mm_fields["num_image_patches"] = MultiModalFieldConfig.batched("image")
        return mm_fields

    def _apply_hf_processor_text_mm(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ):
        """
        Apply the HF processor on the prompt text and multi-modal data
        together.

        In addition, return whether prompt replacements have been applied.
        """
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)
        processor_data["return_mm_token_type_ids"] = True

        processed_data = self._call_hf_processor(
            prompt=prompt_text,
            mm_data=processor_data,
            mm_kwargs=hf_processor_mm_kwargs,
            tok_kwargs=tokenization_kwargs,
        )
        processed_data.update(passthrough_data)

        (prompt_ids,) = processed_data.pop("input_ids").tolist()
        mm_token_type_ids = (
            processed_data.pop("mm_token_type_ids")
            if "mm_token_type_ids" in processed_data
            else processed_data.pop("token_type_ids")
        )  # for gemma3 only

        return prompt_ids, processed_data, mm_token_type_ids

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        mm_items = self._to_mm_items(mm_data)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        if not isinstance(prompt, str):
            # the prompt is the tokenized ids which is not supported
            # by the hf_processor, which is why we would need to decode the ids
            # into string
            prompt = hf_processor.decode(prompt)

        (prompt_ids, processed_data, mm_token_type_ids) = (
            self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )
        )

        # HF processor will return `mm_token_type_ids` from which
        # we can infer mm_placeholders. Until then hardcode to make code run
        # Below tested on Llava. Prompts and `mm_token_type_ids` are always bs=1
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        images = mm_items.get_items("image", ImageProcessorItems)
        mm_processor_kwargs = self.info.ctx.model_config.mm_processor_kwargs or {}
        image_sizes = []
        for item_idx in range(len(images)):
            image_size = images.get_image_size(item_idx)
            image_sizes.append((image_size.height, image_size.width))

        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes, **mm_processor_kwargs
        )

        mm_placeholders = {}
        split_sizes = mm_tokens_per_modality["num_image_tokens"]
        if split_sizes:
            chunked_mm_positions = torch.split(mm_positions, split_sizes)
            mm_tokens = torch.tensor(prompt_ids)[mm_token_type_ids[0].bool()]
            chunked_mm_tokens = torch.split(mm_tokens, split_sizes)
            ranges = [
                PlaceholderRange(
                    offset=positions[0].item(),
                    length=positions.shape[0],
                    is_embed=(mm_tokens == hf_processor.image_token_id).bool(),
                )
                for positions, mm_tokens in zip(chunked_mm_positions, chunked_mm_tokens)
            ]
            mm_placeholders = {"image": ranges}

        num_image_patches = (
            torch.tensor(mm_tokens_per_modality["num_image_patches"])
            if "num_image_patches" in mm_tokens_per_modality
            else None
        )
        processed_data["num_image_patches"] = num_image_patches
        mm_kwargs = MultiModalKwargs.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(
                processed_data, hf_processor_mm_kwargs, num_image_patches
            ),
        )

        mm_hashes = self._hash_mm_items(
            mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        )
        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class ConfigOverride:
    """Context manager to temporarily override config attributes."""

    def __init__(self, config: PretrainedConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.kwargs_original = {}
        self.kwargs_delete = set()

    def __enter__(self):
        """Override config attributes."""
        for key, value in self.kwargs.items():
            if not hasattr(self.config, key):
                self.kwargs_delete.add(key)
            self.kwargs_original[key] = getattr(self.config, key, None)
            setattr(self.config, key, value)
        return self.config

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore original config attributes."""
        for key, value in self.kwargs_original.items():
            if key in self.kwargs_delete:
                delattr(self.config, key)
            else:
                setattr(self.config, key, value)


class TransformersModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        logger.info("Using Transformers backend.")

        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config: CacheConfig = vllm_config.cache_config
        device_config: DeviceConfig = vllm_config.device_config
        model_config: ModelConfig = vllm_config.model_config
        parallel_config: ParallelConfig = vllm_config.parallel_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config
        self.cache_config = cache_config
        self.device_config = device_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.quant_config = quant_config

        self.pp_group = get_pp_group()
        self.pp_size = self.pp_group.world_size
        self.pp_rank = self.pp_group.rank_in_group
        self.tp_size = get_tensor_model_parallel_world_size()

        # vLLM handles interleaved sliding window attention by creating a new
        # interleaved_sliding_window attribute and deleting the sliding_window
        # attribute. This breaks the constructors in Transformers so we
        # temporarily add the attribute back to construct the model.
        config_override = nullcontext()
        if hasattr(config, "interleaved_sliding_window"):
            config_override = ConfigOverride(
                config, sliding_window=config.interleaved_sliding_window
            )

        # Use meta device to delay allocating GPU tensors
        with torch.device("meta"), config_override:
            # FIXME(Isotr0py): We need to refactor this part in the future to
            # avoid registering an extra model layer, otherwise we will need a
            # weights mapper to rename weights.
            self.model: PreTrainedModel = AutoModel.from_config(
                config,
                attn_implementation="vllm",
                torch_dtype=model_config.dtype,
                trust_remote_code=model_config.trust_remote_code,
            )

        self.pipeline_parallel()
        self.tensor_parallel()

        # Input embeddings
        if not isinstance(self.model.get_input_embeddings(), PPMissingLayer):
            self.model.set_input_embeddings(
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    quant_config=quant_config,
                )
            )

        # Attention layers
        self.attention_instances = self.create_attention_instances()

        # Initialize buffers (e.g. rotary embedding inverse frequency)
        self.init_buffers(self.model)

        # Initialize any parameters that have not had their modules replaced
        self.init_parameters(self.model)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def pipeline_parallel(self):
        """
        Apply the model's pipeline parallelization plan.
        """
        if self.pp_size <= 1:
            return

        if not self.model.supports_pp_plan:
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel yet!"
            )

        module_lists = []
        module_list_idx = None
        pp_plan = list(self.model._pp_plan.keys())
        for i, name in enumerate(pp_plan):
            if isinstance(getattr(self.model, name), nn.ModuleList):
                module_lists.append(name)
                module_list_idx = i

        if len(module_lists) > 1:
            raise ValueError(
                "Pipeline parallel of models with multiple `ModuleList`s "
                "in the base model are not supported yet!"
            )
        if module_list_idx is None:
            raise ValueError(f"Could not find `ModuleList` in {type(self.model)}")

        # Layers before module list
        for name in pp_plan[:module_list_idx]:
            if self.pp_group.is_first_rank or (
                self.config.tie_word_embeddings and self.pp_group.is_last_rank
            ):
                continue
            setattr(self.model, name, PPMissingLayer())

        # Module list
        start_layer, end_layer = get_pp_indices(
            self.config.num_hidden_layers, self.pp_rank, self.pp_size
        )
        layers_name = pp_plan[module_list_idx]
        layers = getattr(self.model, layers_name)
        for i in range(len(layers)):
            if start_layer <= i and i < end_layer:
                continue
            layers[i] = PPMissingLayer(return_tuple=True)

        # Layers after module list
        for name in pp_plan[module_list_idx + 1 :]:
            # Modules that should be on last rank
            if not self.pp_group.is_last_rank:
                setattr(self.model, name, PPMissingLayer())

    def tensor_parallel(self):
        """
        Apply the model's tensor parallelization plan.
        Currently only supports linear layers.
        """
        if not self.model.supports_tp_plan:
            if self.tp_size <= 1:
                return

            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!"
            )

        tp_plan = self.model._tp_plan

        def _tensor_parallel(module: nn.Module, prefix: str = ""):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear
                    ):
                        new_module = replace_linear_class(
                            child_module, style, self.quant_config
                        )
                        setattr(module, child_name, new_module)
                        log_replacement(qual_name, child_module, new_module)
                else:
                    _tensor_parallel(child_module, prefix=qual_name)

        _tensor_parallel(self.model)

    def create_attention_instances(self) -> dict[int, Attention]:
        """
        Create `Attention` instances to inform KV cache allocation.
        """
        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        start, end = get_pp_indices(
            self.config.num_hidden_layers, self.pp_rank, self.pp_size
        )

        attention_instances = {}
        for i in range(start, end):
            # Handle interleaved sliding window attention
            sliding_window = None
            if (
                hasattr(self.config, "interleaved_sliding_window")
                and hasattr(self.config, "sliding_window_pattern")
                and ((i + 1) % self.config.sliding_window_pattern > 0)
            ):
                sliding_window = self.config.interleaved_sliding_window

            attention_instances[i] = Attention(
                num_heads=num_heads,
                head_size=head_size,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=head_size**-0.5,
                num_kv_heads=num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                per_layer_sliding_window=sliding_window,
                prefix=f"{i}.attn",
            )
        return attention_instances

    def init_buffers(self, module: nn.Module):
        """
        If a `buffer` is on the `meta` device, then its parent
        `module` is the original module created by:

        ```python
        with torch.device("meta"):
            self.model: PreTrainedModel = AutoModel.from_config(...)
        ```

        This means that:
        - `type(module)` is a class from `transformers`
        - This class is constructed using a `PretrainedConfig`
        """
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == torch.device("meta"):
                if module == self.model:
                    logger.warning(
                        "To initialize buffers correctly, we instantiate the "
                        "parent module and and extract the value of the "
                        "buffer from it. In this case, the parent module is "
                        "the base model. Instantiating the entire model here "
                        "risks GPU OOM. Could this buffer be moved to a child "
                        "module?"
                    )
                new_buffer = getattr(type(module)(self.config), name)
                setattr(module, name, new_buffer)
        for child in module.children():
            self.init_buffers(child)

    def init_parameters(self, module: nn.Module):
        """
        If a `parameter` is on the `meta` device, then its parent
        `module` is the original module created by:

        ```python
        with torch.device("meta"):
            self.model: PreTrainedModel = AutoModel.from_config(...)
        ```
        """
        for name, param in module.named_parameters(recurse=False):
            if param.device == torch.device("meta"):
                new_param = nn.Parameter(
                    torch.empty_like(param.data, device=self.device_config.device)
                )
                setattr(module, name, new_param)
        for child in module.children():
            self.init_parameters(child)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        if input_ids is not None:
            input_ids = input_ids[None, ...]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[None, ...]

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=positions[None, ...],
            attention_instances=self.attention_instances,
            return_dict=False,
        )[0][
            0, ...
        ]  # we remove batch dimension for now

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        loaded_params = set[str]()
        for name, loaded_weight in weights:
            # Use "model" instead of base_model_prefix because
            # the base model attribute in vLLM is always `model`
            if not name.startswith(prefix := "model."):
                name = prefix + name

            if is_pp_missing_parameter(name, self):
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


@support_torch_compile
class TransformersForCausalLM(nn.Module, SupportsQuant, SupportsLoRA, SupportsPP):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config

        self.model = TransformersModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.get_input_embeddings()
                )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    # FIXME(Isotr0py): Don't use any weights mapper for Transformers backend,
    # this makes thing complicated. We need to remove this mapper after refactor
    # `TransformersModel` in the future.
    @property
    def hf_to_vllm_mapper(self):
        prefix_mapper = {
            name: "model." + name for name, _ in self.model.model.named_children()
        }
        return WeightsMapper(
            orig_to_new_substr={"model.": "model.model."},
            orig_to_new_prefix=prefix_mapper,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile
class TransformersForMultimodalLM(
    nn.Module, SupportsMultiModal, SupportsQuant, SupportsLoRA, SupportsPP
):
    # Refer to `vllm/model_executor/models/llava.py`
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config

        with init_on_device_without_buffers(vllm_config.device_config.device):
            self.model = TransformersModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.get_input_embeddings()
                )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @property
    def hf_to_vllm_mapper(self):
        prefix_mapper = {
            name: "model." + name for name, _ in self.model.model.named_children()
        }
        return WeightsMapper(
            orig_to_new_substr={"model.": "model.model."},
            orig_to_new_prefix=prefix_mapper,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_supported_multimodal_processors(
        self,
    ) -> Mapping[str, type[BaseMultiModalProcessor]]:
        return {
            "image": MultiModalProcessor,
        }

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_embeds = kwargs.get("image_embeds", None)
        return {"image": image_embeds} if image_embeds is not None else None

    def get_mm_processing_info(self) -> MultiModalProcessingInfo:
        return MultiModalProcessingInfo(
            ctx=self.make_empty_intermediate_tensors_factory
        )

    def get_mm_dummy_inputs_builder(self) -> MultiModalDummyInputsBuilder:
        return MultiModalDummyInputsBuilder(info=self.get_mm_processing_info())

    def init_multimodal_modules(
        self,
    ) -> dict[str, type[BaseMultiModalProcessor]]:
        """
        Initialize processor class for each input type

        model_config = ModelConfig(
            model,
            tokenizer,
            tokenizer_mode,
            trust_remote_code,
            revision,
            code_revision,
            max_model_len,
            quantization,
            enforce_eager,
        )
        """

        return {"image": MultiModalProcessor(self.get_mm_processing_info())}

    def process_multimodal_inputs(
        self,
        inputs: MultiModalInputs,
        **kwargs,
    ) -> torch.Tensor:
        """
        Preprocess multimodal data and return the corresponding embeddings.
        """
        mm_kwargs: MultiModalKwargs = inputs.get("mm_kwargs", {})
        image_embeds = mm_kwargs.get("image_embeds")
        if image_embeds is None:
            # No multimodal data
            return None

        processor_data = {}
        try:
            processor_data = self.get_multimodal_embeddings(image_embeds=image_embeds)
        except Exception as e:
            raise ValueError(f"Failed to process multimodal inputs: {e}") from e

        if processor_data is None:
            # No multimodal data was found; this should not happen
            return None

        return processor_data

    def _process_multimodal_inputs_per_prompt(
        self,
        inputs: PromptInputs,
    ) -> Union[MultiModalInputs, PromptInputs]:
        """
        Apply multimodal processor to convert
        MM data + prompt to MM inputs.
        """
        # Inputs are `PromptInputs` if there are no images
        prompt_inputs = cast(PromptInputs, inputs)
        if not prompt_inputs.get("multi_modal_data"):
            return inputs

        # Otherwise, `inputs` are of type `ExplicitEncoderDecoderPrompt`
        mm_data = prompt_inputs["multi_modal_data"]
        mm_inputs = {
            mm_input_type: mm_processor.apply(
                prompt=prompt_inputs["prompt"],
                mm_data=mm_data,
                hf_processor_mm_kwargs={},  # TODO: Fill this from config
            )
            for mm_input_type, mm_processor in self.multimodal_processors.items()
        }

        return cast(MultiModalInputs, mm_inputs)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Prepare input embeddings with optional multimodal embeddings.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)
        multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)

        if multimodal_embeddings is not None:
            # TODO: Perform multimodal embedding fusion logic here
            # This is a placeholder and would need to be implemented
            # based on the specific multimodal fusion strategy
            pass

        return inputs_embeds
