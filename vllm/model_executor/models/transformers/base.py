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
"""Transformers modeling backend base class."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import regex as re
import torch
import transformers
from packaging.version import Version
from torch import nn
from transformers import AutoModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.config.utils import getattr_iter
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.interfaces_base import VllmModel
from vllm.model_executor.models.transformers.utils import (
    get_feature_request_tip,
    init_on_device_without_buffers,
    log_replacement,
    replace_conv_class,
    replace_linear_class,
    replace_rms_norm_class,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from vllm.config import VllmConfig
else:
    PreTrainedModel = object

logger = init_logger(__name__)


def vllm_flash_attention_forward(
    # Transformers args
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    # Transformers kwargs
    scaling: float | None = None,
    # vLLM kwargs
    attention_instances: dict[int, Attention] | None = None,
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


class Base(
    nn.Module,
    VllmModel,
    SupportsQuant,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
    embedding_modules = ["embed_tokens"]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__()
        logger.info("Using Transformers modeling backend.")

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

        # Attrs for weight loading (see self.load_weights)
        self.skip_prefixes: list[str] = []
        """Skip loading weights whose qualname starts with these prefixes."""
        self.skip_substrs: list[str] = []
        """Skip loading weights whose qualname contains these substrings."""
        self.ignore_unexpected_prefixes: list[str] = []
        """Ignore unexpected weights whose qualname starts with these prefixes."""
        self.ignore_unexpected_suffixes: list[str] = []
        """Ignore unexpected weights whose qualname ends with these suffixes."""

        # Attrs for Eagle3 (see self.set_aux_hidden_state_layers)
        self._target_class: type[nn.Module] = nn.Module
        """Target class for Eagle3 aux hidden state recording."""
        self._layer_names: dict[int, str] = {}
        """Mapping from layer index to layer name for Eagle3."""
        self._output_aux_hidden_states_kwargs: dict[str, bool] = {}
        """Kwargs to pass to model forward for Eagle3 aux hidden states."""

        if self.quant_config:
            quant_method_name = self.quant_config.get_name()
            # Check for unsupported quantization methods.
            if quant_method_name == "mxfp4":
                raise NotImplementedError(
                    "Transformers modeling backend does "
                    "not support MXFP4 quantization yet."
                )
            # Skip loading extra bias for GPTQ models.
            if "gptq" in quant_method_name:
                self.ignore_unexpected_suffixes.append(".bias")

        # Set correct attn and init on "meta" to delay allocating GPU tensors
        self.text_config._attn_implementation = "vllm"
        with init_on_device_without_buffers("meta"):
            self.model: PreTrainedModel = AutoModel.from_config(
                self.config,
                dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )

        # Create weight mapper
        self.hf_to_vllm_mapper = WeightsMapper()
        orig_to_new_regex = self.hf_to_vllm_mapper.orig_to_new_regex

        if Version(transformers.__version__) >= Version("5.0.0.dev0"):
            from transformers.conversion_mapping import (
                WeightRenaming,
                get_model_conversion_mapping,
            )

            for mapping in get_model_conversion_mapping(self.model):
                # Handle weights which have been renamed in Transformers
                if isinstance(mapping, WeightRenaming):
                    compiled_sources = mapping.compiled_sources
                    target_pattern = mapping.target_patterns[0]
                    orig_to_new_regex[compiled_sources] = target_pattern
                # TODO: Handle WeightConverter to enable layer merging

        # Standardise base model prefix
        bmp = self.model.base_model_prefix
        expected_bmp = r"model\.\1"
        # Handle checkpoints saved with different base model prefix
        if bmp and bmp != "model":
            different_bmp_pattern = re.compile(rf"^{bmp}\.(.+)")
            orig_to_new_regex[different_bmp_pattern] = expected_bmp
        # Handle checkpoints saved without base model prefix
        model_children = "|".join(name for name, _ in self.model.named_children())
        missing_bmp_pattern = re.compile(rf"^(?!model\.)(({model_children}).+)")
        orig_to_new_regex[missing_bmp_pattern] = expected_bmp

        # Remove\ layers not on this pipeline parallel rank
        self.pipeline_parallel()
        # Substitute remaining layers with vLLM's layers as needed
        self.recursive_replace()
        # Create attention instances for KV cache allocation
        self.attention_instances = self.create_attention_instances()

        # Input embeddings
        input_embeddings = self.model.get_input_embeddings()
        if not isinstance(input_embeddings, PPMissingLayer):
            # Some models scale embeddings inside the input embedding layer
            self.embed_scale = getattr(input_embeddings, "embed_scale", None)
            names = ("embedding_size", "hidden_size")
            embedding_dim = getattr_iter(self.text_config, names, None)
            assert embedding_dim is not None
            self.model.set_input_embeddings(
                VocabParallelEmbedding(
                    self.text_config.vocab_size,
                    embedding_dim=embedding_dim,
                    org_num_embeddings=self.text_config.vocab_size,
                    quant_config=self.quant_config,
                )
            )

        # Initialize any parameters that have not had their modules replaced
        self.init_parameters(self.model)

        # Pipeline parallel intermediate tensors
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.text_config.hidden_size
        )

    def pipeline_parallel(self):
        """
        Apply the model's pipeline parallelization plan.
        """
        if self.pp_group.world_size <= 1:
            return

        if not self.model.supports_pp_plan:
            tip = get_feature_request_tip(
                self.model_config.model, self.model_config.trust_remote_code
            )
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel. {tip}"
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
                getattr(self.text_config, "tie_word_embeddings", False)
                and self.pp_group.is_last_rank
            ):
                continue
            setattr(self.model, name, PPMissingLayer())

        # Module list
        start_layer, end_layer = get_pp_indices(
            self.text_config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        layers_name = pp_plan[module_list_idx]
        layers = getattr(self.model, layers_name)
        for i in range(len(layers)):
            if start_layer <= i and i < end_layer:
                continue
            layers[i] = PPMissingLayer()

        # Layers after module list
        for name in pp_plan[module_list_idx + 1 :]:
            # Modules that should be on last rank
            if not self.pp_group.is_last_rank:
                setattr(self.model, name, PPMissingLayer())

    def recursive_replace(self):
        """Recursively replace modules in the model as needed.

        Currently, this replaces:

        - `nn.Linear` with vLLM's tensor parallel linear classes
        - `*RMSNorm` with vLLM's `RMSNorm`
        """
        tp_plan = self.model.tp_plan

        if not tp_plan and self.tp_group.world_size > 1:
            tip = get_feature_request_tip(
                self.model_config.model, self.model_config.trust_remote_code
            )
            raise ValueError(
                f"{type(self.model)} does not support tensor parallel. {tip}"
            )

        # Prefix the patterns because we always start from `self.model`
        tp_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}

        def _recursive_replace(module: nn.Module, prefix: str):
            for child_name, child_module in module.named_children():
                new_module = child_module
                qual_name = maybe_prefix(prefix, child_name)
                # Populate Eagle3 attrs
                if (
                    isinstance(module, nn.ModuleList)
                    and len(module) == self.text_config.num_hidden_layers
                ):
                    self._target_class = type(child_module)
                    layer_name = qual_name.removeprefix("model.")
                    self._layer_names[int(child_name)] = layer_name
                # Replace modules as needed
                if isinstance(child_module, nn.Linear):
                    generator = (p for p in tp_plan if re.match(p, qual_name))
                    pattern = next(generator, None)
                    # Some weight loaders expect all linear layers to inherit
                    # LinearBase, so we set a default style which causes any
                    # unspecified layers to be replaced with ReplicatedLinear
                    style = tp_plan.get(pattern, "replicate")
                    new_module = replace_linear_class(
                        child_module, style, self.quant_config, prefix=qual_name
                    )
                elif isinstance(child_module, (nn.Conv2d, nn.Conv3d)):
                    new_module = replace_conv_class(child_module)
                elif child_module.__class__.__name__.endswith("RMSNorm"):
                    new_module = replace_rms_norm_class(
                        child_module, self.text_config.hidden_size
                    )
                else:
                    _recursive_replace(child_module, prefix=qual_name)

                if new_module is not child_module:
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)

        _recursive_replace(self.model, prefix="model")

    def create_attention_instances(self) -> dict[int, Attention]:
        """
        Create `Attention` instances to inform KV cache allocation.
        """
        text_config = self.text_config

        num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        logits_soft_cap = getattr(text_config, "attn_logit_softcapping", None)

        # In encoder models, the attention layers will have `is_causal=False`
        is_encoder = lambda module: not getattr(module, "is_causal", True)
        has_encoder = lambda model: any(is_encoder(m) for m in model.modules())
        is_multimodal = lambda config: config != config.get_text_config()
        # vLLM does not support encoder-decoder models, so if any encoder layer is
        # found in a text only model, we assume the whole model is an encoder model
        if has_encoder(self.model) and not is_multimodal(self.config):
            self.check_version("5.0.0", "encoder models support")
            attn_type = AttentionType.ENCODER_ONLY
        else:
            attn_type = AttentionType.DECODER

        pp_rank = self.pp_group.rank_in_group
        pp_size = self.pp_group.world_size
        start, end = get_pp_indices(text_config.num_hidden_layers, pp_rank, pp_size)

        attention_instances = {}
        for i in range(start, end):
            # Handle interleaved sliding window attention
            per_layer_sliding_window = None
            if (
                hasattr(self.config, "layer_types")
                and self.config.layer_types[i] == "sliding_attention"
            ):
                per_layer_sliding_window = self.config.sliding_window

            attn_cls = (
                EncoderOnlyAttention
                if attn_type == AttentionType.ENCODER_ONLY
                else Attention
            )
            attention_instances[i] = attn_cls(
                num_heads=num_heads,
                head_size=head_size,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=head_size**-0.5,
                num_kv_heads=num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                logits_soft_cap=logits_soft_cap,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{i}.attn",
                attn_type=attn_type,
            )
        return attention_instances

    def init_parameters(self, module: nn.Module, dtype: torch.dtype | None = None):
        """
        If a `parameter` is on the `meta` device, then its parent
        `module` is the original module created by:

        ```python
        with torch.device("meta"):
            self.model: "PreTrainedModel" = AutoModel.from_config(...)
        ```
        """

        def _init_parameters(module: nn.Module, dtype: torch.dtype | None):
            for name, param in module.named_parameters(recurse=False):
                if param.device == torch.device("meta"):
                    new_param = nn.Parameter(
                        torch.empty_like(
                            param.data,
                            dtype=dtype or self.model_config.dtype,
                            device=self.device_config.device,
                        )
                    )
                    setattr(module, name, new_param)
            for child in module.children():
                _init_parameters(child, dtype)

        _init_parameters(module, dtype)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if self.embed_scale is not None:
            inputs_embeds *= self.embed_scale
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        if input_ids is not None:
            input_ids = input_ids[None, ...]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[None, ...]

        # If the model scales embeddings inside the input embedding layer we must
        # ensure they are scaled here since VocabParallelEmbedding will not do it
        if (
            self.embed_scale is not None
            and input_ids is not None
            and inputs_embeds is None
        ):
            inputs_embeds = self.embed_input_ids(input_ids)
            input_ids = None

        if self.model_config.uses_mrope:
            position_ids = positions[:, None]
        else:
            position_ids = positions[None, ...]

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=position_ids,
            attention_instances=self.attention_instances,
            return_dict=False,
            **self._output_aux_hidden_states_kwargs,
            **kwargs,
        )
        # We must remove the batch dimension from these outputs
        hidden_states = outputs[0][0, ...]
        if self._output_aux_hidden_states_kwargs:
            aux_hidden_states = [x[0][0, ...] for x in outputs[1:]]

        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self._output_aux_hidden_states_kwargs and len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @staticmethod
    def check_version(min_version: str, feature: str):
        installed = Version(transformers.__version__)
        required = Version(min_version)
        if installed < required:
            raise ImportError(
                f"Transformers modeling backend requires transformers>={required} "
                f"for {feature}, but got {installed}"
            )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.check_version("5.0.0", "Eagle3 support")
        from transformers.utils.generic import OutputRecorder

        # The default value in PreTrainedModel is None
        if self.model._can_record_outputs is None:
            self.model._can_record_outputs = {}

        target_class = self._target_class
        for layer in layers:
            # layer - 1 because we want the input to the layer
            layer_name = self._layer_names[layer - 1]
            layer_key = f"aux_hidden_state_{layer}"
            aux_hidden_state_i = OutputRecorder(target_class, layer_name=layer_name)
            self.model._can_record_outputs[layer_key] = aux_hidden_state_i
            self._output_aux_hidden_states_kwargs[f"output_{layer_key}"] = True

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = self.text_config.num_hidden_layers
        return (2, num_layers // 2, num_layers - 3)
