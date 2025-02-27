# SPDX-License-Identifier: Apache-2.0

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
import re
from typing import Iterable, Literal, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA, RowParallelLinearWithShardedLoRA)
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsQuant
from .utils import maybe_prefix

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
        attention_instances: Optional[list[Attention]] = None,
        **kwargs):
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
        quant_config=None) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.
    
    `quant_config` is not yet supported.
    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(
            f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
    }.get(style, ReplicatedLinear)

    lora_linear_cls = {
        ColumnParallelLinear: {
            True: ColumnParallelLinearWithShardedLoRA,  # fully sharded
            False: ColumnParallelLinearWithLoRA  # not fully sharded
        },
        RowParallelLinear: {
            True: RowParallelLinearWithShardedLoRA,
            False: RowParallelLinearWithLoRA
        },
        # ReplicatedLinear doesn't support fully sharded LoRA yet,
        # so we use the same class for both cases.
        ReplicatedLinear: {
            True: ReplicatedLinearWithLoRA,
            False: ReplicatedLinearWithLoRA
        }
    }

    class HFCompatibleLinear(vllm_linear_cls):
        """
        Wrapper class that removes `output_bias` from returned output.
        """

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)[0]

        @classmethod
        def get_lora_class(cls, fully_sharded: bool = False):
            """
            Get the LoRA class corresponding to the current transformer
            linear class.

            Args:
                fully_sharded (bool): If True, select the LoRA class variant
                that supports fully sharded LoRA. Defaults to False.

            """
            return lora_linear_cls[vllm_linear_cls][fully_sharded]

    return HFCompatibleLinear(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
    )


class TransformersModel(nn.Module, SupportsQuant):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"
                         ]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        self.model: PreTrainedModel = AutoModel.from_config(
            self.config,
            attn_implementation="vllm",
            torch_dtype=vllm_config.model_config.dtype,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        prefix = self.model.base_model_prefix

        # MLP modifications
        self.apply_base_model_tp_plan(self.model)

        # Attention modifications (assumes 1 attention op per hidden layer)
        tp_size = get_tensor_model_parallel_world_size()
        self.attention_instances = [
            Attention(
                num_heads=divide(config.num_attention_heads, tp_size),
                head_size=config.head_dim,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=config.head_dim**-0.5,
                num_kv_heads=divide(config.num_key_value_heads, tp_size),
                cache_config=cache_config,
                quant_config=self.quant_config,
                prefix=f"{i}.attn") for i in range(config.num_hidden_layers)
        ]

        # Model modifications
        self.replace_vocab_embed_class(self.model)

        # ForCausalLM modifications
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=self.quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = get_sampler()

    def apply_base_model_tp_plan(self, module: nn.Module, prefix: str = ""):
        """
        Apply the base model tensor parallelization plan to a module.
        Currently only supports linear layers.
        """
        if (self.config.base_model_tp_plan is None
                and get_tensor_model_parallel_world_size() > 1):
            raise ValueError(
                "Trying to run tensor parallelization but the model does not "
                "support it yet!")

        for child_name, child_module in module.named_children():
            qual_name = maybe_prefix(prefix, child_name)
            for pattern, style in self.config.base_model_tp_plan.items():
                if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear):
                    new_module = replace_linear_class(child_module, style,
                                                      self.quant_config)
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)
            else:
                self.apply_base_model_tp_plan(child_module, prefix=qual_name)

    def replace_vocab_embed_class(self, module: nn.Module):
        # Use native set input embeddings
        new_module = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            quant_config=None,
        )
        log_replacement("input embedding", self.model.get_input_embeddings(),
                        new_module)
        self.model.set_input_embeddings(new_module)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids[None, ...],
            use_cache=False,
            position_ids=positions[None, ...],
            intermediate_tensors=intermediate_tensors,
            attention_instances=self.attention_instances,
            return_dict=False)[0][0, ...]  # we remove batch dimension for now
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set[str]()
        for name, loaded_weight in weights:
            if name not in params_dict:
                name = f"{self.model.base_model_prefix}.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params
