# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

logger = init_logger(__name__)


class DummyInputLayerNorm(nn.Module):

    def __init__(self, weight=None, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight) if weight is not None else None
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        return x


class DummyOutputNorm(nn.Module):

    def forward(self, x, residual):
        if residual is None:
            return x
        else:
            return x + residual, None


class EAGLE(nn.Module):
    """This class implements the EAGLE draft model from the paper: https://arxiv.org/pdf/2401.15077
    Reference implementation: https://github.com/SafeAILab/EAGLE
    
    Differences from reference implementation:
    1. In reference, LlamaDecoderLayer implementation doesn't have 
       input_layernorm for 1st decoder layer (https://github.com/SafeAILab/EAGLE/blob/7d065d084443fbfd386f88839efd7193c12be869/eagle/model/cnets.py#L427).
       Following this approach, our implementation also disables
       the input_layernorm for the first decoder layer.
    2. We allow any decoder layer to be used in EAGLE whereas in reference 
       decoder layer is fixed to be LlamaDecoderLayer.
    3. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute.
    4. We allow an enhanced EAGLE architecture similar to the DeepSeek MTP 
       module with regards to the use of additional RMS norms. The original 
       EAGLE architecture 1) skips the pre-attention norm in its first 
       transformer block, and 2) skips the final output norm, both of which we 
       found to be suboptimal. We also add the support for separate norms
       applying to both the token embedding and hidden states before projection
       as in DeepSeek MTP, which we found to improve performance as well.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        architectures = getattr(self.config.model, "architectures", [])
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)

        self.model = model_cls(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))

        self.fc = nn.Linear(config.model.hidden_size * 2,
                            config.model.hidden_size,
                            bias=getattr(self.config, "eagle_fc_bias", False))

        # Modify layer normalization and residual connections as suggested
        # in the EAGLE framework: https://github.com/SafeAILab/EAGLE
        # While weights and biases are generally not needed,
        # they are retained here to support certain unit tests
        # (e.g., spec_decode/e2e/test_eagle_correctness.py).
        if not hasattr(self.config.model,
                       "skip_prenorm") or self.config.model.skip_prenorm:
            self.model.model.layers[0].input_layernorm = DummyInputLayerNorm(
                weight=self.model.model.layers[0].input_layernorm.weight)

        if not hasattr(
                self.config.model,
                "skip_output_norm") or self.config.model.skip_output_norm:
            self.model.model.norm = DummyOutputNorm()

        self.add_para_norm = False
        if hasattr(self.config.model,
                   "add_para_norm") and self.config.model.add_para_norm:
            self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.add_para_norm = True

        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=self.truncated_vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the EAGLE
        # checkpoint file has token_map tensor.
        self.token_map = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # Handle both empty previous_hidden_states
        # and mismatched batch size
        batch_size = inputs_embeds.size(0)
        if previous_hidden_states.size(0) == 0 or \
           previous_hidden_states.size(0) != batch_size:
            hidden_dim = self.config.model.hidden_size
            device = inputs_embeds.device
            # Create zero tensor with matching batch size
            previous_hidden_states = \
                torch.zeros(batch_size, hidden_dim, device=device)

        if self.add_para_norm:
            inputs_embeds = torch.cat([
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_states)
            ],
                                      dim=-1)
        else:
            inputs_embeds = torch.cat([inputs_embeds, previous_hidden_states],
                                      dim=-1)

        inputs_embeds = self.fc(inputs_embeds)

        inputs_embeds[positions == 0] = 0  # masking inputs at position=0

        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        if self.token_map is not None:
            _logits = logits
            logits = -torch.inf * torch.ones(
                size=(*_logits.shape[:-1], self.orig_vocab_size),
                device=_logits.device,
                dtype=_logits.dtype)

            logits[..., self.token_map] = _logits

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # This implementation is incompitable with https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
        # due to missing lm_head weights and its config being that of a
        # Llama model. Here's a compatible version with the same weights:
        # https://huggingface.co/abhigoyal/EAGLE-LLaMA3-Instruct-8B-vllm
        # Also, here's an example script for converting trained EAGLE
        # checkpoint to vLLM compatible version: https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d
        model_weights = {}
        for name, loaded_weight in weights:
            if name == "token_map":
                if self.config.truncated_vocab_size < self.config.vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name.startswith("fc.weight"):
                weight_loader = getattr(self.fc.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.fc.weight, loaded_weight)
            elif name.startswith("fc.bias"):
                if self.fc.bias is not None:
                    weight_loader = getattr(self.fc.bias, "weight_loader",
                                            default_weight_loader)
                    weight_loader(self.fc.bias, loaded_weight)
                else:
                    logger.warning_once("Found bias in the loaded weights but "
                                        "the model config doesn't have bias.")
            elif name.startswith("enorm.weight"):
                weight_loader = getattr(self.enorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.enorm.weight, loaded_weight)
            elif name.startswith("hnorm.weight"):
                weight_loader = getattr(self.hnorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.hnorm.weight, loaded_weight)
            elif name.startswith("model.lm_head.") or name.startswith(
                    "model.model."):
                model_weights[name.split("model.", 1)[-1]] = loaded_weight
            elif name.startswith("lm_head.") or name.startswith("model."):
                model_weights[name] = loaded_weight
            else:
                model_weights[f"model.{name}"] = loaded_weight

        if "lm_head.weight" in model_weights:
            lm_head_weight = model_weights.pop("lm_head.weight")

            if self.token_map is not None and\
                lm_head_weight.shape[0] > self.token_map.shape[0]:

                lm_head_weight = lm_head_weight[self.token_map]

        else:
            # NOTE(Shangming): initialize the placeholder for lm_head weight.
            lm_head_weight = torch.zeros(
                self.lm_head.org_vocab_size,
                self.lm_head.embedding_dim,
                dtype=self.config.torch_dtype,
            )

        weight_loader = getattr(self.lm_head.weight, "weight_loader",
                                default_weight_loader)
        weight_loader(self.lm_head.weight, lm_head_weight)

        self.model.load_weights(model_weights.items())
