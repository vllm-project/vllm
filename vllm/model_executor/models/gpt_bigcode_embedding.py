# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/gpt2/modeling_gpt2.py
# Copyright 2023 The vLLM team.
# Copyright 2023 CTranslate2, and Michael Feil
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Inference-only GPTBigCode model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import GPTBigCodeConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import is_pp_missing_parameter
from vllm.model_executor.models.gpt_bigcode import (
    GPTBigCodeModel)
from typing import Iterable, List, Optional, Tuple
import torch.nn.functional as F
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolingSequenceGroupOutput, PoolerOutput
from typing import Iterable, List, Optional, Set, Tuple, Union


class TokenClassifierMixin:
    """
    A class providing functionality for token classification, including
    initializing a classification head, setting candidate token positions, and pooling.
    """

    def init_pooler(self, hidden_size: int, end_token_only: bool, n_classes: int,
                    candidate_start_ndx: int, candidate_end_ndx: int):
        """
        Initializes the classifier head used for token classification.

        Args:
            hidden_size (int): The size of the hidden representations.
            end_token_only (bool): If True, only end tokens are used for classification.
            n_classes (int): The number of classes for classification.
            candidate_start_ndx (int): The token ID for candidate start tokens.
            candidate_end_ndx (int): The token ID for candidate end tokens.
        """
        if end_token_only:
            self.start_end_classifier = nn.Linear(hidden_size, n_classes)
        else:
            self.start_end_classifier = nn.Linear(hidden_size * 2, n_classes)
        self.end_token_only = end_token_only
        self.candidate_start_ndx, self.candidate_end_ndx = candidate_start_ndx, candidate_end_ndx
        self.cand_starts, self.cand_ends = None, None

    def set_candidate_positions(self, input_ids: torch.Tensor):
        """
        Sets the positions of candidate start and end tokens.

        Args:
            input_ids (torch.Tensor): Input token IDs.
        """
        self.cand_starts = (input_ids == self.candidate_start_ndx)
        self.cand_ends = (input_ids == self.candidate_end_ndx)

    def pooler(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata) -> Optional[PoolerOutput]:
        """
        Pools hidden states based on candidate token positions.

        Args:
            hidden_states (torch.Tensor): The hidden states output from the model.
            pooling_metadata (PoolingMetadata): Metadata containing prompt lengths.

        Returns:
            Optional[PoolerOutput]: The output containing pooled representations.
        """
        # Extract candidate token positions and apply pooling logic
        # extract candidate
        all_logits = []
        if self.cand_starts.any() and self.cand_ends.any():
            # get length per prompt
            all_token_lengths = pooling_metadata.prompt_lens

            max_length = max(all_token_lengths)
            # need to pad hidden_states, cand_start_mask, and cand_end_mask to max_length
            all_padded_hidden_states = []
            all_padded_start_masks = []
            all_padded_end_masks = []
            offset = 0
            num_candidates = []
            for token_length in all_token_lengths:
                # pad hidden states
                hidden_st = hidden_states[offset: offset + token_length, :]
                hidden_st = F.pad(input=hidden_st, pad=(0, 0, 0, max_length - token_length), mode='constant', value=0)
                hidden_st = hidden_st.view(1, -1, hidden_states.shape[-1])  # 1 x max_length x hidden_size
                all_padded_hidden_states.append(hidden_st)

                # pad cand start/end
                cand_starts = self.cand_starts[offset: offset + token_length]
                cand_ends = self.cand_ends[offset: offset + token_length]
                cand_starts = F.pad(input=cand_starts, pad=(0, max_length - token_length), mode='constant', value=0)
                cand_ends = F.pad(input=cand_ends, pad=(0, max_length - token_length), mode='constant', value=0)
                cand_starts = cand_starts.view(1, cand_starts.shape[-1])  # 1 x max_length
                cand_ends = cand_ends.view(1, cand_starts.shape[-1])  # 1 x max_length
                all_padded_start_masks.append(cand_starts)
                all_padded_end_masks.append(cand_ends)

                num_candidate = len(cand_ends.nonzero(as_tuple=True)[0])
                num_candidates.append(num_candidate)

                offset += token_length

            # batch of hidden_states and start_mask / end_mask
            padded_hidden_states = torch.concat(all_padded_hidden_states, dim=0)
            padded_start_masks = torch.concat(all_padded_start_masks, dim=0)
            padded_end_masks = torch.concat(all_padded_end_masks, dim=0)

            # indices of the vectors for special start and end tokens
            start_indices = padded_start_masks.nonzero(as_tuple=True)
            end_indices = padded_end_masks.nonzero(as_tuple=True)

            start_vectors = padded_hidden_states[start_indices]
            end_vectors = padded_hidden_states[end_indices]

            if self.end_token_only:
                candidate_vectors = end_vectors
            else:
                candidate_vectors = torch.cat((start_vectors, end_vectors), dim=-1)
            # print("candidate_vectors", candidate_vectors.shape)
            cand_logits = self.start_end_classifier(candidate_vectors)
            # print("cand_logits", cand_logits.shape)

            # split cand logit into list of logits
            offset = 0
            for num_candidate in num_candidates:
                all_logits.append(
                    cand_logits[offset:offset + num_candidate, :].detach().cpu().view(-1))  # Is .cpu() bad here?
                offset += num_candidate

        if len(all_logits) > 0:
            pooled_outputs = [
                PoolingSequenceGroupOutput(data) for data in all_logits
            ]
        else:
            pooled_outputs = [PoolingSequenceGroupOutput([None])]

        return PoolerOutput(outputs=pooled_outputs)


class GPTBigCodeForEmbeddingConfig(GPTBigCodeConfig):
    """
    Configuration class for GPTBigCodeForEmbedding, extending GPTBigCodeConfig
    to include additional attributes for token classification.
    """
    model_type = "gpt_bigcode_embedding"

    def __init__(
            self,
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=None,
            activation_function="gelu_pytorch_tanh",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            attention_softmax_in_fp32=True,
            scale_attention_softmax_in_fp32=True,
            multi_query=True,
            end_token_only=False,
            n_classes: int = 6,
            candidate_start_ndx: int = 5173,
            candidate_end_ndx: int = 5090,
            **kwargs,
    ):
        """
        Initializes the GPTBigCodeForEmbeddingConfig.

        Args:
            end_token_only (bool): If True, only end tokens are used for classification.
            n_classes (int): The number of classification classes.
            candidate_start_ndx (int): Token ID for start candidates.
            candidate_end_ndx (int): Token ID for end candidates.
        """
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            scale_attention_softmax_in_fp32=scale_attention_softmax_in_fp32,
            multi_query=multi_query,
            **kwargs)
        self.end_token_only = end_token_only
        self.n_classes = n_classes
        self.candidate_start_ndx = candidate_start_ndx
        self.candidate_end_ndx = candidate_end_ndx


class GPTBigCodeForEmbedding(nn.Module, SupportsLoRA, SupportsPP, TokenClassifierMixin):
    """
    GPTBigCode model adapted for token span classification.
    """
    packed_modules_mapping = {"c_attn": ["c_attn"]}

    supported_lora_modules = ["c_fc", "c_proj", "wte", "c_attn"]

    embedding_modules = {
        "wte": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config: GPTBigCodeForEmbeddingConfig = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.transformer = GPTBigCodeModel(vllm_config=vllm_config, prefix=prefix)

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.init_pooler(
            self.transformer.config.hidden_size,
            end_token_only=config.end_token_only,
            n_classes=config.n_classes,
            candidate_start_ndx=config.candidate_start_ndx,
            candidate_end_ndx=config.candidate_end_ndx,
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # set the positions of the candidate markers, this is not available later
        self.set_candidate_positions(input_ids)
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata, intermediate_tensors,
                                         inputs_embeds)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                continue
            if ".attn.bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # TODO (@robertgshaw2-neuralmagic): move to fp8 linear method
            if "c_attn.input_scale" in name or "c_attn.weight_scale" in name:
                weight_loader(param, loaded_weight, 'q')
                weight_loader(param, loaded_weight, 'k')
                weight_loader(param, loaded_weight, 'v')
            else:
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
