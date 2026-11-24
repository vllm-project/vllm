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
"""Transformers modeling backend mixin for causal language models."""

import operator
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.last_layer_ffn import install_transformer_ffn_hooks
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .base import Base
from .fuser import get_fuser
from .fusers.glu import GLUFuser
from .fusers.rms_norm import RMSNormFuser
from .fx_utils import trace

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _terminal_ffn(block: torch.nn.Module) -> str | None:
    """Identify a token-wise GLU before fusion, with no later token mixing."""
    graph = trace(block)
    if graph is None or not any(n.op == "output" for n in graph.nodes):
        return None
    selected = None
    for node in graph.nodes:
        if node.op == "call_module":
            child = block.get_submodule(node.target)
            if selected is not None:
                if not (
                    isinstance(child, (RMSNorm, torch.nn.LayerNorm))
                    or get_fuser(child, RMSNormFuser) is not None
                ):
                    return None
                continue
            fuser = get_fuser(child, GLUFuser)
            if fuser is None or fuser.down_name is None:
                continue
            # A GLU match may be only a subgraph; reject extra computation.
            inner = trace(child)
            names = {*fuser.linear_names, fuser.act_name, fuser.down_name}
            if (
                inner is not None
                and all(
                    n.op in ("placeholder", "output")
                    or (n.op == "call_module" and n.target in names)
                    or (n.op == "call_function" and n.target == operator.mul)
                    for n in inner.nodes
                )
                and any(n.op == "output" for n in inner.nodes)
            ):
                selected = node.target
        elif selected is not None and not (
            node.op == "output"
            or (node.op == "call_function" and node.target == operator.add)
        ):
            return None
    return selected


class CausalMixin(VllmModelForTextGeneration, Base):
    embedding_modules = {"lm_head": "output_embeddings"}

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        tie_word_embeddings = self._get_tie_word_embeddings()

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if tie_word_embeddings:
                for module in self.model.get_input_embeddings().modules():
                    if isinstance(module, VocabParallelEmbedding):
                        self.lm_head = self.lm_head.tie_weights(module)
                        break

            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size,
                scale=getattr(self.text_config, "logit_scale", 1.0),
                soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
            )
        else:
            self.lm_head = PPMissingLayer()

    def recursive_replace(self):
        self._ffn_pruning_supported = False
        target = None
        if (
            envs.VLLM_ENABLE_LAST_LAYER_FFN_PRUNING
            and self.pp_group.is_last_rank
            and not self.model_config.is_moe
            and not self.model_config.is_multimodal_model
            and self.quant_config is None
            and (layers := self._decoder_layers) is not None
            and len(layers)
        ):
            target = _terminal_ffn(layers[-1])
        super().recursive_replace()
        if target is not None:
            layers = self._decoder_layers
            assert layers is not None
            install_transformer_ffn_hooks(layers[-1].get_submodule(target))
            self._ffn_pruning_supported = True

    @property
    def supports_last_layer_ffn_pruning(self) -> bool:
        return self._ffn_pruning_supported

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the lm_head bias."""
        lm_head_bias = set()
        load_device = self.vllm_config.load_config.device or self.device_config.device

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                if name.endswith("lm_head.bias") and self.pp_group.is_last_rank:
                    # load_weights is called outside vLLM's device context,
                    # so we must enter it again when registering the bias.
                    with torch.device(load_device):
                        self.lm_head._register_bias()
                    self.lm_head.bias.weight_loader(self.lm_head.bias, weight)
                    lm_head_bias.add(name)
                else:
                    yield name, weight

        return super().load_weights(auto_load_lm_head_bias(weights)) | lm_head_bias

    def compute_logits_local(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """Logits for this TP rank's vocab shard, for batch-sharded sampling."""
        bias = self.lm_head.bias
        return self.logits_processor(
            self.lm_head, hidden_states, bias, skip_gather=True
        )

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
