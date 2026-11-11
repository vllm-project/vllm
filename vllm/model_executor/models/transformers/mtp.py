# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The vLLM team.
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
"""Transformers modeling backend mixin for multi-token prediction draft models."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM
from transformers.conversion_mapping import WeightTransform

from vllm.model_executor.models.transformers.causal import CausalMixin
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class MTPMixin(CausalMixin):
    """Serve the MTP layers of a model which opts into the Transformers MTP standard
    (i.e. its config sets `num_mtp_layers`) as a draft model.

    `self.model` is the Transformers `MtpModel`, whose embedding and LM head the
    proposer replaces with the target's.
    """

    def _patch_config(self):
        # `vllm_config.model_config` is the target's, which must not be patched
        assert self.vllm_config.speculative_config is not None
        self.model_config = self.vllm_config.speculative_config.draft_model_config
        self.config = self.model_config.hf_config
        self.text_config = self.config.get_text_config()
        super()._patch_config()

    def _create_model(self) -> "PreTrainedModel":
        self.check_version("5.20.0", "MTP")
        from transformers.modeling_layers import MtpModel

        main_model = AutoModelForCausalLM.from_config(**self._from_config_kwargs)
        num_mtp_layers = getattr(self.text_config, "num_mtp_layers", None)
        mtp_model = MtpModel(main_model, num_mtp_layers)
        self.mtp_weights = MtpModel.get_mtp_weights_regex(main_model)
        self.mtp_weight_conversions = mtp_model.get_weight_conversions(main_model)

        # The proposer shares the target's LM head with the draft
        del mtp_model.shared_head
        # With PP, the last rank has no target embedding to share with the draft
        if self.pp_group.world_size > 1:
            mtp_model._keys_to_ignore_on_load_unexpected = [
                r"layers\..*embed_tokens\.weight" if "embed_tokens" in k else k
                for k in mtp_model._keys_to_ignore_on_load_unexpected
            ]
        self.text_config = mtp_model.config
        return mtp_model

    def _weight_conversions(self) -> list[WeightTransform]:
        return self.mtp_weight_conversions

    def _tp_plan(self) -> dict:
        tp_plan = self.text_config.base_model_tp_plan or {}
        return {
            k.replace("layers.*.", "layers.*.mtp_block.", 1): v
            for k, v in tp_plan.items()
            if k.startswith("layers.*.")
        }

    # The draft runs whole on the last PP rank
    def pipeline_parallel(self):
        pass

    def _pp_indices(self) -> tuple[int, int]:
        return 0, self.text_config.num_hidden_layers

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        embed_prefix = None
        if self.pp_group.world_size > 1:
            embed = self.model.get_input_embeddings()
            embed_prefix = next(n for n, m in self.named_modules() if m is embed) + "."

        def is_draft_weight(name: str) -> bool:
            if self.mtp_weights.search(name):
                return True
            if embed_prefix is None:
                return False
            mapped = self.hf_to_vllm_mapper.map_name(name)
            return mapped is not None and mapped.startswith(embed_prefix)

        loaded = super().load_weights(w for w in weights if is_draft_weight(w[0]))
        if embed_prefix is not None:
            self.has_own_embed_tokens = any(n.startswith(embed_prefix) for n in loaded)
        return loaded

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Add batch dimension before entering Transformers modules
        inputs_embeds = inputs_embeds[None, ...]
        hidden_states = hidden_states[None, ...]

        hidden_states = self.model.forward_layer(
            spec_step_idx % len(self.model.layers),
            hidden_states,
            positions[None, ...],
            mtp_cache=None,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Remove batch dimension after exiting Transformers modules
        return hidden_states[0, ...]

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        # The MTP layers share the LM head
        return super().compute_logits(hidden_states)
