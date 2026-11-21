# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiDE (fastino/GLiDE-27B): a Qwen3.5 decision model with recurrent depth.

The checkpoint is a `Qwen3_5ForConditionalGeneration` without an LM head, plus
`readout.safetensors`, one row per answer code, and `decision_config.json`.
It is read like Decision 3.0: the readout scores the codes at the last prompt
token and is served as the `score` head of a classification model.
`GlideForSequenceClassificationConfig` applies the checkpoint's attention mode,
label space and recurrent depth (`midloop`), which runs blocks of middle layers
more than once inside the forward pass.
"""

from copy import copy
from itertools import islice

import torch
from transformers.models.qwen3_vl import Qwen3VLProcessor

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from .d3 import D3ForSequenceClassification
from .interfaces_base import default_pooling_type
from .qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ProcessingInfo,
)
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor

# The image budget of the checkpoint's runtime (`glide/images.py`), which
# differs from its processor config: 256 x 256 to 1024 x 1024 pixels.
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_048_576}


class GlideProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        processor = copy(super().get_hf_processor(**kwargs))
        processor.image_processor = copy(processor.image_processor)
        processor.image_processor.size = dict(IMAGE_SIZE)
        return processor


class GlideModel(Qwen3_5Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        depth = getattr(self.config, "recurrent_depth", None)
        self.recurrent_blocks = (
            {last: first for first, last in depth["blocks"]} if depth else {}
        )
        self.recurrent_passes = depth["passes"] if depth else 1

    def _repeat_block(
        self,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
        first: int,
        last: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the checkpoint's damped Euler steps over layers `first..last`."""
        step = 1.0 / self.recurrent_passes
        x = block_input
        for n in range(self.recurrent_passes):
            if n > 0:
                block_output, residual = x.clone(), None
                for layer in islice(self.layers, first, last + 1):
                    block_output, residual = layer(
                        positions=positions,
                        hidden_states=block_output,
                        residual=residual,
                    )
                block_output = block_output + residual
            x = torch.lerp(x.float(), block_output.float(), step).to(x.dtype)
        return x

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not self.recurrent_blocks:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )

        # Recurrent GLiDE is a dense classification model on one pipeline rank.
        assert intermediate_tensors is None
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        block_firsts = self.recurrent_blocks.values()
        block_inputs: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in block_firsts:
                block_inputs[layer_idx] = (
                    hidden_states.clone()
                    if residual is None
                    else hidden_states + residual
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (first := self.recurrent_blocks.get(layer_idx)) is not None:
                hidden_states = self._repeat_block(
                    block_inputs.pop(first),
                    hidden_states + residual,
                    first,
                    layer_idx,
                    positions,
                )
                residual = None
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class GlideForCausalLM(Qwen3_5ForCausalLM):
    model_cls = GlideModel


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=GlideProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class GlideForSequenceClassification(D3ForSequenceClassification):
    # The checkpoint keeps the names of a Qwen3.5 generation checkpoint.
    hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
    language_model_cls = GlideForCausalLM
