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
"""Wrapper around `transformers` models for pooling tasks."""

from typing import Optional, Union

import torch
from transformers import AutoModelForSequenceClassification

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import (
    ClassifierPooler,
    CLSPool,
    DispatchPooler,
    Pooler,
)
from vllm.sequence import IntermediateTensors

from .interfaces_base import VllmModelForPooling
from .transformers import TransformersBase, can_enable_torch_compile
from .transformers_moe import TransformersMoEBase
from .utils import WeightsMapper


class TransformersPoolingBase(TransformersBase, VllmModelForPooling):
    hf_to_vllm_mapper = WeightsMapper(
        # These are applied in order, so the order matters!
        orig_to_new_prefix={
            # Handle BERT-like models
            "roberta": "model",
            "bert": "model",
            # Add `model.` prefix for base model checkpoints
            "": "model.",
            # Remove `model.` prefix if it was already there
            "model.model.": "model.",
            # Classifier/scoring heads will be adjacent to `model`
            "model.score": "classifier",
            "model.classifier": "classifier",
        },
        orig_to_new_suffix={
            # Replace legacy suffixes used for norms
            ".gamma": ".weight",
            ".beta": ".bias",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Skip unsupported/unwanted output embeddings layers
        self.skip_prefixes.extend(
            [
                "model.lm_head.",
                "model.predictions.",
                "model.qa_outputs.",
                "model.embeddings_project.",
                "model.discriminator_predictions.",
            ]
        )

        # Some encoder models have the position_ids buffer in the checkpoint.
        # vLLM will always pass position_ids as an argument, so we skip loading
        # the buffer if it exists
        self.skip_substrs.append("position_ids")

        # Some encoder models have the bias of the final classifier layer
        # in the checkpoint. vLLM does not use this bias, so we skip loading
        # it if it exists
        self.skip_substrs.append("score.bias")

        # roberta-like models an extra padding in positions.
        # FIXME(Isotr0py): This is quite hacky for roberta edge case,
        # we should find a better way to handle this.
        self.is_roberta = "roberta" in self.text_config.model_type
        self.padding_idx = self.text_config.pad_token_id

    def create_attention_instances(
        self, attn_type: AttentionType = AttentionType.DECODER
    ) -> dict[int, Attention]:
        # TODO(hmellor): Better way to detect encoder models
        # In encoder models, the attention layers will have `is_causal=False`
        is_encoder = lambda m: not getattr(m, "is_causal", True)
        # vLLM does not support encoder-decoder models, so if any encoder layer
        # is found, we assume the whole model is an encoder model
        if any(is_encoder(m) for m in self.model.modules()):
            attn_type = AttentionType.ENCODER_ONLY

        # Check minimum transformers version for encoder models support
        if attn_type == AttentionType.ENCODER_ONLY:
            self.check_version("4.57.0.dev0", "encoder models support")

        return super().create_attention_instances(attn_type)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.is_roberta:
            # RoBERTa-specific positions padding
            positions += self.padding_idx + 1
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersEmbeddingModel(TransformersPoolingBase):
    default_pooling_type = "CLS"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {
                "encode": Pooler.for_encode(pooler_config),
                "embed": Pooler.for_embed(pooler_config),
            }
        )


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersForSequenceClassification(TransformersPoolingBase):
    default_pooling_type = "CLS"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        # Certain information about the the model and classifier can only be
        # inferred from the `ForSequenceClassification` class. Therefore, we
        # instantiate it on the "meta" device to avoid allocating GPU memory.
        with torch.device("meta"):
            seq_cls_model = AutoModelForSequenceClassification.from_config(
                self.config,
                torch_dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )

        # When used for sequence classification, some models have their
        # pooling layers removed. Make sure this is reflected in vLLM.
        for module in seq_cls_model.modules():
            if hasattr(module, "pooler") and module.pooler is None:
                self.model.pooler = None
                break
        if self.model.pooler is not None:
            raise ValueError(
                "Sequence classification models with pooling layers are not "
                "supported yet in the Transformers backend."
            )

        # Unlike `lm_head`, `classifier` is not always `nn.Linear`.
        self.classifier = seq_cls_model.classifier
        self.init_parameters(self.classifier, dtype=self.model_config.head_dtype)

        class ClassifierWithReshape(self.classifier.__class__):
            """CLSPool has already been applied in `pooling`.
            Add dim to match expected input shape of `classifier.forward`."""

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    args = (args[0].unsqueeze(1), *args[1:])
                return super().forward(*args, **kwargs)

        self.classifier.__class__ = ClassifierWithReshape

        self.pooler = DispatchPooler(
            {
                "encode": Pooler.for_encode(pooler_config),
                "classify": ClassifierPooler(
                    pooling=CLSPool(),
                    classifier=self.classifier,
                    act_fn=ClassifierPooler.act_fn_for_seq_cls(
                        vllm_config.model_config
                    ),
                ),
                "score": ClassifierPooler(
                    pooling=CLSPool(),
                    classifier=self.classifier,
                    act_fn=ClassifierPooler.act_fn_for_cross_encoder(
                        vllm_config.model_config
                    ),
                ),
            }
        )


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEEmbeddingModel(TransformersMoEBase, TransformersEmbeddingModel):
    pass


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEForSequenceClassification(
    TransformersMoEBase, TransformersForSequenceClassification
):
    pass
