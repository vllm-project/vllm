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
"""Transformers modeling backend mixins for pooling models."""

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSequenceClassification

from vllm.config.utils import getattr_iter
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class EmbeddingMixin(VllmModelForPooling):
    default_seq_pooling_type = "CLS"

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_embedding(pooler_config)


class SequenceClassificationMixin(SupportsCrossEncoding, VllmModelForPooling):
    default_seq_pooling_type = "CLS"

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        # Certain information about the the model and classifier can only be
        # inferred from the `ForSequenceClassification` class. Therefore, we
        # instantiate it on the "meta" device to avoid allocating GPU memory.
        with torch.device("meta"):
            seq_cls_model = AutoModelForSequenceClassification.from_config(
                self.config,
                dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )

        # When used for sequence classification, some models have their
        # pooling layers removed. Make sure this is reflected in vLLM.
        for module in seq_cls_model.modules():
            if hasattr(module, "pooler") and module.pooler is None:
                self.model.pooler = None
                break

        # Unlike `lm_head`, `classifier` is not always `nn.Linear`.
        self.classifier = getattr_iter(seq_cls_model, ["classifier", "score"], None)
        if self.classifier is None:
            raise ValueError(
                "Could not find `classifier` or `score` layer in the "
                "`AutoModelForSequenceClassification` instance."
            )
        self.init_parameters(self.classifier, dtype=self.model_config.head_dtype)

        class ClassifierWithReshape(self.classifier.__class__):
            """
            Token extraction has already been applied in `pooler.pooling`.
            Add dim to match expected input shape of `classifier.forward`.
            """

            def forward(self, *args, **kwargs):
                if len(args) > 0:
                    args = (args[0].unsqueeze(1), *args[1:])
                return super().forward(*args, **kwargs)

        self.classifier.__class__ = ClassifierWithReshape

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            classifier=self.classifier,
        )
