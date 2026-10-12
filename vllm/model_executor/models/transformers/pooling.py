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

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import regex as re
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from vllm.config.utils import getattr_iter
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.utils import WeightsMapper

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ClassifierWithReshape(nn.Module):
    """Token extraction has already been applied in `pooler.pooling`.

    Add dim to match expected input shape of `classifier.forward`.
    """

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            args = (args[0].unsqueeze(1), *args[1:])
        return super().forward(*args, **kwargs)


class _PreClassifierHead(nn.Module):
    """Classifier head for models that apply an activation between
    pre_classifier and classifier outside any nn.Module (e.g. DistilBERT).

    Accepts/returns [B, H] → [B, C] directly, bypassing ClassifierWithReshape.
    """

    def __init__(self, pre_classifier: nn.Linear, classifier: nn.Linear) -> None:
        super().__init__()
        self.pre_classifier = pre_classifier
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_classifier(x)
        x = torch.relu(x)
        return self.classifier(x)


# Classifier heads keyed by `model_type`. Values are a factory, which accepts
# the HF ForSequenceClassification model and returns an nn.Module implementing
# the full head, and a mapper from checkpoint names to the head's module layout.
# Models whose heads apply operations outside any nn.Module (e.g. DistilBERT's
# ReLU) must be registered here rather than relying on generic discovery.
_ClassifierHead: TypeAlias = tuple[Callable[[nn.Module], nn.Module], WeightsMapper]
_CLASSIFIER_HEADS: dict[str, _ClassifierHead] = {
    "distilbert": (
        lambda m: _PreClassifierHead(
            pre_classifier=m.pre_classifier,
            classifier=m.classifier,
        ),
        # A single pattern, as separate prefixes would be applied in sequence
        WeightsMapper(
            orig_to_new_regex={
                re.compile(r"^(pre_classifier|classifier)\."): r"classifier.\1."
            }
        ),
    ),
}


class EmbeddingMixin(VllmModelForPooling, Base):
    default_seq_pooling_type = "CLS"

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_embedding(pooler_config)


class SequenceClassificationMixin(SupportsCrossEncoding, VllmModelForPooling, Base):
    default_seq_pooling_type = "CLS"

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        # Certain information about the model and classifier can only be
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
        head_entry = _CLASSIFIER_HEADS.get(self.config.model_type)
        if head_entry is not None:
            # Model-specific head: builds the full classification chain,
            # returns [B, C] directly, bypassing ClassifierWithReshape.
            factory, weights_mapper = head_entry
            self.classifier = factory(seq_cls_model)
            self.init_parameters(self.classifier, dtype=self.model_config.head_dtype)
            self.hf_to_vllm_mapper |= weights_mapper
        else:
            self.classifier = getattr_iter(seq_cls_model, ["classifier", "score"], None)
            if self.classifier is None:
                raise ValueError(
                    "Could not find `classifier` or `score` layer in the "
                    "`AutoModelForSequenceClassification` instance."
                )
            self.init_parameters(self.classifier, dtype=self.model_config.head_dtype)

            # Order `ClassifierWithReshape` ahead of the classifier's own class so that
            # its `super().forward(...)` reaches the original implementation.
            self.classifier.__class__ = type(
                "ClassifierWithReshape",
                (ClassifierWithReshape, type(self.classifier)),
                {},
            )

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            classifier=self.classifier,
        )
