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
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

from .base import Base
from .fusers import ClassifierFuser
from .fusers.classifier import trace_classification_model

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ClassifierWithReshape(nn.Module):
    """Token extraction has already been applied in `pooler.pooling`.

    Add dim to match expected input shape of `classifier.forward`.
    """

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            args = (args[0].unsqueeze(1), *args[1:])
        return super().forward(*args, **kwargs)


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


class ClassificationMixin(VllmModelForPooling, Base):
    default_seq_pooling_type = "CLS"

    def init_classifier(
        self, auto_cls: type["_BaseAutoModelClass"], pooled: bool
    ) -> nn.Module:
        """Adopt the classifier, and the head before it, from `auto_cls`.

        Args:
            auto_cls: The auto class of the classification model.
            pooled: Whether the classifier's input is pooled.

        Returns:
            The module that maps hidden states to logits.

        """
        # Certain information about the model and classifier can only be
        # inferred from the `For*Classification` class. Therefore, we
        # instantiate it on the "meta" device to avoid allocating GPU memory.
        with torch.device("meta"):
            cls_model = auto_cls.from_config(
                self.config,
                dtype=self.model_config.dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )

        # When used for classification, some models have their
        # pooling layers removed. Make sure this is reflected in vLLM.
        for module in cls_model.modules():
            if hasattr(module, "pooler") and module.pooler is None:
                self.model.pooler = None
                break

        # Unlike `lm_head`, `classifier` is not always `nn.Linear`.
        classifier = getattr_iter(cls_model, ["classifier", "score"], None)
        if classifier is None:
            raise ValueError(
                "Could not find `classifier` or `score` layer in the "
                f"`{auto_cls.__name__}` instance."
            )

        graph = trace_classification_model(cls_model)
        fuser = (
            None
            if graph is None
            else ClassifierFuser.match(graph, cls_model, pooled=pooled)
        )
        if fuser:
            logger.debug(fuser.info(auto_cls.__name__))
            for name in dict.fromkeys(n.split(".")[0] for n in fuser.module_names):
                module = cls_model.get_submodule(name)
                if next(module.parameters(), None) is None:
                    continue
                if hasattr(self, name):
                    raise ValueError(
                        f"Classification head module {name!r} would overwrite an "
                        "existing attribute."
                    )
                dtype = self.model_config.head_dtype if module is classifier else None
                self.init_parameters(module, dtype=dtype)
                setattr(self, name, module)
            return fuser.fuse(cls_model, "", self.vllm_config)

        if graph is None or not ClassifierFuser.classifies_hidden_states(
            graph, cls_model
        ):
            raise ValueError(
                f"Could not match the head of {type(cls_model).__name__}, and its "
                "classifier does not read the last hidden state, so applying the "
                "classifier alone would give wrong results."
            )
        self.classifier = classifier
        self.init_parameters(self.classifier, dtype=self.model_config.head_dtype)
        if pooled:
            # Order `ClassifierWithReshape` ahead of the classifier's own class so
            # that its `super().forward(...)` reaches the original implementation.
            self.classifier.__class__ = type(
                "ClassifierWithReshape",
                (ClassifierWithReshape, type(self.classifier)),
                {},
            )
        return self.classifier


class SequenceClassificationMixin(SupportsCrossEncoding, ClassificationMixin):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        classifier = self.init_classifier(
            AutoModelForSequenceClassification, pooled=True
        )

        self.pooler = DispatchPooler.for_seq_cls(pooler_config, classifier=classifier)


class TokenClassificationMixin(ClassificationMixin):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForPooling.__init__ and call the next class in MRO
        super(VllmModelForPooling, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        classifier = self.init_classifier(AutoModelForTokenClassification, pooled=False)
        self.pooler = pooler_for_token_classify(pooler_config, classifier=classifier)
