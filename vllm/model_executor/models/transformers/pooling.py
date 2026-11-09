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

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

from vllm.config.utils import getattr_iter
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.model_executor.models.interfaces import SupportsCrossEncoding
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.transformers.fx_utils import forward_parameters
from vllm.model_executor.models.transformers.utils import (
    OpaqueBaseModel,
    head_forward_kwargs,
    init_on_device_without_buffers,
    replace_base_model,
)
from vllm.transformers_utils.dynamic_module import get_transformers_model_class

from .base import Base

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    from vllm.config import VllmConfig
    from vllm.config.pooler import SequencePoolingType


class ClassificationHead:
    """Runs a `For*Classification` model whose base model is `OpaqueBaseModel`.

    Pooled hidden states are passed as sequences of length 1, so the model's own
    pooling is a no-op and only the layers after it apply. Not an `nn.Module`, so
    that its weights stay registered under their checkpoint names.
    """

    def __init__(self, model: PreTrainedModel, pooled: bool, dtype: torch.dtype):
        self.model = model
        self.pooled = pooled
        self.dtype = dtype
        self.kwargs = head_forward_kwargs(model)
        self.takes_attention_mask = "attention_mask" in forward_parameters(type(model))

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """The logits for `inputs_embeds` of shape `[batch, seq_len, hidden]`."""
        kwargs = dict(self.kwargs)
        if self.takes_attention_mask:
            kwargs["attention_mask"] = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )
        return self.model(inputs_embeds=inputs_embeds, **kwargs).logits

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        if self.pooled:
            return self.forward(hidden_states.unsqueeze(1))
        return self.forward(hidden_states.unsqueeze(0))[0]


def _pool(
    hidden_states: torch.Tensor, pooling_type: "SequencePoolingType"
) -> torch.Tensor:
    if pooling_type == "CLS":
        return hidden_states[:1]
    if pooling_type == "LAST":
        return hidden_states[-1:]
    return hidden_states.mean(0, keepdim=True)


def probe_classification_head(
    head: ClassificationHead,
    hidden_size: int,
    pooling_type: "SequencePoolingType | None",
) -> str | None:
    """Why `head` can't follow vLLM's pooling, or `None` if it can.

    Runs a copy of `head` with random weights on CPU. Pooled heads must give the
    same logits for a whole sequence as for the sequence pooled with
    `pooling_type` (if `None`, they only have to run). Token heads must give the
    same logits for a sequence as for its parts, because vLLM concatenates the
    tokens of all requests.
    """
    model = copy.deepcopy(head.model)
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            setattr(module, name, nn.Parameter(torch.randn(param.shape)))
    probe = ClassificationHead(model, head.pooled, torch.float32)
    hidden_states = torch.randn(5, hidden_size)
    try:
        with torch.no_grad():
            if head.pooled:
                expected = probe.forward(hidden_states.unsqueeze(0))
                actual = probe(_pool(hidden_states, pooling_type or "CLS"))
                if pooling_type is None:
                    actual = expected
            else:
                expected = probe(hidden_states)
                actual = torch.cat([probe(hidden_states[:2]), probe(hidden_states[2:])])
    except Exception as e:
        return f"it failed to run on hidden states ({type(e).__name__}: {e})"
    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        if head.pooled:
            return f"it does not pool like vLLM's {pooling_type!r} pooling"
        return "it mixes tokens, which would mix the tokens of different requests"
    return None


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
    ) -> ClassificationHead:
        """Adopt everything after the base model in the model's `auto_cls` class.

        Args:
            auto_cls: The auto class of the classification model.
            pooled: Whether the classifier's input is pooled.

        Returns:
            The head that maps hidden states to logits.

        """
        model = self._create_classification_model(auto_cls)

        # When used for classification, some models have their
        # pooling layers removed. Make sure this is reflected in vLLM.
        for module in model.modules():
            if hasattr(module, "pooler") and module.pooler is None:
                self.model.pooler = None
                break

        # Unlike `lm_head`, `classifier` is not always `nn.Linear`.
        classifier = getattr_iter(model, ["classifier", "score"], None)
        if classifier is None:
            raise ValueError(
                "Could not find `classifier` or `score` layer in the "
                f"`{type(model).__name__}` instance."
            )

        name = type(model).__name__
        if not replace_base_model(model, type(self.model)):
            raise ValueError(f"Could not find the base model in {name}.")
        head = ClassificationHead(model.eval(), pooled, self.model_config.dtype)
        pooler_config = self.model_config.pooler_config
        assert pooler_config is not None
        # A pooling type chosen by the user is not checked against the model's
        sources = getattr(self.model_config, "_pooler_config_sources", {})
        user_pooling = sources.get("seq_pooling_type") == "user"
        pooling_type = None if user_pooling else pooler_config.seq_pooling_type
        hidden_size = self.text_config.hidden_size
        if reason := probe_classification_head(head, hidden_size, pooling_type):
            raise ValueError(f"Could not use the head of {name}, because {reason}.")

        for child_name, child in model.named_children():
            if isinstance(child, OpaqueBaseModel):
                continue
            if next(child.parameters(), None) is None:
                continue
            if hasattr(self, child_name):
                raise ValueError(
                    f"Classification head module {child_name!r} would overwrite an "
                    "existing attribute."
                )
            setattr(self, child_name, child)
        # Like vLLM's own models, only the classifier runs in `head_dtype`
        head_dtype = self.model_config.head_dtype
        self.init_parameters(classifier, dtype=head_dtype)
        self.init_parameters(model)
        classifier.register_forward_pre_hook(
            lambda module, args: (args[0].to(head_dtype), *args[1:])
        )
        return head

    def _create_classification_model(
        self, auto_cls: type["_BaseAutoModelClass"]
    ) -> PreTrainedModel:
        """The model's Transformers class, from its architectures or `auto_cls`,
        on the "meta" device."""
        suffix = auto_cls.__name__.removeprefix("AutoModel")
        model_cls = next(
            (
                cls
                for arch in self.config.architectures or []
                if arch.endswith(suffix)
                and (cls := get_transformers_model_class(arch, self.model_config))
            ),
            None,
        )
        dtype = self.model_config.dtype
        with init_on_device_without_buffers("meta"):
            if model_cls is not None:
                return model_cls._from_config(self.config, dtype=dtype)
            return auto_cls.from_config(
                self.config,
                dtype=dtype,
                trust_remote_code=self.model_config.trust_remote_code,
            )


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
