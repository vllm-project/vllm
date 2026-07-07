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
"""Wrapper around `transformers` models"""

from typing import TYPE_CHECKING

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.causal import CausalMixin
from vllm.model_executor.models.transformers.legacy import LegacyMixin
from vllm.model_executor.models.transformers.moe import MoEMixin
from vllm.model_executor.models.transformers.multimodal import (
    MultiModalDummyInputsBuilder,
    MultiModalMixin,
    MultiModalProcessingInfo,
    MultiModalProcessor,
)
from vllm.model_executor.models.transformers.pooling import (
    EmbeddingMixin,
    SequenceClassificationMixin,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.attention import Attention


def vllm_attention_forward(
    # Transformers args
    module: "torch.nn.Module",
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attention_mask: "torch.Tensor",
    # Transformers kwargs
    scaling: float | None = None,
    # vLLM kwargs
    attention_instances: dict[int, "Attention"] | None = None,
    **kwargs,
):
    self_attn = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.impl.scale = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(query, key, value), None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_attention_forward


# Text only models
class TransformersForCausalLM(CausalMixin, Base): ...


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, Base): ...


# Multimodal models
@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalMoEForCausalLM(
    MoEMixin, MultiModalMixin, CausalMixin, Base
): ...


# Embedding models
class TransformersEmbeddingModel(EmbeddingMixin, LegacyMixin, Base): ...


class TransformersMoEEmbeddingModel(EmbeddingMixin, MoEMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalEmbeddingModel(EmbeddingMixin, MultiModalMixin, Base): ...


# Sequence classification models
class TransformersForSequenceClassification(
    SequenceClassificationMixin, LegacyMixin, Base
): ...


class TransformersMoEForSequenceClassification(
    SequenceClassificationMixin, MoEMixin, Base
): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
class TransformersMultiModalForSequenceClassification(
    SequenceClassificationMixin, MultiModalMixin, Base
): ...


def __getattr__(name: str):
    """Handle imports of non-existent classes with a helpful error message."""
    if name not in globals():
        raise AttributeError(
            "The Transformers modeling backend does not currently have a class to "
            f"handle the requested model type: {name}. Please open an issue at "
            "https://github.com/vllm-project/vllm/issues/new"
        )
    return globals()[name]
