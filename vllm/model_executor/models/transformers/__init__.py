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

from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.causal import CausalMixin
from vllm.model_executor.models.transformers.legacy import LegacyMixin
from vllm.model_executor.models.transformers.moe import MoEMixin
from vllm.model_executor.models.transformers.multimodal import (
    DYNAMIC_ARG_DIMS,
    MultiModalDummyInputsBuilder,
    MultiModalMixin,
    MultiModalProcessingInfo,
    MultiModalProcessor,
)
from vllm.model_executor.models.transformers.pooling import (
    EmbeddingMixin,
    SequenceClassificationMixin,
)
from vllm.model_executor.models.transformers.utils import can_enable_torch_compile
from vllm.multimodal import MULTIMODAL_REGISTRY


# Text only models
@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersForCausalLM(CausalMixin, Base): ...


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEForCausalLM(MoEMixin, CausalMixin, Base): ...


# Multimodal models
@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims=DYNAMIC_ARG_DIMS, enable_if=can_enable_torch_compile
)
class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims=DYNAMIC_ARG_DIMS, enable_if=can_enable_torch_compile
)
class TransformersMultiModalMoEForCausalLM(
    MoEMixin, MultiModalMixin, CausalMixin, Base
): ...


# Embedding models
@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersEmbeddingModel(EmbeddingMixin, LegacyMixin, Base): ...


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEEmbeddingModel(EmbeddingMixin, MoEMixin, Base): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims=DYNAMIC_ARG_DIMS, enable_if=can_enable_torch_compile
)
class TransformersMultiModalEmbeddingModel(EmbeddingMixin, MultiModalMixin, Base): ...


# Sequence classification models
@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersForSequenceClassification(
    SequenceClassificationMixin, LegacyMixin, Base
): ...


@support_torch_compile(enable_if=can_enable_torch_compile)
class TransformersMoEForSequenceClassification(
    SequenceClassificationMixin, MoEMixin, Base
): ...


@MULTIMODAL_REGISTRY.register_processor(
    MultiModalProcessor,
    info=MultiModalProcessingInfo,
    dummy_inputs=MultiModalDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims=DYNAMIC_ARG_DIMS, enable_if=can_enable_torch_compile
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
