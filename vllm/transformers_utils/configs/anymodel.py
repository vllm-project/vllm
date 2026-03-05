# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AnyModel configuration.

AnyModel wraps arbitrary base architectures (Llama, Qwen, etc.) with
per-layer heterogeneous configs via ``block_configs``.  This thin
PretrainedConfig subclass lets HuggingFace load the config when
``model_type`` is ``"anymodel"`` while still accepting all base-model
attributes through ``**kwargs``.
"""

from transformers import PretrainedConfig


class AnyModelConfig(PretrainedConfig):
    model_type = "anymodel"
