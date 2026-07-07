# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM MoE DSA (``glm_moe_dsa``) config wrapper.

GLM-5.2 checkpoints (e.g. ``nvidia/GLM-5.2-NVFP4``) ship
``layer_types: ["deepseek_sparse_attention", ...]``, which the transformers
config validator rejects (the value is not in ``ALLOWED_LAYER_TYPES``). The
attention type is fully determined by the DSA fields (``index_topk`` etc.) and
DeepSeek V3.2 checkpoints of the same architecture omit ``layer_types``
entirely, so simply drop the field before validation.
"""

from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig as HFGlmMoeDsaConfig,
)


class GlmMoeDsaConfig(HFGlmMoeDsaConfig):
    def __init__(self, **kwargs):
        layer_types = kwargs.get("layer_types")
        if layer_types is not None and "deepseek_sparse_attention" in layer_types:
            kwargs.pop("layer_types")
        super().__init__(**kwargs)
