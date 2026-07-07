# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM5-Next vision tower.

The architecture is identical to the GLM-OCR vision tower
(``GlmOcrVisionTransformer``); the only difference is the patch-merger's
bottleneck width, which projects through the vision config's
``projection_intermediate_size`` instead of ``text_config.intermediate_size``.
The merger output stays ``out_hidden_size`` (== text ``hidden_size``), so the
resulting embeddings splice directly into the language-model stream.
"""

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm_ocr import (
    GlmOcrPatchMerger,
    GlmOcrVisionTransformer,
)


class Glm5NextVisionPatchMerger(GlmOcrPatchMerger):
    pass


class Glm5NextVisionTransformer(GlmOcrVisionTransformer):
    def __init__(
        self,
        text_config,
        vision_config,
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            text_config,
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Override the merger to use the GLM5-Next-specific bottleneck width
        # (vision_config.projection_intermediate_size) instead of
        # text_config.intermediate_size used by GLM-OCR.
        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )
