# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo
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


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Wires up the vLLM-native processor for the multimodal checkpoint.

    The checkpoint's ``processor_config.json`` declares a custom ``processor_class``
    (``Glm46VProcessor``) and stores its image/video processor configs inline (no
    standalone ``preprocessor_config.json``), so ``AutoProcessor`` cannot resolve
    the config. We bypass it and build our own ``Glm5NextProcessor``
    (``vllm/transformers_utils/processors/glm5next.py``), a faithful port of the
    training-side pipeline that no longer imports transformers' ``GlmgaImageProcessor``
    / ``GlmgaVideoProcessor`` / ``Glm46VProcessor``. The port applies
    ``patch_expand_factor`` (checkpoint ships 2) inside ``smart_resize``'s spatial
    factor; dropping it (as GLM-4V's processor does) yields a wrong patch grid.
    """

    def get_hf_processor(self, **kwargs: object):
        proc = getattr(self, "_glm5_hf_processor", None)
        if proc is None:
            from vllm.transformers_utils.processors.glm5next import Glm5NextProcessor

            proc = Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)
            self._glm5_hf_processor = proc
        return proc
