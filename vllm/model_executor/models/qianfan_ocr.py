# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# QianfanOCR is built on InternVL with a Qwen3 language backbone.
# The model architecture and weights are fully compatible with InternVLChatModel,
# only the config model_type / architectures strings differ.

from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.processors.internvl import (
    InternVLImageProcessor,
    InternVLProcessor,
)

from .internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
    InternVLChatModel,
)


class QianfanOCRProcessingInfo(BaseInternVLProcessingInfo):
    """Image-only ProcessingInfo for QianfanOCR (no video support)."""

    def get_hf_processor(self, **kwargs: object) -> InternVLProcessor:
        config = self.get_hf_config()
        vision_config = config.vision_config

        kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        kwargs.setdefault("image_size", vision_config.image_size)
        kwargs.setdefault("min_dynamic_patch", config.min_dynamic_patch)
        kwargs.setdefault("max_dynamic_patch", config.max_dynamic_patch)
        kwargs.setdefault("dynamic_image_size", config.dynamic_image_size)
        kwargs.setdefault("use_thumbnail", config.use_thumbnail)

        image_processor = InternVLImageProcessor(**kwargs)
        image_size = image_processor.image_size
        patch_size = vision_config.patch_size
        downsample_ratio = config.downsample_ratio
        image_seq_length = int((image_size // patch_size) ** 2 * (downsample_ratio**2))

        return InternVLProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=image_processor,
            video_processor=None,
            image_seq_length=image_seq_length,
            ctx_video_token=None,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor,
    info=QianfanOCRProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder,
)
class QianfanOCRForConditionalGeneration(InternVLChatModel):
    """QianfanOCR multimodal model.

    Identical in structure to InternVLChatModel (InternViT vision encoder +
    pixel-shuffle MLP connector + Qwen3 language model).  This class exists
    solely to register the ``QianfanOCRForConditionalGeneration`` architecture
    name that appears in the model's config.json.
    """

    def _patch_quant_config(
        self, config: PretrainedConfig, quant_config: QuantizationConfig
    ) -> None:
        super()._patch_quant_config(config, quant_config)
        # ignore vit layers to preserve model performance
        if isinstance(quant_config, Fp8Config):
            _FP8_IGNORED_LAYERS = [
                *(
                    layer
                    for i in range(config.vision_config.num_hidden_layers)
                    for layer in [
                        f"vision_model.encoder.layers.{i}.attn.qkv",
                        f"vision_model.encoder.layers.{i}.attn.proj",
                        f"vision_model.encoder.layers.{i}.mlp.fc1",
                        f"vision_model.encoder.layers.{i}.mlp.fc2",
                    ]
                ),
                "language_model.lm_head",
                "mlp1.1",
                "mlp1.3",
            ]
            for layer in _FP8_IGNORED_LAYERS:
                if layer not in quant_config.ignored_layers:
                    quant_config.ignored_layers.append(layer)
