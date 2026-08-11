# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 with a MoonViT vision tower and PatchMerger projector."""

from collections.abc import Mapping

from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
    MaxImageTokenMeta,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import VisionChunkImage
from vllm.multimodal.processing import BaseProcessingInfo, InputProcessingContext
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5v import Glm5vConfig
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)
from vllm.utils.import_utils import is_numba_available

from .utils import WeightsMapper, init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)


class Glm5vProcessingInfo(KimiK25ProcessingInfo):
    """Use Kimi vision preprocessing with GLM's image placeholder token."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        BaseProcessingInfo.__init__(self, ctx)

        self.hf_config = hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        processor_cls = KimiK25FusedVisionProcessor if is_numba_available() else None
        logger.info_once(
            "Using %s image preprocessing for GLM-5.2-Vision.",
            "fused CPU" if processor_cls is not None else "remote HF",
        )
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls_overrides=processor_cls,
        )

        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if not is_valid_resolved:
            raise ValueError("GLM-5.2-Vision tokenizer does not contain <|image|>")
        if resolved_token_id != config_token_id:
            logger.warning_once(
                "GLM-5.2-Vision config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|image|> (%d). Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            hf_config.media_placeholder_token_id = resolved_token_id

        self.media_token_id = resolved_token_id
        self.media_token = "<|image|>"
        self.image_processor = image_processor
        self.hf_processor = KimiK25Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            media_token_id=resolved_token_id,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_config(self) -> Glm5vConfig:
        return self.ctx.get_hf_config(Glm5vConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"vision_chunk": None}


class Glm5vDummyInputsBuilder(KimiK25DummyInputsBuilder):
    """Profile the image path supported by the published checkpoint."""

    def get_dummy_mm_items(self):
        image = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=1,
        )[0]
        return [VisionChunkImage(type="image", image=image)]


class Glm5vMultiModalProcessor(KimiK25MultiModalProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm5vMultiModalProcessor,
    info=Glm5vProcessingInfo,
    dummy_inputs=Glm5vDummyInputsBuilder,
)
class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    """GLM MoE-DSA text model with MoonViT vision inputs."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|begin_of_image|><|image|><|end_of_image|>"
        if modality == "video":
            return "<|glm5v_video_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config: Glm5vConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "vision_chunk"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "vision_tower"),
            ).to(device=self.device, dtype=model_config.dtype)
            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=None,
                prefix=maybe_prefix(prefix, "mm_projector"),
            ).to(device=self.device, dtype=model_config.dtype)

        self.quant_config = quant_config
        text_architectures = config.text_config.architectures or [
            "GlmMoeDsaForCausalLM"
        ]
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=text_architectures,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder = config.media_placeholder_token_id
