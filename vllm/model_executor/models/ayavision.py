# SPDX-License-Identifier: Apache-2.0
from typing import (Any, Iterable, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
from torch import nn
from transformers import GotOcr2ImageProcessor
from transformers.activations import ACT2FN
from transformers.models.aya_vision import AyaVisionConfig
from transformers.models.aya_vision.processing_aya_vision import (
    AyaVisionProcessor, AyaVisionProcessorKwargs)

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .siglip import SiglipEncoderInfo, SiglipVisionModel
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
from .vision import get_vision_encoder_info


class AyaVisionImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(num_patches_total, num_channels, height, width)`

    `num_patches_total` is the total number of patches over each image over each
    prompt in the batch.
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond to patch tokens.

    Shape: `(batch_size, num_images, num_embeds)`
    """

    num_embeds: Union[torch.Tensor, list[torch.Tensor]]
    """Shape: `(batch_size, num_images)`"""


class AyaVisionMultiModalProjector(nn.Module):

    def __init__(self, config: AyaVisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.alignment_intermediate_size = getattr(
            config, "alignment_intermediate_size",
            config.text_config.hidden_size)
        self.layernorm = nn.LayerNorm(config.vision_config.hidden_size *
                                      (config.downsample_factor**2),
                                      eps=config.adapter_layer_norm_eps)

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            bias=True,
        )

        self.act = ACT2FN["silu"]  # SwiGLU uses SiLU activation
        # For SwiGLU, project down to half size since we split intermediate dim
        self.linear_2 = nn.Linear(self.alignment_intermediate_size // 2,
                                  config.text_config.hidden_size,
                                  bias=True)

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features):  # B, S, D
        batch_size, seq_length, _ = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(image_features.shape[0], width,
                                                height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size, width, int(height / self.downsample_factor),
            int(channels * self.downsample_factor))
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size, int(height / self.downsample_factor),
            int(width / self.downsample_factor), -1)
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features


class AyaVisionProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(AyaVisionConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object) -> AyaVisionProcessor:
        return self.ctx.get_hf_processor(AyaVisionProcessor, **kwargs)

    def get_image_processor(self) -> GotOcr2ImageProcessor:
        return self.get_hf_processor().image_processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_max_image_tokens(self) -> int:
        vision_encoder_info: SiglipEncoderInfo = self.get_vision_encoder_info()
        patch_size: int = vision_encoder_info.get_patch_size()
        processor: AyaVisionProcessor = self.get_hf_processor()
        tokenizer = processor.tokenizer
        image_string = processor._prompt_split_image(patch_size)
        return len(tokenizer(image_string))

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        # image_processor = self.get_image_processor() height =
        # image_processor.size['height'] width = image_processor.size['width']
        # max_pathes = image_processor.max_pathes return ImageSize(height=height
        # * max_pathes, width=width * max_pathes) It is ok to do so since aya
        # vision model will do resize
        return ImageSize(height=9999999, width=9999999)

    def _resolve_image_kwargs(
        self,
        processor: AyaVisionProcessor,
        keys: set[str],
    ) -> dict[str, Any]:
        image_processor = processor.image_processor
        kwargs = processor._merge_kwargs(
            AyaVisionProcessorKwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        )

        images_kwargs = kwargs["images_kwargs"]

        def _resolve_kw(key: str):
            val = getattr(image_processor, key)
            if val is None:
                val = images_kwargs[key]

            return val

        return {k: _resolve_kw(k) for k in keys}


class AyaVisionDummyInputsBuilder(
        BaseDummyInputsBuilder[AyaVisionProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class AyaVisionModalProcessor(BaseMultiModalProcessor[AyaVisionProcessingInfo]
                              ):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AyaVisionModalProcessor,
    info=AyaVisionProcessingInfo,
    dummy_inputs=AyaVisionDummyInputsBuilder)
class AyaVisionForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: AyaVisionConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.vocab_size = config.text_config.vocab_size
        self.multi_modal_projector = AyaVisionMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Cohere2ForCausalLM"])

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def sampler(self):
        return self.language_model.sampler

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            if d.shape != expected_dims:
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_dims}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

    def _image_pixels_to_features(
            self, vision_tower: SiglipVisionModel, pixel_values: torch.Tensor,
            vision_feature_layer: int,
            vision_feature_select_strategy: str) -> torch.Tensor:
        target_dtype = vision_tower.get_input_embeddings().weight.dtype
        image_features = vision_tower(pixel_values.to(dtype=target_dtype),
                                      output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[
            vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
