# SPDX-License-Identifier: Apache-2.0
from typing import (Any, Iterable, Literal, Mapping, Optional, Sequence, Set,
                    Tuple, TypedDict, Union)

import torch
from torch import nn
from transformers import BatchFeature, GotOcr2ImageProcessor
from transformers.activations import ACT2FN
from transformers.models.aya_vision import AyaVisionConfig
from transformers.models.aya_vision.processing_aya_vision import (
    AyaVisionProcessor, AyaVisionProcessorKwargs)

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .siglip import SiglipEncoderInfo, SiglipVisionModel
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)
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

    vision_feature_layer: int
    vision_feature_select_strategy: str


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

    def get_num_patches(self,
                        *,
                        processor: AyaVisionProcessor,
                        image_width: int,
                        image_height: int,
                        patch_size: dict,
                        min_patches: int,
                        max_patches: int,
                        use_thumbnail: bool = True) -> int:
        """
        reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/ayavision/processing_ayavision.py#L100
        """
        image_processor: GotOcr2ImageProcessor = processor.image_processor

        patch_size_height, patch_size_width = patch_size["height"], patch_size[
            "width"]

        num_columns, num_rows = image_processor.get_optimal_tiled_canvas(
            (image_height, image_width), (patch_size_height, patch_size_width),
            min_patches, max_patches)
        num_blocks = num_columns * num_rows

        if use_thumbnail and num_blocks != 1:
            num_patches = num_blocks + 1
        else:
            num_patches = num_blocks

        return num_patches


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


class AyaVisionMultiModalProcessor(
        BaseMultiModalProcessor[AyaVisionProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        pixel_values = hf_inputs.get("pixel_values", torch.empty(0))
        num_patches = pixel_values.shape[0]
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            num_patches=num_patches,
            num_embeds=MultiModalFieldConfig.batched("image"),
            embed_is_patch=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _prompt_split_image(self, num_patches, processor: AyaVisionProcessor):

        img_patches_per_tile = (processor.img_size // processor.patch_size)**2
        img_string = f"{processor.start_of_img_token}"
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{processor.tile_token}_{idx}" + f"{processor.img_patch_token}" * img_patches_per_tile

        img_string += f"{processor.tile_global_token}" + f"{processor.img_patch_token}" * img_patches_per_tile
        img_string += f"{processor.end_of_img_token}"
        return img_string

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = hf_processor.image_processor
        image_token = hf_processor.image_token

        def get_replacement(item_idx: int):
            images = mm_items.get("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            num_patches = self.info.get_num_patches(
                hf_processor,
                image_width=image_size["width"],
                image_height=image_size["height"],
                patch_size=hf_processor.patch_size,
                min_patches=image_processor.min_patches,
                max_patches=image_processor.max_patches)
            return self._prompt_split_image(num_patches=num_patches,
                                            processor=hf_processor)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    AyaVisionMultiModalProcessor,
    info=AyaVisionProcessingInfo,
    dummy_inputs=AyaVisionDummyInputsBuilder)
class AyaVisionForConditionalGeneration(nn.Module, SupportsMultiModal):

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
                                                  prefix, "vision_model"))
        self.vocab_size = config.text_config.vocab_size
        self.multi_modal_projector = AyaVisionMultiModalProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "model"),
            architectures=["CohereForCausalLM"])

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=(['lm_head','rotary_emb']
                           if self.config.tie_word_embeddings else None))
        return loader.load_weights(weights)

    def _process_image_input(
            self, image_input: AyaVisionImagePixelInputs) -> torch.Tensor:
        # TODO: Implement this
        assert self.vision_encoder is not None
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        # TODO: Implement this Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # TODO: Implement this
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index)

        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                pixel_values: torch.Tensor, **kwargs) -> SamplerOutput:
        # TODO: Implement this
        pass
