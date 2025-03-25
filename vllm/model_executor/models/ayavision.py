# SPDX-License-Identifier: Apache-2.0
from functools import cached_property
from typing import (Any, Iterable, Literal, Mapping, Optional, Sequence, Set,
                    Tuple, TypedDict, Union, cast)

import torch
from torch import nn
from transformers import BatchFeature, GotOcr2ImageProcessor
from transformers.activations import ACT2FN
from transformers.models.aya_vision import AyaVisionConfig
from transformers.models.aya_vision.processing_aya_vision import (
    AyaVisionProcessor, AyaVisionProcessorKwargs)
from transformers.models.got_ocr2.image_processing_got_ocr2 import (
    get_optimal_tiled_canvas)

from vllm.config import VllmConfig
from vllm.jsontree import json_map_leaves
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate,
                                        encode_tokens)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
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

    def _prompt_split_image(self, num_patches, processor: AyaVisionProcessor):

        img_patches_per_tile = (processor.img_size // processor.patch_size)**2
        img_string = f"{processor.start_of_img_token}"
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{processor.tile_token}_{idx}" + f"{processor.img_patch_token}" * img_patches_per_tile

        img_string += f"{processor.tile_global_token}" + f"{processor.img_patch_token}" * img_patches_per_tile
        img_string += f"{processor.end_of_img_token}"
        return img_string

    def get_max_image_tokens(self) -> int:
        hf_processor: AyaVisionProcessor = self.get_hf_processor()
        image_processor = hf_processor.image_processor
        tokenizer = hf_processor.tokenizer
        num_patches = self.get_num_patches(
            image_width=image_processor.size['width'],
            image_height=image_processor.size['height'],
            patch_size=hf_processor.patch_size,
            min_patches=image_processor.min_patches,
            max_patches=image_processor.max_patches)
        image_string = self._prompt_split_image(num_patches,
                                                processor=hf_processor)
        x = encode_tokens(
            tokenizer,
            image_string,
            add_special_tokens=False,
        )
        return len(x)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        height = image_processor.size['height']
        width = image_processor.size['width']
        max_patches = image_processor.max_patches
        return ImageSize(height=height * max_patches,
                         width=width * max_patches)

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
                        image_width: int,
                        image_height: int,
                        patch_size: int,
                        min_patches: int,
                        max_patches: int,
                        use_thumbnail: bool = True) -> int:
        """
        reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/ayavision/processing_ayavision.py#L100
        """
        num_columns, num_rows = get_optimal_tiled_canvas(
            (image_height, image_width), (patch_size, patch_size), min_patches,
            max_patches)
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
        image_size = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=image_size.width,
                                   height=image_size.height,
                                   num_images=num_images)
        }
        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class AyaVisionMultiModalProcessor(
        BaseMultiModalProcessor[AyaVisionProcessingInfo]):
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
        )

        # HF processor pops the `num_patches` kwarg, which is needed by vLLM
        if (images := mm_data.get("images")) is not None:
            assert isinstance(images, list)

            parsed_images = (self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", ImageProcessorItems))
            image_sizes = [
                parsed_images.get_image_size(i)
                for i in range(len(parsed_images))
            ]
            hf_processor = self.info.get_hf_processor(**mm_kwargs)
            image_processor = hf_processor.image_processor
            num_patches = [self.info.get_num_patches(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    patch_size=hf_processor.patch_size,
                    min_patches=image_processor.min_patches,
                    max_patches=image_processor.max_patches) for image_size in image_sizes]
            image_repl_features = [self.info._prompt_split_image(num_patches=x,
                                                 processor=hf_processor) for x in num_patches]


            tokenizer = self.info.get_tokenizer()
            image_repls_feature_tokens = [
                tokenizer.encode(image_repl, add_special_tokens=False)
                for image_repl in image_repl_features
            ]
            num_embeds = [
                len(image_repl_feature_tokens)
                for image_repl_feature_tokens in image_repls_feature_tokens
            ]
            processed_outputs["num_embeds"] = torch.tensor(num_embeds)

            hf_config = self.info.get_hf_config()

            embed_is_patch = [
                torch.tensor(image_repl_tokens) == hf_config.image_token_index
                for image_repl_tokens in image_repls_feature_tokens
            ]
            processed_outputs["embed_is_patch"] = embed_is_patch
            processed_outputs["num_patches"] = torch.tensor(num_patches)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            num_embeds=MultiModalFieldConfig.batched("image"),
            embed_is_patch=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

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
            images: ImageProcessorItems = mm_items.get("image",
                                                       ImageProcessorItems)
            image_size: ImageSize = images.get_image_size(item_idx)
            num_patches = self.info.get_num_patches(
                image_width=image_size.width,
                image_height=image_size.height,
                patch_size=hf_processor.patch_size,
                min_patches=image_processor.min_patches,
                max_patches=image_processor.max_patches)
            return self.info._prompt_split_image(num_patches=num_patches,
                                                 processor=hf_processor)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


def _get_num_hidden_layers(hf_config: AyaVisionConfig) -> int:
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest m
    elif isinstance(feature_layers, (list, tuple)):
        return max(
            _get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(f"vision_layer_feature type: {type(feature_layers)}"
                    " is not supported")


def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


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
        num_hidden_layers = _get_num_hidden_layers(config)
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        self.vision_tower = SiglipVisionModel(
            config.vision_config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers,
            prefix=maybe_prefix(prefix, "vision_model"))
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
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(['lm_head', 'rotary_emb']
                           if self.config.tie_word_embeddings else None))
        return loader.load_weights(weights)

    def _image_pixels_to_features(self, vision_tower: SiglipVisionModel,
                                  pixel_values: torch.Tensor,
                                  **kwargs) -> torch.Tensor:
        target_dtype = vision_tower.get_input_embeddings().weight.dtype
        image_features = vision_tower(pixel_values.to(dtype=target_dtype),
                                      **kwargs)

        def select_features(leaf: torch.Tensor):
            return self._select_image_features(
                leaf,
                strategy=self.config.vision_feature_select_strategy,
            )

        return cast(
            Union[torch.Tensor, tuple[torch.Tensor, ...]],
            json_map_leaves(select_features, image_features),
        )

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _process_image_input(self, image_input: AyaVisionImagePixelInputs,
                             **kwargs) -> tuple[torch.Tensor, ...]:
        assert self.vision_tower is not None
        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]
        image_features = self._image_pixels_to_features(
            self.vision_tower, pixel_values=pixel_values)
        image_embeds = self.multi_modal_projector(image_features)
        return image_embeds.split(num_patches.tolist()) if num_patches is not None else image_embeds

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

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[AyaVisionImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        num_patches = kwargs.pop("num_crops", None)
        embed_is_patch = kwargs.pop("embed_is_patch", None)
        num_embeds = kwargs.pop("num_embeds", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, "Aya Vision does not support image_embeds."

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")
        if num_patches is None:
            # num_patches got poped here:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/aya_vision/processing_aya_vision.py#L210
            num_patches =  torch.tensor([x.shape[0] for x in pixel_values if x.ndim==4])
            # TODO remove debug
            print("check num_pathes | "*10)
            print(torch.tensor([x.shape for x in pixel_values if x.ndim!=4]))
        if num_patches is not None and not isinstance(num_patches, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_patches. "
                             f"Got type: {type(num_patches)}")

        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError("Incorrect type of embed_is_patch. "
                             f"Got type: {type(embed_is_patch)}")

        if not isinstance(num_embeds, (torch.Tensor, list)):
            raise ValueError("Incorrect type of num_embeds. "
                             f"Got type: {type(num_embeds)}")

        pixel_values = flatten_bn(pixel_values, concat=True)

        return AyaVisionImagePixelInputs(
            type="pixel_values",
            pixel_values=self._validate_pixel_values(pixel_values),
            num_patches=num_patches,
            embed_is_patch=embed_is_patch,
            num_embeds=num_embeds,
        )

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        image_features = self._process_image_input(image_input, **kwargs)
        if kwargs.get("v0_path", False):
            return image_features
        return image_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)
