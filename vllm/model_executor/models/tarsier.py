# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import (Final, Literal, Optional, Protocol, TypedDict, TypeVar,
                    Union, cast)

import torch
import torch.nn as nn
from transformers import BatchFeature, CLIPVisionConfig
from transformers import LlavaConfig as HfLlavaConfig
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.models.llava import LlavaProcessor
from transformers.processing_utils import (ProcessingKwargs, Unpack,
                                           _validate_images_text_input_order)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from vllm.config import VllmConfig
from vllm.inputs import InputProcessingContext
from vllm.jsontree import json_map_leaves
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.llava import LlavaDummyInputsBuilder
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)
from .vision import VisionEncoderInfo, get_vision_encoder_info


class TarsierImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor


class TarsierImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor


TarsierImageInputs = Union[TarsierImagePixelInputs,
                           TarsierImageEmbeddingInputs]


class TarsierHfConfig(Protocol):  # Based on the Tarsier's LlavaConfig
    vision_config: Final[PretrainedConfig]
    text_config: Final[PretrainedConfig]  # Added from Tarsier's LlavaConfig
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]
    projector_hidden_act: Final[str]
    image_newline_idx: Final[int]
    image_new_idx: Final[int]
    multimodal_projector_bias: bool = True


class TarsierProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


class TarsierProcessor(LlavaProcessor):

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput],
                    list[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[TarsierProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError(
                "You have to specify at least one of `images` or `text`.")

        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            TarsierProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string,"
                             " or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            pixel_values = image_inputs["pixel_values"]
            height, width = get_image_size(to_numpy_array(pixel_values[0]))
            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size +
                1) + self.num_additional_image_tokens + 1
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token,
                                        self.image_token * num_image_tokens)
                prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors", None)
        text_inputs = self.tokenizer(prompt_strings,
                                     **output_kwargs["text_kwargs"])
        return BatchFeature(data={
            **text_inputs,
            **image_inputs
        },
                            tensor_type=return_tensors)


class TarsierMultiModalProjector(nn.Module):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str,
                 multimodal_projector_bias: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.linear_1 = ColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             bias=multimodal_projector_bias,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          bias=multimodal_projector_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class TarsierProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> TarsierHfConfig:
        return self.ctx.get_hf_config(HfLlavaConfig)

    def get_vision_encoder_info(self) -> VisionEncoderInfo:
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object) -> TarsierProcessor:
        hf_processor = self.ctx.get_hf_processor(TarsierProcessor, **kwargs)
        # Patch for patch_size if needed (copied from vLLM LLaVA)
        if hasattr(hf_processor,
                   'patch_size') and hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default":
            return encoder_num_image_tokens - 1
        if strategy == "full":
            return encoder_num_image_tokens
        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()
        num_projected_patches = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )
        if num_projected_patches <= 0:
            default_size = self.get_image_size_with_most_features()
            num_projected_patches_default = self._apply_feature_select_strategy(
                hf_config.vision_feature_select_strategy,
                vision_encoder_info.get_num_image_tokens(
                    image_width=default_size.width,
                    image_height=default_size.height,
                ),
            )
            if num_projected_patches_default <= 0:
                raise ValueError(
                    "Could not determine a valid number of image patches.")
            num_projected_patches = num_projected_patches_default
        num_height_patches = int(math.sqrt(num_projected_patches))
        total_image_tokens_for_llm = num_projected_patches \
            + num_height_patches + 1
        return total_image_tokens_for_llm

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )

    def get_image_newline_idx(self) -> int:
        return self.get_hf_config().image_newline_idx

    def get_image_new_idx(self) -> int:
        return self.get_hf_config().image_new_idx


_I_Tarsier = TypeVar("_I_Tarsier", bound=TarsierProcessingInfo)


class TarsierDummyInputsBuilder(LlavaDummyInputsBuilder[_I_Tarsier]):

    pass


class TarsierMultiModalProcessor(BaseMultiModalProcessor[_I_Tarsier]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index  # The <IMAGE> token ID

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_projected_patches = images.get_feature_size(item_idx)
                # This assumes num_projected_patches is a perfect square
                num_height_patches = int(math.sqrt(num_projected_patches))
                num_final_image_tokens = num_projected_patches \
                + num_height_patches + 1
            else:
                image_size = images.get_image_size(item_idx)
                num_final_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_final_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],  # Replace each single <IMAGE> token
                replacement=get_replacement,
            ),
        ]


def _build_tarsier_hf_info(
        ctx: InputProcessingContext) -> TarsierProcessingInfo:
    return TarsierProcessingInfo(ctx)


def _build_tarsier_hf_processor(
    info: _I_Tarsier,
    dummy_inputs: BaseDummyInputsBuilder[_I_Tarsier],
    *,
    cache: Optional[ProcessingCache] = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, TarsierProcessingInfo):
        return TarsierMultiModalProcessor(
            info,
            dummy_inputs,
            cache=cache,
        )
    raise NotImplementedError(type(info))


def init_vision_tower_for_tarsier(
    hf_config: TarsierHfConfig,  # Use the Tarsier specific config protocol
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> Union[CLIPVisionModel, SiglipVisionModel]:
    vision_config = hf_config.vision_config

    feature_layers = hf_config.vision_feature_layer
    base_num_hidden_layers = vision_config.num_hidden_layers

    def _get_layer_index(feature_layer_index: int,
                         num_hidden_layers_total: int) -> int:
        if feature_layer_index < 0:
            return num_hidden_layers_total + feature_layer_index + 1
        return feature_layer_index

    if isinstance(feature_layers, int):
        num_hidden_layers_to_init = _get_layer_index(feature_layers,
                                                     base_num_hidden_layers)
    elif isinstance(feature_layers, (list, tuple)):
        num_hidden_layers_to_init = max(
            _get_layer_index(idx, base_num_hidden_layers)
            for idx in feature_layers)
    else:
        raise TypeError(f"vision_layer_feature type: {type(feature_layers)}"
                        " is not supported")

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_to_init,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_to_init,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config for Tarsier: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_processor(_build_tarsier_hf_processor,
                                        info=_build_tarsier_hf_info,
                                        dummy_inputs=TarsierDummyInputsBuilder)
class TarsierForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: TarsierHfConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config  # Storing the Tarsier-specific HF config
        self.vision_tower = init_vision_tower_for_tarsier(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        projector_bias = getattr(config, "multimodal_projector_bias", True)

        self.multi_modal_projector = TarsierMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=projector_bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.
            text_config,  # Use text_config from Tarsier's main config
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.register_buffer('image_newline_idx_tensor',
                             torch.tensor([config.image_newline_idx],
                                          dtype=torch.long),
                             persistent=False)
        self.register_buffer('image_new_idx_tensor',
                             torch.tensor([config.image_new_idx],
                                          dtype=torch.long),
                             persistent=False)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)  # Assuming 3 channels
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")
        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[TarsierImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return TarsierImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return TarsierImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features
        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: Union[torch.Tensor, list[torch.Tensor]],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        # From vLLM LLaVA, vision tower output handling
        image_hidden_states = vision_tower(pixel_values)
        if not isinstance(image_hidden_states, torch.Tensor):
            raise TypeError(
                f"image_hidden_states type: {type(image_hidden_states)}"
                " is not supported")

        def select_features_fn(leaf: torch.Tensor):
            return self._select_image_features(
                leaf,
                strategy=self.config.vision_feature_select_strategy,
            )

        selected_features = cast(
            Union[torch.Tensor, tuple[torch.Tensor, ...]],
            json_map_leaves(select_features_fn, image_hidden_states),
        )
        return selected_features

    def _add_tarsier_split_tokens(
            self, projected_image_features: torch.Tensor) -> torch.Tensor:
        """
        Implements Tarsier's `add_split_tokens` logic.
        """
        num_images, num_projected_patches, embed_dim = \
            projected_image_features.shape
        num_height_patches = int(math.sqrt(num_projected_patches))
        num_width_patches = num_projected_patches // num_height_patches
        device = projected_image_features.device
        embedding_layer = self.language_model.model.embed_tokens
        image_newline_emb = embedding_layer(
            self.image_newline_idx_tensor.to(device)).squeeze(0)
        image_new_emb = embedding_layer(
            self.image_new_idx_tensor.to(device)).squeeze(0)
        try:
            current_image_features_grid = projected_image_features.view(
                num_images, num_height_patches, num_width_patches, embed_dim)
        except RuntimeError as e:
            raise RuntimeError(
                "Cannot reshape projected_image_features"
                f" with shape {projected_image_features.shape} "
                f"to ({num_images}, {num_height_patches},"
                f" {num_width_patches}, {embed_dim}). "
                "Ensure num_projected_patches is compatible"
                " with a grid structure. "
                f"num_projected_patches={num_projected_patches}, "
                f"derived num_height_patches={num_height_patches}. ") from e

        image_newline_expanded = image_newline_emb.expand(
            (num_images, num_height_patches, 1, embed_dim))
        features_with_newlines = torch.cat(
            [current_image_features_grid, image_newline_expanded],
            dim=2  # Concatenate along width dim
        )
        new_num_patches_after_newline = num_projected_patches \
            + num_height_patches
        features_with_newlines_flat = features_with_newlines.view(
            num_images, new_num_patches_after_newline, embed_dim)
        image_new_expanded = image_new_emb.expand((num_images, 1, embed_dim))
        final_image_features = torch.cat(
            [features_with_newlines_flat, image_new_expanded],
            dim=1  # Concatenate along patch sequence dim
        )
        return final_image_features

    def _process_image_pixels(
        self,
        inputs: TarsierImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None
        pixel_values = inputs["pixel_values"]
        image_features_selected = self._image_pixels_to_features(
            self.vision_tower, pixel_values)  # type: ignore
        if isinstance(image_features_selected, torch.Tensor):
            projected_features = self.multi_modal_projector(
                image_features_selected)
            final_features = self._add_tarsier_split_tokens(projected_features)
            return final_features
        else:
            raise TypeError(
                f"_image_pixels_to_features type:"
                f" {type(image_features_selected)} is not supported")

    def _process_image_input(
        self,
        image_input: TarsierImageInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if image_input["type"] == "image_embeds":
            projected_features = image_input["data"]
            if isinstance(projected_features, torch.Tensor):
                return self._add_tarsier_split_tokens(projected_features)
            else:
                raise ValueError("Incorrect type of image_embeds. "
                                 f"Got type: {type(projected_features)}. ")
        assert self.vision_tower is not None
        return self._process_image_pixels(image_input)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )
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
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
