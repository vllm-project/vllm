# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Final, Literal, Protocol, TypeAlias, TypeVar

import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    CLIPVisionConfig,
    LlavaConfig,
    PixtralVisionConfig,
    PretrainedConfig,
    SiglipVisionConfig,
)
from transformers.models.llava import LlavaProcessor
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .pixtral import PixtralHFEncoderInfo, PixtralHFVisionModel
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_num_selected_vision_tokens, get_vision_encoder_info


class LlavaImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height
        - w: Width

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]


class PixtralHFImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels
        - h: Height
        - w: Width

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    type: Literal["pixel_values_pixtral"] = "pixel_values_pixtral"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "c", "h", "w", dynamic_dims={"h", "w"}),
    ]


class LlavaImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ifs: Image feature size
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor, TensorShape("bn", "ifs", "hs")]


LlavaImageInputs: TypeAlias = (
    LlavaImagePixelInputs | PixtralHFImagePixelInputs | LlavaImageEmbeddingInputs
)


class LlavaMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        multimodal_projector_bias: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            text_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LlavaLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[int | list[int]]


class LlavaLikeProcessor(Protocol):
    image_token: Final[str]


class BaseLlavaProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> LlavaLikeConfig:
        return self.ctx.get_hf_config(LlavaConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> LlavaLikeProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        return get_num_selected_vision_tokens(
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
            hf_config.vision_feature_select_strategy,
        )

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


_I = TypeVar("_I", bound=BaseLlavaProcessingInfo)


class LlavaDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class LlavaProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(LlavaProcessor, **kwargs)
        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor


class BaseLlavaMultiModalProcessor(BaseMultiModalProcessor[_I]):
    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


class LlavaMultiModalProcessor(BaseLlavaMultiModalProcessor[LlavaProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


class PixtralHFProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)


class PixtralHFMultiModalProcessor(BaseMultiModalProcessor[PixtralHFProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        pixel_values = processed_outputs.get("pixel_values")
        if pixel_values is not None:
            # Avoid padding since we need the output for each image to be
            # independent of other images for the cache to work correctly
            image_sizes = processed_outputs["image_sizes"]
            assert len(pixel_values) == len(image_sizes)

            processed_outputs["pixel_values"] = [
                p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
            ]

        return processed_outputs

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
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_break_id = vocab[processor.image_break_token]
        image_token_id = hf_config.image_token_index
        image_end_id = vocab[processor.image_end_token]

        assert isinstance(hf_config.vision_config, PixtralVisionConfig)
        encoder_info = PixtralHFEncoderInfo(hf_config)

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = encoder_info.get_patch_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


def _build_llava_or_pixtral_hf_info(
    ctx: InputProcessingContext,
) -> BaseLlavaProcessingInfo:
    hf_config = ctx.get_hf_config(LlavaConfig)

    if isinstance(hf_config.vision_config, PixtralVisionConfig):
        return PixtralHFProcessingInfo(ctx)

    return LlavaProcessingInfo(ctx)


def _build_llava_or_pixtral_hf_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, PixtralHFProcessingInfo):
        return PixtralHFMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    if isinstance(info, LlavaProcessingInfo):
        return LlavaMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    raise NotImplementedError(type(info))


def _get_num_hidden_layers(hf_config: LlavaLikeConfig) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.

    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(_get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(
        f"vision_layer_feature type: {type(feature_layers)} is not supported"
    )


def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


def init_vision_tower_for_llava(
    hf_config: LlavaLikeConfig,
    quant_config: QuantizationConfig | None,
    *,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel:
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, PixtralVisionConfig):
        return PixtralHFVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_processor(
    _build_llava_or_pixtral_hf_processor,
    info=_build_llava_or_pixtral_hf_info,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class LlavaForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # NOTE: These are special cases for Pixtral-12B in the HF-format
        # https://huggingface.co/mistral-community/pixtral-12b/blob/main/config.json  # noqa
        if (
            config.text_config.architectures is None
            and config.text_config.model_type == "mistral"
        ):
            config.text_config.architectures = ["MistralForCausalLM"]
        if (
            config.projector_hidden_act is None
            and config.vision_config.hidden_act == "gelu"
        ):
            config.projector_hidden_act = "gelu"

        # TODO: Optionally initializes this for supporting embeddings.
        if multimodal_config.get_limit_per_prompt("image"):
            self.vision_tower = init_vision_tower_for_llava(
                config,
                quant_config,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LlavaMultiModalProjector(
                vision_hidden_size=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                projector_hidden_act=config.projector_hidden_act,
                multimodal_projector_bias=config.multimodal_projector_bias,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )
        else:
            self.vision_tower = None
            self.multi_modal_projector = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if self.config.vision_config.model_type == "pixtral":
                return PixtralHFImagePixelInputs(
                    type="pixel_values_pixtral",
                    pixel_values=pixel_values,
                )

            expected_h = expected_w = self.config.vision_config.image_size
            return LlavaImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        if image_embeds is not None:
            if self.config.vision_config.model_type == "pixtral":
                raise ValueError("Pixtral-HF does not support image_embeds.")

            return LlavaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_tower: CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel,
        pixel_values: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        return vision_tower(
            pixel_values,
            feature_select_strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(
        self,
        inputs: LlavaImagePixelInputs | PixtralHFImagePixelInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(
        self,
        image_input: LlavaImageInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        image_features = self._process_image_pixels(image_input)

        if isinstance(image_features, torch.Tensor):
            return self.multi_modal_projector(image_features)

        feature_sizes = [image_feature.shape[0] for image_feature in image_features]

        image_embeds = self.multi_modal_projector(torch.cat(image_features))
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for LLaVA-1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"USER: <image>\\nWhat's the content of the image?\\nASSISTANT:"`.

        Tokenizer outputs:
        `[1, 3148, 1001, 29901, 29871, 32000, 29871, 13, 5618, 29915, 29879,
        278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 3148, 1001, 29901, 29871, 32000, ..., 32000, 29871, 13, 5618,
        29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566,
        29901]`.

        We insert 575 tokens so that including the original image token in the
        input, there are a total of 576 (24 * 24) image tokens, which
        corresponds to the number of image tokens inputted to the language
        model, i.e. the number of image tokens outputted by the visual encoder.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        Info:
            [`LlavaImageInputs`][vllm.model_executor.models.llava.LlavaImageInputs]
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.vision_tower is None and self.multi_modal_projector is None:
            skip_prefixes.extend(["vision_tower.", "multi_modal_projector."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class MantisProcessingInfo(LlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        hf_config = self.get_hf_config()
        vision_info = self.get_vision_encoder_info()

        kwargs.setdefault("patch_size", vision_info.get_patch_size())
        kwargs.setdefault(
            "vision_feature_select_strategy",
            hf_config.vision_feature_select_strategy,
        )

        return self.ctx.get_hf_processor(LlavaProcessor, **kwargs)


class MantisMultiModalProcessor(LlavaMultiModalProcessor):
    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        # Assume that it doesn't depend on the image size
        num_image_tokens = self.info.get_num_image_tokens(
            image_width=-1,
            image_height=-1,
        )

        result = super().apply(
            prompt,
            mm_data,
            hf_processor_mm_kwargs,
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        mm_items = self._to_mm_items(mm_data)
        mm_item_counts = mm_items.get_all_counts()
        mm_kwargs = result["mm_kwargs"]
        mm_hashes = result["mm_hashes"]

        # We reimplement the functionality of MLlavaProcessor from
        # https://github.com/TIGER-AI-Lab/Mantis.git
        def get_replacement_mantis(item_idx: int):
            return "".join(
                [
                    f"(image {item_idx + 1}: <Image>",  # 7 tokens
                    "<image>" * num_image_tokens,
                    "</Image>)",  # 3 tokens
                ]
            )

        mantis_mm_repls = self._bind_and_group_updates(
            [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id] * num_image_tokens,
                    replacement=get_replacement_mantis,
                )
            ],
            mm_item_counts,
        )

        prompt_ids, _ = self._apply_prompt_updates(
            result["prompt_token_ids"],
            mantis_mm_repls,
        )

        orig_repls = self._get_mm_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_placeholders = self._find_mm_placeholders(prompt_ids, orig_repls)
        self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholder_ranges,
        )


# To use this model, please use
# `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'`
@MULTIMODAL_REGISTRY.register_processor(
    MantisMultiModalProcessor,
    info=MantisProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class MantisForConditionalGeneration(LlavaForConditionalGeneration):
    pass
