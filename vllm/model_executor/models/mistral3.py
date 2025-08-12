# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import (Final, Literal, Optional, Protocol, Set, Tuple, TypedDict,
                    TypeVar, Union)

import torch
import torch.nn as nn
from transformers import (BatchFeature, Mistral3Config, PixtralVisionConfig,
                          PretrainedConfig)
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig
from vllm.inputs import InputProcessingContext
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .pixtral import PixtralHFEncoderInfo, PixtralHFVisionModel
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)
from .vision import (get_vision_encoder_info, scatter_patch_features,
                     select_patch_features)


class Mistral3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values_pixtral"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    embed_is_patch: Union[torch.Tensor, list[torch.Tensor]]
    """
    A boolean mask indicating which image embeddings correspond
    to patch tokens.
    
    Shape: `(batch_size, num_images, num_embeds)`
    """


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, vision_hidden_size: int, spatial_merge_size: int,
                 patch_size: int):
        super().__init__()

        self.vision_hidden_size = vision_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.merging_layer = nn.Linear(vision_hidden_size *
                                       self.spatial_merge_size**2,
                                       vision_hidden_size,
                                       bias=False)

    def forward(self, image_features: torch.Tensor,
                image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [(image_size[0] // self.patch_size,
                        image_size[1] // self.patch_size)
                       for image_size in image_sizes]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
                image_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0,
                                                            1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size)
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 spatial_merge_size: int,
                 patch_size: int,
                 projector_hidden_act: str,
                 multimodal_projector_bias: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.norm = RMSNorm(vision_hidden_size, eps=1e-5)
        self.patch_merger = Mistral3PatchMerger(
            vision_hidden_size=vision_hidden_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size)

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

    def forward(self, image_features: torch.Tensor,
                image_sizes: torch.Tensor) -> torch.Tensor:
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LlavaLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]


class LlavaLikeProcessor(Protocol):
    image_token: Final[str]


class BaseLlavaProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> LlavaLikeConfig:
        return self.ctx.get_hf_config(Mistral3Config)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> LlavaLikeProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        vision_encoder_info = self.get_vision_encoder_info()
        return vision_encoder_info.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
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


class Mistral3DummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token
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


class Mistral3ProcessingInfo(BaseLlavaProcessingInfo):

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)


class Mistral3MultiModalProcessor(
        BaseMultiModalProcessor[Mistral3ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
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

            hf_config = self.info.get_hf_config()
            vision_config = hf_config.vision_config
            assert isinstance(vision_config, PixtralVisionConfig)
            encoder_info = PixtralHFEncoderInfo(vision_config)

            tile_sizes = [
                encoder_info.get_patch_grid_size(
                    image_width=pixel_value.shape[-1],
                    image_height=pixel_value.shape[-2],
                ) for pixel_value in processed_outputs["pixel_values"]
            ]
            embed_is_patch = [
                torch.tensor(([True] * ncols + [False]) * nrows)
                for ncols, nrows in tile_sizes
            ]
            processed_outputs["embed_is_patch"] = embed_is_patch

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            embed_is_patch=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_break_id = vocab[processor.image_break_token]
        image_token_id = hf_config.image_token_index
        image_end_id = vocab[processor.image_end_token]

        vision_config = hf_config.vision_config
        assert isinstance(vision_config, PixtralVisionConfig)
        encoder_info = PixtralHFEncoderInfo(vision_config)

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = encoder_info.get_patch_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


def _build_mistral3_info(
    ctx: InputProcessingContext, ) -> BaseLlavaProcessingInfo:
    hf_config = ctx.get_hf_config(Mistral3Config)
    assert isinstance(hf_config.vision_config, PixtralVisionConfig)
    return Mistral3ProcessingInfo(ctx)


def _build_mistral3_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: Optional[ProcessingCache] = None,
    enable_sanity_checks: bool = True,
) -> BaseMultiModalProcessor:
    assert isinstance(info, Mistral3ProcessingInfo)
    return Mistral3MultiModalProcessor(
        info,
        dummy_inputs,  # type: ignore
        cache=cache,
        enable_sanity_checks=enable_sanity_checks,
    )


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
        return max(
            _get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(f"vision_layer_feature type: {type(feature_layers)}"
                    " is not supported")


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
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
) -> PixtralHFVisionModel:
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    assert isinstance(vision_config, PixtralVisionConfig)

    return PixtralHFVisionModel(
        vision_config,
        quant_config=quant_config,
        num_hidden_layers_override=num_hidden_layers,
        require_post_norm=require_post_norm,
        prefix=prefix,
    )


# TODO(mgoin): Support V1, there are issues with image batching/chunking
# that need to be resolved first.
@MULTIMODAL_REGISTRY.register_processor(
    _build_mistral3_processor,
    info=_build_mistral3_info,
    dummy_inputs=Mistral3DummyInputsBuilder)
class Mistral3ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                       SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # NOTE: These are special cases for Pixtral-12B in the HF-format
        # https://huggingface.co/mistral-community/pixtral-12b/blob/main/config.json  # noqa
        if (config.text_config.architectures is None
                and config.text_config.model_type == "mistral"):
            config.text_config.architectures = ["MistralForCausalLM"]
        if (config.projector_hidden_act is None
                and config.vision_config.hidden_act == "gelu"):
            config.projector_hidden_act = "gelu"

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.multi_modal_projector = Mistral3MultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            spatial_merge_size=config.spatial_merge_size,
            patch_size=config.vision_config.patch_size,
            multimodal_projector_bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Mistral3ImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        assert pixel_values is not None
        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        assert self.config.vision_config.model_type == "pixtral"
        embed_is_patch = kwargs.pop("embed_is_patch")
        if not isinstance(embed_is_patch, (torch.Tensor, list)):
            raise ValueError("Incorrect type of embed_is_patch. "
                             f"Got type: {type(embed_is_patch)}")

        return Mistral3ImagePixelInputs(
            type="pixel_values_pixtral",
            pixel_values=flatten_bn(pixel_values),
            embed_is_patch=flatten_bn(embed_is_patch),
        )

    def _process_image_input(
        self,
        image_input: Mistral3ImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_sizes = [(img.shape[-2], img.shape[-1])
                       for img in image_input["pixel_values"]]

        image_features = self.vision_tower(image_input["pixel_values"])

        if isinstance(image_features, torch.Tensor):
            return self.multi_modal_projector(image_features, image_sizes)

        feature_sizes = [
            image_feature.shape[0] // self.config.spatial_merge_size**2
            for image_feature in image_features
        ]

        image_embeds = self.multi_modal_projector(torch.cat(image_features),
                                                  image_sizes)
        if len(feature_sizes) > 1:
            image_embeds = torch.split(image_embeds, feature_sizes)
        else:
            image_embeds = (image_embeds, )
        return image_embeds

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)

        return scatter_patch_features(
            vision_embeddings,
            image_input["embed_is_patch"],
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                select_patch_features(multimodal_embeddings),
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
        """Run forward pass for Mistral3.

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
            pixel_values: The pixels in each input image.

        See also:
            :class:`Mistral3ImagePixelInputs`
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

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
