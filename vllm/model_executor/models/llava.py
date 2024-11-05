from functools import cached_property
from typing import (Iterable, List, Literal, Mapping, Optional, Protocol,
                    Tuple, TypedDict, Union)

import torch
import torch.nn as nn
from PIL import Image
from transformers import (CLIPVisionConfig, LlavaConfig, PixtralVisionConfig,
                          PretrainedConfig, SiglipVisionConfig)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import (CLIPVisionModel, dummy_image_for_clip,
                   dummy_seq_data_for_clip, get_max_clip_image_tokens,
                   input_processor_for_clip)
from .interfaces import SupportsMultiModal, SupportsPP
from .pixtral import (PixtralHFVisionModel, dummy_image_for_pixtral_hf,
                      dummy_seq_data_for_pixtral_hf,
                      get_max_pixtral_hf_image_tokens,
                      input_processor_for_pixtral_hf)
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip, get_max_siglip_image_tokens,
                     input_processor_for_siglip)
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    merge_multimodal_embeddings)


class LlavaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, height, width)`

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """


class LlavaImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaImageInputs = Union[LlavaImagePixelInputs, LlavaImageEmbeddingInputs]


# TODO(xwjiang): Run benchmark and decide if TP.
class LlavaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def get_max_llava_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(LlavaConfig)
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_image_tokens = get_max_clip_image_tokens(vision_config)
    elif isinstance(vision_config, SiglipVisionConfig):
        num_image_tokens = get_max_siglip_image_tokens(vision_config)
    elif isinstance(vision_config, PixtralVisionConfig):
        num_image_tokens = get_max_pixtral_hf_image_tokens(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    strategy = hf_config.vision_feature_select_strategy
    if strategy == "default":
        return num_image_tokens - 1
    elif strategy == "full":
        return num_image_tokens
    else:
        raise ValueError(f"Unexpected select feature strategy: {strategy}")


def dummy_data_for_llava(ctx: InputContext, seq_len: int,
                         mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(LlavaConfig)
    vision_config = hf_config.vision_config
    num_images = mm_counts["image"]

    image_feature_size = get_max_llava_image_tokens(ctx)

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data, ranges = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            num_images,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_clip(vision_config, num_images)
        return DummyData(seq_data, mm_data, ranges)
    elif isinstance(vision_config, SiglipVisionConfig):
        seq_data, ranges = dummy_seq_data_for_siglip(
            vision_config,
            seq_len,
            num_images,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_siglip(vision_config, num_images)
        return DummyData(seq_data, mm_data, ranges)
    elif isinstance(vision_config, PixtralVisionConfig):
        seq_data, ranges = dummy_seq_data_for_pixtral_hf(
            vision_config,
            seq_len,
            num_images,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_pixtral_hf(vision_config, num_images)
        return DummyData(seq_data, mm_data, ranges)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def input_processor_for_llava(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaConfig)
    vision_config = hf_config.vision_config

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        image_feature_size = get_max_llava_image_tokens(ctx)
    elif is_list_of(image_data, Image.Image):
        image_feature_size = [get_max_llava_image_tokens(ctx)
                              ] * len(image_data)
    elif isinstance(image_data, torch.Tensor):
        num_images, image_feature_size, hidden_size = image_data.shape
    elif is_list_of(image_data, torch.Tensor):
        image_feature_size = [item.shape[1] for item in image_data]
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    if isinstance(vision_config, CLIPVisionConfig):
        return input_processor_for_clip(
            model_config,
            vision_config,
            inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return input_processor_for_siglip(
            model_config,
            vision_config,
            inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )
    elif isinstance(vision_config, PixtralVisionConfig):
        # We ignore image_feature_size_override since we have non-uniform
        # image sizes for Pixtral
        return input_processor_for_pixtral_hf(
            model_config,
            vision_config,
            inputs,
            image_token_id=hf_config.image_token_index,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


class LlavaLikeConfig(Protocol):
    vision_config: PretrainedConfig
    vision_feature_layer: int


def init_vision_tower_for_llava(
    hf_config: LlavaLikeConfig,
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
):
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the required feature layer
    vision_feature_layer = hf_config.vision_feature_layer
    if vision_feature_layer < 0:
        num_hidden_layers = hf_config.vision_config.num_hidden_layers \
            + vision_feature_layer + 1
    else:
        num_hidden_layers = vision_feature_layer + 1

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


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llava_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava)
class LlavaForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self,
                 config: LlavaConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

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
            prefix="vision_tower")
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)

        self.language_model = init_vllm_registered_model(
            config.text_config,
            cache_config,
            quant_config,
            prefix="language_model")

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

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

    def _validate_image_sizes(self, images: List[torch.Tensor],
                              sizes: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(sizes, list):
            sizes = [sizes]

        total_images = sum(size.numel() // 2 for size in sizes)
        if total_images != len(images):
            raise ValueError("Mismatch in number of images. "
                             f"Expected {total_images}, got {len(images)}")
        img_idx = 0
        for size in sizes:
            # Flatten the size tensor to a list of (height, width) pairs
            size = size.view(-1, 2).tolist()
            for expected_h, expected_w in size:
                if img_idx >= len(images):
                    raise ValueError("Ran out of images before sizes. "
                                     f"{img_idx} >= {len(images)}")
                img = images[img_idx]
                if img.shape[-2:] != (expected_h, expected_w):
                    raise ValueError(
                        "Image size mismatch. Expected "
                        f"{(expected_h, expected_w)}, got {img.shape[-2:]}")
                if img.shape[-3] != 3:
                    raise ValueError("Image channel mismatch. Expected 3, "
                                     f"got {img.shape[-3]}")
                img_idx += 1
        return images

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # Case for models like PixtralHF that have dynamic image sizes
            # so we need to produce a list of tensors
            if image_sizes is not None:
                images = pixel_values

                def flatten_to_3d_tensors(item):
                    if isinstance(item, torch.Tensor):
                        if item.dim() >= 3:
                            return [t for t in item.view(-1, *item.shape[-3:])]
                        else:
                            raise ValueError(
                                f"Unexpected tensor dimension: {item.dim()}")
                    elif isinstance(item, list):
                        return [
                            t for subitem in item
                            for t in flatten_to_3d_tensors(subitem)
                        ]
                    else:
                        raise ValueError(f"Unexpected type: {type(item)}")

                # Restructure the batched images into a list of lists of images
                images = flatten_to_3d_tensors(pixel_values)

                return LlavaImagePixelInputs(
                    type="pixel_values",
                    data=self._validate_image_sizes(images, image_sizes),
                )

            return LlavaImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel,
                            PixtralHFVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(self,
                              inputs: LlavaImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(self,
                             image_input: LlavaImageInputs) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        image_features = self._process_image_pixels(image_input)
        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
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
            pixel_values: The pixels in each input image.

        See also:
            :class:`LlavaImageInputs`
        """
        if intermediate_tensors is not None:
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is not None:
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)

                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.config.image_token_index)
            else:
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    input_ids)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
