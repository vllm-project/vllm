from typing import (Dict, Iterable, List, Literal, Optional, Tuple, TypedDict,
                    Union)

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)
from typing_extensions import NotRequired

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.inputs import INPUT_REGISTRY, InputContext
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.image import ImagePixelData
from vllm.sequence import SamplerOutput

from .clip import (dummy_feature_data_for_clip, dummy_pixel_data_for_clip,
                   dummy_seq_data_for_clip, get_clip_patch_grid_length)
from .interfaces import SupportsVision
from .llava import LlavaMultiModalProjector, merge_vision_embeddings

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, num_channels, height, width)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""


class LlavaNextImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, 1 + num_patches, image_feature_size, hidden_size)"""

    image_sizes: NotRequired[torch.Tensor]
    """Shape: (batch_size, 2)"""


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageFeatureInputs]


def _get_llava_next_num_unpadded_features(
    height: int,
    width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    # Taken from: https://github.com/huggingface/text-generation-inference/blob/799a193b109662743bed1b18a09af1fdcd508c8b/server/text_generation_server/models/vlm_causal_lm.py#L111
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio: float = width / height
    current_aspect_ratio: float = current_width / current_height
    if aspect_ratio > current_aspect_ratio:
        new_height = (height * current_width) // width
        current_height = new_height
    else:
        new_width = (width * current_height) // height
        current_width = new_width

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def _get_llava_next_image_feature_size(
    hf_config: LlavaNextConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = get_clip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = num_patches * num_patches

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(input_height, input_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_config.image_size,
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = _get_llava_next_num_unpadded_features(input_height, input_width,
                                                  num_patches,
                                                  num_patch_height,
                                                  num_patch_width)

        return unpadded_feature_size + newline_feature_size + base_feature_size

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def dummy_data_for_llava_next(ctx: InputContext, seq_len: int):
    multimodal_config = ctx.get_multimodal_config()
    hf_config = ctx.get_hf_config(LlavaNextConfig)
    vision_config = hf_config.vision_config

    #TODO: change the logic for dummy data to support dynamic shape
    _, _, dummy_height, dummy_width = multimodal_config.image_input_shape
    image_feature_size = _get_llava_next_image_feature_size(
        hf_config, input_height=dummy_height, input_width=dummy_width)

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        image_input_type = multimodal_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType
        mm_data: MultiModalData
        if image_input_type == ImageInputType.PIXEL_VALUES:
            mm_data = dummy_pixel_data_for_clip(
                vision_config,
                image_width_override=dummy_width,
                image_height_override=dummy_height,
            )
        elif image_input_type == ImageInputType.IMAGE_FEATURES:
            mm_data = dummy_feature_data_for_clip(
                vision_config,
                image_feature_size_override=image_feature_size,
            )

        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def _pixel_mapper(ctx: InputContext,
                  data: ImagePixelData) -> Dict[str, torch.Tensor]:
    image = data.image

    if isinstance(image, torch.Tensor):
        pixel_values = image.to(ctx.model_config.dtype)
        batch_size, _, _, h, w = pixel_values.shape
        image_sizes = torch.tensor([(w, h) for _ in range(batch_size)])

        return {"pixel_values": pixel_values, "image_sizes": image_sizes}

    # Temporary patch before dynamic number of image tokens is supported
    _, _, h, w = ctx.get_multimodal_config().image_input_shape
    if (w, h) != (image.width, image.height):
        logger.warning(
            "Dynamic image shape is currently not supported. "
            "Resizing input image to (%d, %d).", w, h)

        data.image = image.resize((w, h))

    return MULTIMODAL_REGISTRY._get_plugin_for_data_type(ImagePixelData) \
        ._default_input_mapper(ctx, data)


@MULTIMODAL_REGISTRY.register_image_feature_input_mapper()
@MULTIMODAL_REGISTRY.register_image_pixel_input_mapper(_pixel_mapper)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava_next)
class LlavaNextForConditionalGeneration(nn.Module, SupportsVision):

    def __init__(self,
                 config: LlavaNextConfig,
                 vlm_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.vlm_config = vlm_config

        if self.vlm_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_tower = CLIPVisionModel(config=config.vision_config)
        else:
            raise TypeError("Image features are not supported by LLaVA-NeXT")

        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)

        self.quant_config = quant_config
        self.language_model = LlamaModel(config.text_config, cache_config,
                                         quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))

    def _validate_image_pixels(self, data: torch.Tensor) -> torch.Tensor:
        _, num_channels, _, _ = self.vlm_config.image_input_shape

        # Note that this is different from that of vLLM vision_language_config
        # since the image is resized by the HuggingFace preprocessor
        height = width = self.config.vision_config.image_size

        if list(data.shape[2:]) != [num_channels, height, width]:
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"num_patches plus {[num_channels, height, width]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != [2]:
            raise ValueError(
                f"The expected image sizes shape is batch dimension plus "
                f"{[2]}. You supplied {data.shape}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_features = kwargs.pop("image_features", None)

        expected_input_type = self.vlm_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if expected_input_type == ImageInputType.PIXEL_VALUES:
            if image_features is not None:
                raise ValueError(
                    "Expected pixel values but got image features")
            if pixel_values is None:
                return None

            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, torch.Tensor):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaNextImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixels(pixel_values),
                image_sizes=self._validate_image_sizes(image_sizes),
            )

        assert expected_input_type != ImageInputType.IMAGE_FEATURES, (
            "Failed to validate this at initialization time")

        return None

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(self, vision_tower: CLIPVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values.to(vision_tower.device),
                                      self.config.vision_feature_layer)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            orig_width, orig_height = image_size
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # image_aspect_ratio == "anyres"
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    (orig_width, orig_height),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                other_patch_embeds = other_patch_embeds \
                    .view(num_patch_width, num_patch_height, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     image_size)
                    other_patch_embeds = torch.cat((
                        other_patch_embeds,
                        self.image_newline[:, None, None] \
                            .expand(*other_patch_embeds.shape[:-1], 1) \
                            .to(other_patch_embeds.device),
                    ), dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
            self, inputs: LlavaNextImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        b, num_patches, c, h, w = pixel_values.shape
        stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)

        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)

        return stacked_image_features.view(b, num_patches,
                                           *stacked_image_features.shape[-2:])

    def _process_image_input(
            self, image_input: LlavaNextImageInputs) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        patch_embeddings = self.multi_modal_projector(image_features)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = image_input["data"].shape[0]
            vision_config = self.config.vision_config
            default_width = default_height = vision_config.image_size
            image_sizes = torch.as_tensor([[default_width, default_height]
                                           for _ in range(batch_size)])

        merged_patch_embeddings = [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features,
                                               strategy="spatial_unpad")
            for i, patch_features in enumerate(patch_embeddings)
        ]

        return torch.stack(merged_patch_embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for LlaVA-NeXT.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.
        Concretely, consider a text prompt:
        "<image>\nUSER: What's the content of the image?\nASSISTANT:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted image has a size of 576 (24 * 24) along the context
        length dimension.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        There will be 576 `32000` in the `input_ids`.
        (32000 is the token id for `<image>`.)

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each grid patch for each input image.
                Expects a batch with shape `[1, num_patches, 3, 336, 336]`.
            image_sizes: The original `(width, height)` for each input image.
                Expects a batch with shape `[1, 2]`.

        See also:
            Each input maps to huggingface implementation, as follows:

            - `pixel_values`: https://github.com/huggingface/transformers/blob/v4.41.1/src/transformers/models/llava_next/modeling_llava_next.py#L690
            - `image_sizes`: https://github.com/huggingface/transformers/blob/v4.41.1/src/transformers/models/llava_next/modeling_llava_next.py#L691
        """
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.vlm_config.image_token_id)

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # post_layernorm is not needed in CLIPVisionModel
            if "vision_model.post_layernorm" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
