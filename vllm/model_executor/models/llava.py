from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
# TODO(xwjiang): We should port CLIPVisionModel's code over to not depend on
# transformers' impl.
from transformers import CLIPVisionModel, LlavaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.sequence import SamplerOutput

from .vlm_base import VisionLanguageModelBase

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


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


def merge_vision_embeddings(input_ids: torch.Tensor,
                            inputs_embeds: torch.Tensor,
                            vision_embeddings: torch.Tensor,
                            image_token_id: int) -> torch.Tensor:
    """In place merges in vision_embeddings with inputs_embeds."""
    mask = (input_ids == image_token_id)

    image_feature_size = vision_embeddings.shape[0] * vision_embeddings.shape[1]
    if mask.sum() != image_feature_size:
        raise ValueError(f"image_feature_size should be {image_feature_size}, "
                         f"but found: {mask.sum()}")

    inputs_embeds[mask] = vision_embeddings.view(image_feature_size,
                                                 vision_embeddings.shape[-1])

    return inputs_embeds


class LlavaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channels, height, width)"""


class LlavaImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, image_feature_size, hidden_size)"""


LlavaImageInputs = Union[LlavaImagePixelInputs, LlavaImageFeatureInputs]


@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class LlavaForConditionalGeneration(VisionLanguageModelBase):

    def __init__(self,
                 config: LlavaConfig,
                 vision_language_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(vision_language_config)

        self.config = config

        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_tower = CLIPVisionModel(config.vision_config)
        else:
            self.vision_tower = None

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

    def _validate_image_data(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != list(
                self.vision_language_config.image_input_shape[1:]):
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"{self.vision_language_config.image_input_shape[1:]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_features = kwargs.pop("image_features", None)

        expected_input_type = self.vision_language_config.image_input_type
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

            return LlavaImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_data(pixel_values),
            )

        if expected_input_type == ImageInputType.IMAGE_FEATURES:
            if pixel_values is not None:
                raise ValueError(
                    "Expected image features but got pixel values")
            if image_features is None:
                return None

            if not isinstance(image_features, torch.Tensor):
                raise ValueError("Incorrect type of image features. "
                                 f"Got type: {type(image_features)}")

            return LlavaImageFeatureInputs(
                type="image_features",
                data=self._validate_image_data(image_features),
            )

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
        # TODO(xwjiang): Maybe port minimal CLIPVisionModel over.
        image_outputs = vision_tower(pixel_values.to(vision_tower.device),
                                     output_hidden_states=True)

        image_features = image_outputs.hidden_states[
            self.config.vision_feature_layer]

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
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        return self.multi_modal_projector(image_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for Llava 1.5.

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

        The model takes two types of image inputs:
        PIXEL_VALUES and IMAGE_FEATURES.
        The following shows how each maps to huggingface implementation.
        PIXEL_VALUES:
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L353
        IMAGE_FEATURES:
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L430
        before going through the multi modal projector.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: For PIXEL_VALUES, expects a batch with shape
                [1, 3, 336, 336].
            image_features: For IMAGE_FEATURES, expects a batch with shape
                [1, 576, 1024].
        """
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.vision_language_config.image_token_id)

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
