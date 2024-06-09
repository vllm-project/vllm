from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import PaliGemmaConfig, SiglipVisionModel

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gemma import GemmaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data
from vllm.sequence import SamplerOutput

from .vlm_base import VisionLanguageModelBase

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.model": "language_model",
}


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()

        self.linear = ColumnParallelLinear(vision_hidden_size,
                                           projection_dim,
                                           bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channels, height, width)"""


class PaliGemmaImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, image_feature_size, hidden_size)"""


PaliGemmaImageInputs = Union[PaliGemmaImagePixelInputs,
                             PaliGemmaImageFeatureInputs]


@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class PaliGemmaForConditionalGeneration(VisionLanguageModelBase):

    def __init__(self,
                 config: PaliGemmaConfig,
                 vision_language_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__(vision_language_config)

        self.config = config

        # TODO(ywang96): Port over SiglipVisionModel & TP
        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_tower = SiglipVisionModel(config.vision_config)
        else:
            self.vision_tower = None

        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projection_dim=config.vision_config.projection_dim)

        self.quant_config = quant_config
        self.language_model = GemmaModel(config.text_config, cache_config,
                                         quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
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
            self, **kwargs: object) -> Optional[PaliGemmaImageInputs]:
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
                raise ValueError("Incorrect type of pixel values")

            return PaliGemmaImagePixelInputs(
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
                raise ValueError("Incorrect type of image features")

            return PaliGemmaImageFeatureInputs(
                type="image_features",
                data=self._validate_image_data(image_features),
            )

        return None

    def _image_pixels_to_features(self, vision_tower: SiglipVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        image_outputs = vision_tower(pixel_values.to(vision_tower.device),
                                     output_hidden_states=True)

        selected_image_features = image_outputs.last_hidden_state

        return selected_image_features

    def _process_image_pixels(
            self, inputs: PaliGemmaImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(
            self, image_input: PaliGemmaImageInputs) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_tower is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        return self.multi_modal_projector(image_features)

    def _merge_vision_embeddings(self, input_ids: torch.Tensor,
                                 inputs_embeds: torch.Tensor,
                                 vision_embeddings: torch.Tensor,
                                 image_token_id: int) -> torch.Tensor:
        """In place merges in vision_embeddings with inputs_embeds."""

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/modeling_paligemma.py#L294 # noqa
        vision_embeddings = vision_embeddings / (self.config.hidden_size**0.5)
        mask = (input_ids == image_token_id)

        image_feature_size = vision_embeddings.shape[
            0] * vision_embeddings.shape[1]
        if mask.sum() != image_feature_size:
            raise ValueError(
                f"image_feature_size should be {image_feature_size}, "
                f"but found: {mask.sum()}")

        inputs_embeds[mask] = vision_embeddings.view(
            image_feature_size, vision_embeddings.shape[-1])

        return inputs_embeds

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                **kwargs: object) -> SamplerOutput:
        """
        The correct prompt format needs to be:
        '<image>' * image_feature_size + '<bos>' + prompt + '\n'

        See https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/paligemma/processing_paligemma.py#L55
        """ # noqa
        parsed_image_input = self._parse_and_validate_image_input(**kwargs)

        if parsed_image_input is not None:
            vision_embeddings = self._process_image_input(parsed_image_input)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)

            inputs_embeds = self._merge_vision_embeddings(
                input_ids, inputs_embeds, vision_embeddings,
                self.config.image_token_index)

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            attn_metadata,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    # Copied from vllm/model_executor/models/gemma.py
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.language_model.embed_tokens.weight,
                                       hidden_states, sampling_metadata)
        return logits

    # Copied from vllm/model_executor/models/gemma.py
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    # Adapted from vllm/model_executor/models/gemma.py
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
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
                for (param_name, shard_name,
                     shard_id) in stacked_params_mapping:
                    if shard_name not in name:
                        continue
                    name = name.replace(shard_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    # lm_head is not used in vllm as it is tied with
                    # embed_token. To prevent errors, skip loading
                    # lm_head.weight.
                    if "lm_head.weight" in name:
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # GemmaRMSNorm is different from Llama's in that it
                    # multiplies (1 + weight) to the output, instead of just
                    # weight.
                    if "norm.weight" in name:
                        loaded_weight += 1.0
                    use_default_weight_loading = True

            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}")
