from typing import Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, LlavaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, SamplerOutput

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   input_processor_for_clip)
from .interfaces import SupportsVision
from .utils import merge_vision_embeddings

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


class LlavaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""


LlavaImageInputs = LlavaImagePixelInputs


def dummy_data_for_llava(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(LlavaConfig)
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        seq_data = dummy_seq_data_for_clip(
            vision_config,
            seq_len,
            image_token_id=hf_config.image_token_index,
        )

        mm_data = dummy_image_for_clip(vision_config)
        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def input_processor_for_llava(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(LlavaConfig)
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        return input_processor_for_clip(
            model_config,
            vision_config,
            llm_inputs,
            image_token_id=hf_config.image_token_index,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava)
class LlavaForConditionalGeneration(nn.Module, SupportsVision):

    def __init__(self,
                 config: LlavaConfig,
                 vlm_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.vlm_config = vlm_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_tower = CLIPVisionModel(config.vision_config)
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
            org_num_embeddings=self.language_model.org_vocab_size,
            quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def _validate_image_data(self, data: torch.Tensor) -> torch.Tensor:
        if list(data.shape[1:]) != list(self.vlm_config.image_input_shape[1:]):
            raise ValueError(
                f"The expected image tensor shape is batch dimension plus "
                f"{self.vlm_config.image_input_shape[1:]}. "
                f"You supplied {data.shape}. "
                f"If you are using vLLM's entrypoint, make sure your "
                f"supplied image input is consistent with "
                f"image_input_shape in engine args.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(pixel_values)}")

        return LlavaImagePixelInputs(
            type="pixel_values",
            data=self._validate_image_data(pixel_values),
        )

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
        image_features = vision_tower(pixel_values,
                                      self.config.vision_feature_layer)

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
    ) -> SamplerOutput:
        """Run forward pass for LLaVA-1.5.

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
            pixel_values: The pixels in each input image.
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
                                            None,
                                            inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
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
