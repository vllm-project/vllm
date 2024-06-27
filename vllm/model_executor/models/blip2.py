from typing import Iterable, List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
# TODO: Port over Blip2VisionModel to vLLM
from transformers import (Blip2Config, Blip2QFormerConfig, Blip2VisionModel,
                          apply_chunking_to_forward)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VisionLanguageConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.opt import OPTModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalData
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
from vllm.sequence import SamplerOutput, SequenceData

from .clip import dummy_feature_data_for_clip, dummy_pixel_data_for_clip
from .interfaces import SupportsVision
from .utils import merge_vision_embeddings

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


class Blip2QFormerMultiHeadAttention(nn.Module):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of "
                f"the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = (config.hidden_size //
                                    config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            kv_hidden_size = config.encoder_hidden_size
        else:
            kv_hidden_size = config.hidden_size
        self.key = nn.Linear(kv_hidden_size, self.all_head_size)
        self.value = nn.Linear(kv_hidden_size, self.all_head_size)

        self.position_embedding_type = getattr(config,
                                               "position_embedding_type",
                                               "absolute")
        if self.position_embedding_type != "absolute":
            raise NotImplementedError("Unsupported position_embedding_type: "
                                      f"{self.position_embedding_type}")

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x = x.view(*x.size()[:-1], self.num_attention_heads,
                   self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_probs = torch.softmax(attention_scores * self.scaling,
                                        dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2],
                                           self.all_head_size)

        return context_layer


class Blip2QFormerSelfOutput(nn.Module):

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerAttention(nn.Module):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.attention = Blip2QFormerMultiHeadAttention(
            config,
            quant_config=quant_config,
            cache_config=cache_config,
            is_cross_attention=is_cross_attention,
        )

        self.output = Blip2QFormerSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_output = self.attention(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class Blip2QFormerIntermediate(nn.Module):

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Blip2QFormerOutput(nn.Module):

    def __init__(self, config: Blip2QFormerConfig) -> None:
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Blip2QFormerLayer(nn.Module):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config,
                                               quant_config=quant_config,
                                               cache_config=cache_config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = Blip2QFormerAttention(
                config,
                quant_config=quant_config,
                cache_config=cache_config,
                is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        query_length: int,
    ):
        attention_output = self.attention(hidden_states)

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                query_attention_output = self.crossattention(
                    query_attention_output,
                    encoder_hidden_states=encoder_hidden_states,
                )

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text],
                                         dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )

        return layer_output

    def feed_forward_chunk(self,
                           attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(
            self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class Blip2QFormerEncoder(nn.Module):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
    ) -> None:
        super().__init__()

        self.config = config

        self.layer = nn.ModuleList([
            Blip2QFormerLayer(config,
                              quant_config=quant_config,
                              cache_config=cache_config,
                              layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        query_length: int,
    ) -> torch.Tensor:
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]

            hidden_states = layer_module(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                query_length=query_length,
            )

        return hidden_states


# Adapted from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/blip_2/modeling_blip_2.py#L1025
class Blip2QFormerModel(nn.Module):

    def __init__(
        self,
        config: Blip2QFormerConfig,
        *,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
    ) -> None:
        super().__init__()

        self.config = config

        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(config,
                                           quant_config=quant_config,
                                           cache_config=cache_config)

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> torch.Tensor:
        query_length = query_embeds.shape[1]

        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        sequence_output = self.encoder(
            embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            query_length=query_length,
        )
        # pooled_output = sequence_output[:, 0, :]

        return sequence_output


class Blip2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: (batch_size, num_channels, height, width)"""


class Blip2ImageFeatureInputs(TypedDict):
    type: Literal["image_features"]
    data: torch.Tensor
    """Shape: (batch_size, image_feature_size, hidden_size)"""


Blip2ImageInputs = Union[Blip2ImagePixelInputs, Blip2ImageFeatureInputs]

# Used internally as placeholders
BLIP2_IMAGE_TOKEN = "<image>"
BLIP2_IMAGE_TOKEN_ID = 50265


def get_blip2_image_feature_size(hf_config: Blip2Config) -> int:
    return hf_config.num_query_tokens


def dummy_data_for_blip2(ctx: InputContext, seq_len: int):
    multimodal_config = ctx.get_multimodal_config()
    hf_config = ctx.get_hf_config(Blip2Config)
    vision_config = hf_config.vision_config

    image_feature_size = get_blip2_image_feature_size(hf_config)
    token_ids = [BLIP2_IMAGE_TOKEN_ID] * image_feature_size
    token_ids += [0] * (seq_len - image_feature_size)
    seq_data = SequenceData(token_ids)

    image_input_type = multimodal_config.image_input_type
    ImageInputType = VisionLanguageConfig.ImageInputType
    mm_data: MultiModalData
    if image_input_type == ImageInputType.PIXEL_VALUES:
        mm_data = dummy_pixel_data_for_clip(vision_config)
    elif image_input_type == ImageInputType.IMAGE_FEATURES:
        mm_data = dummy_feature_data_for_clip(vision_config)

    return seq_data, mm_data


def input_processor_for_blip2(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or not isinstance(
            multi_modal_data, (ImagePixelData, ImageFeatureData)):
        return llm_inputs

    hf_config = ctx.get_hf_config(Blip2Config)
    image_feature_size = get_blip2_image_feature_size(hf_config)

    # The original model places image tokens at the front
    # https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/blip_2/modeling_blip_2.py#L1514
    new_token_ids = [BLIP2_IMAGE_TOKEN_ID] * image_feature_size
    new_token_ids += llm_inputs["prompt_token_ids"]

    new_prompt = llm_inputs.get("prompt")
    if new_prompt is not None:
        new_prompt = BLIP2_IMAGE_TOKEN * image_feature_size + new_prompt

    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


@MULTIMODAL_REGISTRY.register_image_feature_input_mapper()
@MULTIMODAL_REGISTRY.register_image_pixel_input_mapper()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_blip2)
@INPUT_REGISTRY.register_input_processor(input_processor_for_blip2)
class Blip2ForConditionalGeneration(nn.Module, SupportsVision):

    def __init__(self,
                 config: Blip2Config,
                 vlm_config: VisionLanguageConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:

        super().__init__()

        self.config = config
        self.vlm_config = vlm_config

        if self.vlm_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_model = Blip2VisionModel(config.vision_config)
        else:
            self.vision_model = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens,
                        config.qformer_config.hidden_size))

        self.qformer = Blip2QFormerModel(config.qformer_config,
                                         cache_config=cache_config,
                                         quant_config=quant_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )

        self.quant_config = quant_config

        self.language_model = OPTModel(config.text_config, cache_config,
                                       quant_config)

        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(self.unpadded_vocab_size,
                                      config.text_config.hidden_size)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
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
            self, **kwargs: object) -> Optional[Blip2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
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

            return Blip2ImagePixelInputs(
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

            return Blip2ImageFeatureInputs(
                type="image_features",
                data=self._validate_image_data(image_features),
            )

        return None

    def _image_pixels_to_features(self, vision_model: Blip2VisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        # TODO: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        vision_outputs = vision_model(pixel_values.to(vision_model.device))
        image_features = vision_outputs[0]

        return image_features

    def _process_image_pixels(self,
                              inputs: Blip2ImagePixelInputs) -> torch.Tensor:
        assert self.vision_model is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_model, pixel_values)

    def _process_image_input(self,
                             image_input: Blip2ImageInputs) -> torch.Tensor:
        if image_input["type"] == "pixel_values":
            assert self.vision_model is not None
            image_features = self._process_image_pixels(image_input)
        else:
            image_features = image_input["data"]

        query_tokens = self.query_tokens.expand(image_features.shape[0], -1,
                                                -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
        )

        return self.language_projection(query_output)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for BLIP-2.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.
        Concretely, consider a text prompt:
        "<image>Question: What's the content of the image? Answer:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted image has a size of 32 (query tokens) along the
        context length dimension.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        There will be 32 `32000` in the `input_ids`.
        (32000 is the token id for `<image>`.)

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        This model has two modes of image inputs:
        `PIXEL_VALUES` and `IMAGE_FEATURES`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each input image.
                Expects a batch with shape `[1, 3, 224, 224]`.
                (Only applicable to `PIXEL_VALUES` mode)
            image_features: The image features for each input image outputted by
                the vision tower before passing to the Q-Former.
                Expects a batch with shape `[1, 32, 2560]`.
                (Only applicable to `IMAGE_FEATURES` mode)

        See also:
            Each input maps to huggingface implementation, as follows:

            - `pixel_values`: https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/blip_2/modeling_blip_2.py#L1445
            - `image_features`: https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/blip_2/modeling_blip_2.py#L1492
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
                if self.vision_model is not None:
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
