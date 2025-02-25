# SPDX-License-Identifier: Apache-2.0

from functools import cached_property
from typing import (Iterable, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
from transformers import (BatchFeature, Blip2Config, Blip2QFormerConfig,
                          apply_chunking_to_forward)

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptReplacementDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .blip import BlipVisionModel
from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

# We use this internally as placeholders since there is no image token
# defined on the HuggingFace repo
_IMAGE_TOKEN_ID = 50265


class Blip2ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class Blip2ImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


Blip2ImageInputs = Union[Blip2ImagePixelInputs, Blip2ImageEmbeddingInputs]


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

        return sequence_output


class Blip2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Blip2Config)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_num_image_tokens()}

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        return hf_config.num_query_tokens


class Blip2DummyInputsBuilder(BaseDummyInputsBuilder[Blip2ProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config

        max_image_size = vision_config.image_size
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=max_image_size,
                                   height=max_image_size,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="",
            mm_data=mm_data,
        )


class Blip2MultiModalProcessor(BaseMultiModalProcessor[Blip2ProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # HF processor always adds placeholders even when there's no image
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        bos_token_id = tokenizer.bos_token_id
        assert isinstance(bos_token_id, int)

        image_token_id = vocab["<image>"]
        num_image_tokens = self.info.get_num_image_tokens()
        image_tokens = [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[bos_token_id],
                replacement=PromptReplacementDetails(
                    full=image_tokens + [bos_token_id],
                    features=image_tokens,
                ),
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(Blip2MultiModalProcessor,
                                        info=Blip2ProcessingInfo,
                                        dummy_inputs=Blip2DummyInputsBuilder)
class Blip2ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_model = BlipVisionModel(config.vision_config, quant_config)

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
            self, **kwargs: object) -> Optional[Blip2ImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # Remove the N dimension until multiple images are supported.
            pixel_values = pixel_values.squeeze(1)

            return Blip2ImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            # Remove the N dimension until multiple images are supported.
            image_embeds = image_embeds.squeeze(1)

            return Blip2ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(self, vision_model: BlipVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_model(pixel_values)

        return image_features

    def _process_image_pixels(self,
                              inputs: Blip2ImagePixelInputs) -> torch.Tensor:
        assert self.vision_model is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_model, pixel_values)

    def _process_image_input(self,
                             image_input: Blip2ImageInputs) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None
        image_features = self._process_image_pixels(image_input)

        query_tokens = self.query_tokens.expand(image_features.shape[0], -1,
                                                -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
        )

        return self.language_projection(query_output)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                _IMAGE_TOKEN_ID)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:
        """Run forward pass for BLIP-2.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"Question: What's the content of the image? Answer:"`.

        Tokenizer outputs:
        `[2, 45641, 35, 653, 18, 5, 1383, 9, 5, 2274, 116, 31652, 35]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends 
        dummy tokens (denoted as `50265`), resulting in:
        `[50265, ..., 50265, 2, 45641, 35, ..., 31652, 35]`.

        We insert 32 tokens since it corresponds to the number of query
        embeddings outputted by the Q-Former and inputted to the language model.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each input image.
        
        See also:
            :class:`Blip2ImageInputs`
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
