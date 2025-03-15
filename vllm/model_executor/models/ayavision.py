# SPDX-License-Identifier: Apache-2.0
# copied from https://github.com/huggingface/transformers/tree/main/src/transformers/models/aya_vision

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Mapping, Sequence

import torch
from PIL import Image
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
                                                    AutoModelForCausalLM)
from transformers.utils import (ModelOutput, add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                replace_return_docstrings)

from vllm.inputs import InputContext
from vllm.model_executor.model_loader.weight_utils import (AutoWeightsLoader,
                                                           WeightsMapper)
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import merge_multimodal_embeddings
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal.parse import ImageItem, ModalityData
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ayavision import AyaVisionConfig

# Define the missing constant
_CONFIG_FOR_DOC = "AyaVisionConfig"


@dataclass
class AyaVisionCausalLMOutputWithPast(ModelOutput):
    """
    Base class for AyaVision causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class AyaVisionMultiModalProjector(nn.Module):

    def __init__(self, config: AyaVisionConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.alignment_intermediate_size = getattr(
            config, "alignment_intermediate_size",
            config.text_config.hidden_size)
        self.layernorm = nn.LayerNorm(config.vision_config.hidden_size *
                                      (config.downsample_factor**2),
                                      eps=config.adapter_layer_norm_eps)

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            bias=True,
        )

        self.act = ACT2FN["silu"]  # SwiGLU uses SiLU activation
        # For SwiGLU, project down to half size since we split intermediate dim
        self.linear_2 = nn.Linear(self.alignment_intermediate_size // 2,
                                  config.text_config.hidden_size,
                                  bias=True)

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features):  # B, S, D
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(image_features.shape[0], width,
                                                height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size, width, int(height / self.downsample_factor),
            int(channels * self.downsample_factor))
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size, int(height / self.downsample_factor),
            int(width / self.downsample_factor), -1)
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features


AYA_VISION_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`AyaVisionConfig`] or [`AyaVisionVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Aya Vision Model outputting raw hidden-states without any specific head on top.",
    AYA_VISION_START_DOCSTRING,
)
class AyaVisionPreTrainedModel(PreTrainedModel):
    config_class = AyaVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AyaVisionVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of AyaVision isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        std = (self.config.initializer_range if hasattr(
            self.config, "initializer_range") else
               self.config.text_config.initializer_range)

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


AYA_VISION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`GotOcr2ImageProcessor.__call__`] for details. [`CohereProcessor`] uses
            [`GotOcr2ImageProcessor`] for processing images.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The AyaVision model which consists of a vision backbone and a language model.""",
    AYA_VISION_START_DOCSTRING,
)
class AyaVisionForConditionalGeneration(AyaVisionPreTrainedModel,
                                        GenerationMixin, SupportsMultiModal):

    def __init__(self, config: AyaVisionConfig, prefix: str = ""):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config, prefix=f"{prefix}.vision_tower")
        self.multi_modal_projector = AyaVisionMultiModalProjector(config, prefix=f"{prefix}.multi_modal_projector")

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{key}"
                for key in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

        # Define weight mapper for loading weights
        self.hf_to_vllm_mapper = WeightsMapper()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self,
                                new_num_tokens: Optional[int] = None,
                                pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        """
        Returns multimodal embeddings generated from multimodal kwargs 
        to be merged with text embeddings.
        """
        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
        
    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[Dict[str, torch.Tensor]]:
        """
        Parse and validate image input from keyword arguments.
        
        Returns:
            Dict containing the pixel_values tensor if valid, None otherwise.
        """
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return None
            
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"Expected pixel_values to be a torch.Tensor, got {type(pixel_values)}")
            
        return {
            "pixel_values": pixel_values,
            "vision_feature_layer": kwargs.get("vision_feature_layer", self.config.vision_feature_layer),
            "vision_feature_select_strategy": kwargs.get(
                "vision_feature_select_strategy", self.config.vision_feature_select_strategy
            )
        }
        
    def _process_image_input(self, image_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process image input through vision encoder and projector.
        
        Args:
            image_input: Dictionary containing pixel_values and other parameters.
            
        Returns:
            Processed image features.
        """
        # Get the patch size and other parameters
        patch_size = self.info.get_patch_size() * self.info.get_downsample_factor()
        
        # Process the image to extract patches if needed
        pixel_values = image_input["pixel_values"]
        
        # Call the existing get_image_features method
        return self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=image_input["vision_feature_layer"],
            vision_feature_select_strategy=image_input["vision_feature_select_strategy"],
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """
        Returns the input embeddings merged from the text embeddings from 
        input_ids and the multimodal embeddings.
        """
        # `get_input_embeddings` should already be implemented for the language 
        # model as one of the requirements of basic vLLM model implementation.
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids, 
                inputs_embeds=inputs_embeds, 
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index)

        return inputs_embeds

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits for the given hidden states."""
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Sample from the given logits."""
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        """Load weights from the given weights."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.FloatTensor`): Tensor containing the processed image features.
        """

        image_features = self.vision_tower(pixel_values,
                                           output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[
            vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)

        return image_features

    @add_start_docstrings_to_model_forward(AYA_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AyaVisionCausalLMOutputWithPast,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, AyaVisionCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:
            `AyaVisionCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`: A `AyaVisionCausalLMOutputWithPast` or a tuple of
            `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
            elements depending on the configuration and inputs.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if intermediate_tensors is not None:
            inputs_embeds = None
        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            if (input_ids is None):
                raise ValueError("You must specify either input_ids or inputs_embeds")
                
            # Add pixel_values to kwargs if provided as a parameter
            if pixel_values is not None:
                kwargs["pixel_values"] = pixel_values
                
            vision_embeddings = self.get_multimodal_embeddings(
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                **kwargs
            )
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            last_cache_position=last_cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        logits = outputs[0] if not return_dict else outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:,
                                                      -(logits.shape[1] -
                                                        1):].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(
                    logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(
                    labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        # Get image features for the output if pixel_values are provided
        image_hidden_states = None
        if pixel_values is not None or kwargs.get("pixel_values") is not None:
            # Create image_input dictionary
            image_input = self._parse_and_validate_image_input(
                pixel_values=pixel_values if pixel_values is not None else kwargs.get("pixel_values"),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy
            )
            if image_input is not None:
                image_hidden_states = self._process_image_input(image_input)

        return AyaVisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values
            if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
            image_hidden_states=image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.
        
        In specific circumstances we don't want to forward image inputs to the model.
        If we're in cached decoding stage, pixel values should be None because input ids 
        do not contain special image token anymore.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # Only pass pixel_values in the first forward pass (cache_position is None or [0])
        # Otherwise we need pixel values to be passed to model
        if cache_position is None or (cache_position is not None and cache_position[0] == 0):
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}  # No limit on number of images
        
    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        """Calculate the maximum number of tokens per image."""
        info = self.info
        patch_size = info.get_patch_size() * info.get_downsample_factor()
        img_size = info.get_image_size()[0]  # Assuming square images
        
        # Calculate patches per image
        img_patches_per_tile = (img_size // patch_size) ** 2
        
        # Calculate total tokens for a single image with structured representation
        # This includes start/end tokens and patch tokens
        special_tokens = info.get_special_tokens()
        special_token_count = 2  # start and end tokens
        
        # For a single image (num_patches=1)
        total_tokens = special_token_count + len(special_tokens['tile_global_token']) + img_patches_per_tile
        
        return {"image": total_tokens}


class AyaVisionProcessingInfo(BaseProcessingInfo):
    """Processing info for AyaVision models."""

    def get_image_token(self) -> str:
        """Get the image token used in the prompt."""
        return "<image>"

    def get_prompt_format(self) -> str:
        """Get the prompt format for the model."""
        return "USER: {prompt}\nASSISTANT:"

    def get_image_size(self) -> Tuple[int, int]:
        """Get the image size required by the model."""
        return (364, 364)  # Updated to match the processor's img_size

    def get_image_mean(self) -> List[float]:
        """Get the image mean for normalization."""
        return [0.48145466, 0.4578275, 0.40821073]

    def get_image_std(self) -> List[float]:
        """Get the image std for normalization."""
        return [0.26862954, 0.26130258, 0.27577711]
    
    def get_patch_size(self) -> int:
        """Get the patch size for image tokenization."""
        return 28
    
    def get_downsample_factor(self) -> int:
        """Get the downsample factor for patch size scaling."""
        return 1
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get the special tokens used for image processing."""
        return {
            "start_of_img_token": "<|START_OF_IMG|>",
            "end_of_img_token": "<|END_OF_IMG|>",
            "img_patch_token": "<|IMG_PATCH|>",
            "img_line_break_token": "<|IMG_LINE_BREAK|>",
            "tile_token": "TILE",
            "tile_global_token": "TILE_GLOBAL"
        }
    
    def _prompt_split_image(self, num_patches: int) -> str:
        """
        Create a structured string representation of image tokens.
        
        Args:
            num_patches: Number of patches in the image
            
        Returns:
            String with appropriate image tokens
        """
        special_tokens = self.get_special_tokens()
        patch_size = self.get_patch_size() * self.get_downsample_factor()
        img_size = self.get_image_size()[0]  # Assuming square images
        
        img_patches_per_tile = (img_size // patch_size) ** 2
        img_string = f"{special_tokens['start_of_img_token']}"
        
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{special_tokens['tile_token']}_{idx}" + f"{special_tokens['img_patch_token']}" * img_patches_per_tile
        
        img_string += f"{special_tokens['tile_global_token']}" + f"{special_tokens['img_patch_token']}" * img_patches_per_tile
        img_string += f"{special_tokens['end_of_img_token']}"
        
        return img_string


class AyaVisionDummyInputsBuilder(BaseDummyInputsBuilder):
    """Dummy inputs builder for AyaVision models."""

    def build_processor_inputs(self) -> ProcessorInputs:
        """Build dummy processor inputs."""
        return ProcessorInputs(
            prompt="USER: <image>\nWhat's in this image?\nASSISTANT:",
            images=[Image.new("RGB", (224, 224), color="white")],
        )

    def get_dummy_processor_inputs(self, seq_len: int, mm_counts: Mapping[str, int]) -> ProcessorInputs:
        """Build dummy processor inputs for memory profiling."""
        num_images = mm_counts.get("image", 0)
        info = self.info
        image_token = info.get_image_token()
        image_size = info.get_image_size()
        
        mm_data = {
            "image": [Image.new("RGB", image_size, color="white") for _ in range(num_images)]
        }
        
        # Create a prompt with the image token
        prompt_text = image_token * num_images
        
        return ProcessorInputs(
            prompt_text=prompt_text,
            mm_data=mm_data,
        )


def input_mapper_for_ayavision(
        ctx: InputContext, data: ModalityData[ImageItem]) -> Dict[str, Any]:
    """Input mapper for AyaVision models."""
    return {"pixel_values": data.data.pixel_values}


class AyaVisionMultiModalProcessor(BaseMultiModalProcessor):
    """Multimodal processor for AyaVision models."""

    def process_image(self, ctx: InputContext,
                      image_item: ImageItem) -> PromptUpdate:
        """
        Process an image for AyaVision models.
        
        Args:
            ctx: Input context
            image_item: Image item containing pixel values
            
        Returns:
            PromptUpdate with the image token replacement and model inputs
        """
        info = AyaVisionProcessingInfo()
        image_token = info.get_image_token()
        
        # Validate pixel values
        if image_item.pixel_values is None:
            raise ValueError("Image pixel values cannot be None")
            
        if not isinstance(image_item.pixel_values, torch.Tensor):
            raise ValueError(f"Expected pixel_values to be a torch.Tensor, got {type(image_item.pixel_values)}")

        # Replace the image token with the actual image
        return PromptUpdate(
            prompt_replacements=[(image_token, image_token)],
            model_inputs={"pixel_values": image_item.pixel_values},
        )

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )
        
    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargs) -> Sequence[PromptUpdate]:
        info = self.info
        image_token = info.get_image_token()
        tokenizer = self.info.get_tokenizer()
        
        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            
            # Determine number of patches based on image size
            # For simplicity, we're assuming 1 patch per image here
            # In a real implementation, this would depend on the image processing logic
            num_patches = 1
            
            # Generate the structured image token representation
            structured_tokens = info._prompt_split_image(num_patches)
            
            # Convert the structured tokens to token IDs
            token_ids = tokenizer.encode(structured_tokens, add_special_tokens=False)
            
            return token_ids
        
        return [
            PromptReplacement(
                modality="image",
                target=tokenizer.encode(image_token, add_special_tokens=False),
                replacement=get_replacement,
            ),
        ]


# Register the AyaVision processor with the MULTIMODAL_REGISTRY
from vllm.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_processor(
    AyaVisionMultiModalProcessor,
    info=AyaVisionProcessingInfo,
    dummy_inputs=AyaVisionDummyInputsBuilder
)
@MULTIMODAL_REGISTRY.register_input_mapper("image", input_mapper_for_ayavision)
class AyaVisionForConditionalGenerationWithProcessor(AyaVisionForConditionalGeneration):
    """AyaVision model with processor registration."""
    pass
