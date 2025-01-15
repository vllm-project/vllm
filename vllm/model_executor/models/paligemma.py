from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
from torch import nn
from transformers import PaliGemmaConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip, get_max_siglip_image_tokens)
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

logger = init_logger(__name__)


class PaliGemmaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class PaliGemmaImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


PaliGemmaImageInputs = Union[PaliGemmaImagePixelInputs,
                             PaliGemmaImageEmbeddingInputs]


def get_max_paligemma_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(PaliGemmaConfig)
    vision_config = hf_config.vision_config

    return get_max_siglip_image_tokens(vision_config)


def dummy_data_for_paligemma(ctx: InputContext, seq_len: int,
                             mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(PaliGemmaConfig)
    vision_config = hf_config.vision_config
    num_images = mm_counts["image"]

    seq_data, ranges = dummy_seq_data_for_siglip(
        vision_config,
        seq_len,
        num_images,
        image_token_id=hf_config.image_token_index,
    )

    mm_data = dummy_image_for_siglip(vision_config, num_images)
    return DummyData(seq_data, mm_data, ranges)


def input_processor_for_paligemma(ctx: InputContext,
                                  inputs: DecoderOnlyInputs):

    """
    The correct prompt format needs to be:
    '<image>' * image_feature_size + '<bos>' + prompt + '\n'

    See https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/paligemma/processing_paligemma.py#L55
    """ # noqa

    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PaliGemmaConfig)

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    image_feature_size = hf_config.text_config.num_image_tokens
    image_token_str = tokenizer.decode(hf_config.image_token_index)
    bos_token = tokenizer.decode(hf_config.bos_token_id)
    image_token_str_pad = image_token_str * image_feature_size
    image_token_ids_pad = [hf_config.image_token_index] * image_feature_size

    orig_prompt = inputs.get("prompt")
    orig_prompt_ids = inputs.get("prompt_token_ids")

    if orig_prompt is not None and image_token_str in orig_prompt:
        logger.warning(
            "The image token '%s' was detected in the prompt and "
            "will be removed. Please follow the proper prompt format"
            " documented on HuggingFace.", image_token_str)
        orig_prompt = orig_prompt.replace(image_token_str, "")
        orig_prompt_ids.remove(hf_config.image_token_index)

    new_prompt = f"{image_token_str_pad}{bos_token}{orig_prompt}\n"

    # The PaliGemma 2 tokenizer does not include a starting BOS token
    if orig_prompt_ids[0] != hf_config.bos_token_id:
        orig_prompt_ids = [hf_config.bos_token_id] + orig_prompt_ids

    new_token_ids = image_token_ids_pad + orig_prompt_ids + [108]  #newline

    # NOTE: Create a defensive copy of the original inputs
    return token_inputs(prompt_token_ids=new_token_ids,
                        prompt=new_prompt,
                        multi_modal_data=multi_modal_data)


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()

        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_paligemma_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_paligemma)
@INPUT_REGISTRY.register_input_processor(input_processor_for_paligemma)
class PaliGemmaForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projection_dim=config.vision_config.projection_dim)

        self.quant_config = quant_config

        if config.text_config.model_type == "gemma":
            config.text_config.architectures = ["GemmaForCausalLM"]
        else:
            config.text_config.architectures = ["Gemma2ForCausalLM"]
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.language_model.logits_processor.scale *= logit_scale

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @property
    def sampler(self):
        return self.language_model.sampler

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
            self, **kwargs: object) -> Optional[PaliGemmaImageInputs]:
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

            return PaliGemmaImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            # Remove the N dimension until multiple images are supported.
            image_embeds = image_embeds.squeeze(1)

            return PaliGemmaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_tower: SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        target_dtype = vision_tower.get_input_embeddings().weight.dtype
        image_features = vision_tower(pixel_values.to(dtype=target_dtype))

        return image_features

    def _process_image_input(
        self,
        image_input: PaliGemmaImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        pixel_values = image_input["data"]
        image_features = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )

        return self.multi_modal_projector(image_features)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/modeling_paligemma.py#L294 # noqa
        vision_embeddings = vision_embeddings * (self.config.hidden_size**-0.5)
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
                self.config.image_token_index)
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object) -> Union[SamplerOutput, IntermediateTensors]:
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
