from functools import cached_property
from types import MethodType
from typing import (Iterable, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, TypedDict, Union)

import torch
import torch.nn as nn
from transformers import (BatchFeature, CLIPVisionConfig, LlavaConfig,
                          PixtralVisionConfig, PretrainedConfig,
                          ProcessorMixin, SiglipVisionConfig)
from transformers.models.llava import LlavaProcessor
from transformers.models.pixtral import PixtralProcessor

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors

from .clip import (CLIPVisionModel, dummy_image_for_clip,
                   get_max_clip_image_tokens)
from .interfaces import SupportsMultiModal, SupportsPP
from .pixtral import (PixtralHFVisionModel, dummy_image_for_pixtral_hf,
                      get_max_pixtral_hf_image_tokens,
                      get_pixtral_hf_image_feature_size)
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     get_max_siglip_image_tokens)
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)


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


class LlavaMultiModalProjector(nn.Module):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.linear_1 = ColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             bias=True,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
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


class LlavaMultiModalProcessor(BaseMultiModalProcessor):

    def _patch_pixtral_processor(self, hf_processor: PixtralProcessor):
        if getattr(hf_processor, "__is_patched__", False):
            return  # Already patched

        image_processor = hf_processor.image_processor  # type: ignore
        orig_preprocess = image_processor.preprocess

        def preprocess(__self, *args, **kwargs):
            hf_inputs = orig_preprocess(*args, **kwargs)
            hf_inputs["is_pixtral"] = torch.tensor(True)
            return hf_inputs

        image_processor.preprocess = MethodType(preprocess, image_processor)

        hf_processor.__is_patched__ = True  # type: ignore

    def _get_hf_processor(self) -> Union[LlavaProcessor, PixtralProcessor]:
        hf_processor = self.ctx.get_hf_processor()
        assert isinstance(hf_processor, (LlavaProcessor, PixtralProcessor))

        if isinstance(hf_processor, PixtralProcessor):
            self._patch_pixtral_processor(hf_processor)

        return hf_processor

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        hf_config = self.ctx.get_hf_config(LlavaConfig)
        image_token_id = hf_config.image_token_index

        processor = self._get_hf_processor()
        if isinstance(processor, PixtralProcessor):
            image_token = processor.image_token
            image_break_token = processor.image_break_token
            image_end_token = processor.image_end_token

            vision_config = hf_config.vision_config
            assert isinstance(vision_config, PixtralVisionConfig)

            def get_replacement_pixtral(item_idx: int):
                image_size = mm_items.get_image_size(item_idx)
                (
                    num_width_tokens,
                    num_height_tokens,
                ) = get_pixtral_hf_image_feature_size(
                    vision_config,
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

                tokens = ([image_token] * num_width_tokens +
                          [image_break_token]) * num_height_tokens
                tokens[-1] = image_end_token

                return "".join(tokens)

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement_pixtral,
                ),
            ]

        max_image_tokens = get_max_llava_image_tokens(self.ctx)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=[image_token_id] * max_image_tokens,
            )
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        hf_config = self.ctx.get_hf_config(LlavaConfig)
        vision_config = hf_config.vision_config
        num_images = mm_counts["image"]

        if isinstance(vision_config, CLIPVisionConfig):
            data = dummy_image_for_clip(vision_config, num_images)
        elif isinstance(vision_config, SiglipVisionConfig):
            data = dummy_image_for_siglip(vision_config, num_images)
        elif isinstance(vision_config, PixtralVisionConfig):
            data = dummy_image_for_pixtral_hf(vision_config, num_images)
        else:
            msg = f"Unsupported vision config: {type(vision_config)}"
            raise NotImplementedError(msg)

        hf_processor = self._get_hf_processor()
        image_token = hf_processor.image_token

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=data,
            mm_processor_kwargs={},
        )


class LlavaLikeConfig(Protocol):
    vision_config: PretrainedConfig
    vision_feature_layer: Union[int, List[int]]


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
    """Given an signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index + 1


def init_vision_tower_for_llava(
    hf_config: LlavaLikeConfig,
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
):
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

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


@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llava_image_tokens)
@MULTIMODAL_REGISTRY.register_processor(LlavaMultiModalProcessor)
class LlavaForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
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
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
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
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        is_pixtral = kwargs.pop("is_pixtral", torch.tensor([False]))
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            assert isinstance(is_pixtral, torch.Tensor)
            if is_pixtral.any():
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
                    data=images,
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
                self.config.image_token_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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


class MantisMultiModalProcessor(LlavaMultiModalProcessor):

    def _get_hf_processor(self) -> ProcessorMixin:
        try:
            from mantis.models.mllava import MLlavaProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "You need to `pip install "
                "git+https://github.com/TIGER-AI-Lab/Mantis.git` "
                "to use this model") from exc

        processor = MLlavaProcessor.from_pretrained(
            self.ctx.model_config.tokenizer)
        assert isinstance(processor, ProcessorMixin)
        return processor


# To use this model, please use
# `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'`
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llava_image_tokens)
@MULTIMODAL_REGISTRY.register_processor(MantisMultiModalProcessor)
class MantisForConditionalGeneration(LlavaForConditionalGeneration):
    pass
