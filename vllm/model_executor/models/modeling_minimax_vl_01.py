"""PyTorch MiniMaxVL01 model."""

import math
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple, Union, Iterable, Final, Protocol, Mapping, TypedDict, Literal, TypeVar

from abc import abstractmethod
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.image_utils import to_numpy_array
from transformers import LlavaNextProcessor

from .utils import AutoWeightsLoader, embed_multimodal, init_vllm_registered_model, maybe_prefix
from vllm.config import VllmConfig
import re
from transformers import (
    BatchFeature,
)
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import ModelOutput
from vllm.sequence import IntermediateTensors
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from vllm.model_executor.layers.sampler import SamplerOutput
from transformers import AutoModel, AutoModelForCausalLM
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .modeling_clip import CLIPVisionModel, CLIPVisionConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm.multimodal import MULTIMODAL_REGISTRY
from .minimax_text_01 import MiniMaxText01ForCausalLM
from vllm.transformers_utils.configs import MiniMaxText01Config, MiniMaxVL01Config
from .llava import LlavaDummyInputsBuilder
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)
from PIL import Image

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import (BaseLlavaMultiModalProcessor, BaseLlavaProcessingInfo,
                    LlavaDummyInputsBuilder, LlavaLikeConfig,
                    LlavaImageInputs, PixtralHFImagePixelInputs, LlavaImagePixelInputs,LlavaImageEmbeddingInputs,
                    LlavaMultiModalProjector, init_vision_tower_for_llava, LlavaMultiModalProcessor)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, embed_multimodal, flatten_bn,
                    init_vllm_registered_model, maybe_prefix)
from .processing_minimax_vl_01 import MiniMaxVL01Processor
from vllm.multimodal.processing import BaseProcessingInfo
import numpy as np
import numpy.typing as npt
from .image_processer import ImageProcessor, _transform

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniMaxVL01Config"


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->MiniMaxVL01
class MiniMaxVL01CausalLMOutputWithPast(ModelOutput):
    """
    Base class for MiniMaxVL01 causal language model (or autoregressive) outputs.

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
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->MiniMaxVL01
class MiniMaxVL01MultiModalProjector(nn.Module):
    def __init__(self, config: MiniMaxVL01Config):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


MINIMAX_VL_01_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the
    generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniMaxVL01Config`] or [`MiniMaxVL01VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniMaxVL01 Model outputting raw hidden-states without any specific head on top.",
    MINIMAX_VL_01_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->MiniMaxVL01,llava->minimax_vl_01
class MiniMaxVL01PreTrainedModel(PreTrainedModel):
    config_class = MiniMaxVL01Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxVL01VisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

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

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


MINIMAX_VL_01_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`MiniMaxVL01ImageProcessor.__call__`] for details. [`MiniMaxVL01Processor`] uses
            [`MiniMaxVL01ImageProcessor`] for processing images.
        image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
            The sizes of the images in the batch, being (height, width) for each image.
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

            Contains pre-computed hidden-states (key and values in the self-attention
            blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

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
"""


# Register the processors
class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """


class LlavaNextImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageEmbeddingInputs]


class LlavaNextLikeConfig(LlavaLikeConfig, Protocol):
    image_grid_pinpoints: Final[list[list[int]]]
class MiniMaxVL01ProcessingInfo(BaseLlavaProcessingInfo):

    def get_hf_config(self) -> LlavaNextLikeConfig:
        from vllm.transformers_utils.configs.configuration_minimax_vl_01 import MiniMaxVL01Config
        return self.ctx.get_hf_config(MiniMaxVL01Config)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(MiniMaxVL01Processor, **kwargs)
        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L113
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=vision_encoder_info.get_patch_grid_length(),
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return unpadded_feature_size + newline_feature_size + base_feature_size

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = (original_height * current_width) // original_width
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = (original_width * current_height) // original_height
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for (height, width) in hf_config.image_grid_pinpoints:
            feat_size = self.get_num_image_tokens(image_width=width,
                                                  image_height=height)
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint

_I = TypeVar("_I", bound=MiniMaxVL01ProcessingInfo)
class BaseLlavaNextMultiModalProcessor(BaseLlavaMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError


class MiniMaxVL01MultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[MiniMaxVL01ProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )
    
_I = TypeVar("_I", bound=MiniMaxVL01ProcessingInfo)
class MiniMaxVL01DummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = "<image>"
        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )
    """
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        
        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

        prompt_text = image_token * num_images
        
        return ProcessorInputs(
            prompt_text=prompt_text,
            mm_data=mm_data,
            hf_processor_mm_kwargs={}
        )

    def _get_dummy_images(
        self,
        *,
        width: int,
        height: int,
        num_images: int,
    ) -> list[npt.NDArray]:
        # 创建有不同颜色的测试图像，使其更具辨识度
        images = []
        for i in range(num_images):
            # 创建不同颜色的图像
            color = ((i * 50) % 255, ((i+1) * 70) % 255, ((i+2) * 90) % 255)
            image = Image.new('RGB', (width, height), color=color)
            # 添加一些简单的形状以增加复杂性
            images.append(image)
        return images
    """

@add_start_docstrings(
    """The MiniMaxVL01 model which consists of a vision backbone and a language model.""",
    MINIMAX_VL_01_START_DOCSTRING,
)
@MULTIMODAL_REGISTRY.register_processor(MiniMaxVL01MultiModalProcessor, 
                                        info=MiniMaxVL01ProcessingInfo, 
                                        dummy_inputs=MiniMaxVL01DummyInputsBuilder)
class MiniMaxVL01ForConditionalGeneration(MiniMaxVL01PreTrainedModel, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        
        # 确保配置是正确的类型
        if not isinstance(config, MiniMaxVL01Config):
            raise ValueError(f"Expected config of type MiniMaxVL01Config, got {type(config)}")
        
        super().__init__(config)
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        
        # 初始化视觉模型
        try:
            vision_config = CLIPVisionConfig.from_dict(config.vision_config.to_dict())
            self.vision_tower = CLIPVisionModel(vision_config)
        except Exception as e:
            logger.error(f"Error initializing vision model: {e}")
            raise

        # 初始化多模态投影器
        self.multi_modal_projector = MiniMaxVL01MultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)

        # 初始化语言模型
        self.vocab_size = config.text_config.vocab_size
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        
        # 设置填充相关参数
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        
        # 完成初始化
        self.post_init()
        
        # 确保语言模型支持生成空中间张量
        if hasattr(self.language_model, 'make_empty_intermediate_tensors'):
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors)
        else:
            logger.warning("Language model does not have make_empty_intermediate_tensors method")

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        if multimodal_embeddings is None:
            return self.language_model.get_input_embeddings(input_ids)

        inputs_embeds = embed_multimodal(
            input_ids,
            self.config.image_token_index,
            self.language_model.model.get_input_embeddings,
            multimodal_embeddings,
        )
        return inputs_embeds

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

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "][<rewritten_image_token>\nWhat is shown in this image?<rewritten_image_token> ",
                    "][<rewritten_image_token>\nWhat is shown in this image?<rewritten_image_token> ",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in minimax 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True if not self.training else False
            if batch_size > 1 and not self.training:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<rewritten_image_token>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids

    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                #print('num_patch_height, num_patch_width',num_patch_height,num_patch_width)
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                #print('unpad', image_feature.shape)
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for MiniMaxVL01.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for each token in the input sequence.
            intermediate_tensors: Optional intermediate tensors for caching.
            inputs_embeds: Optional pre-computed embeddings.
            **kwargs: Additional arguments including:
                - pixel_values: The pixels in each grid patch for each input image.
                - image_sizes: The original `(height, width)` for each input image.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            # 处理图像输入
            pixel_values = kwargs.pop("pixel_values", None)
            image_sizes = kwargs.pop("image_sizes", None)
            
            if pixel_values is not None and len(input_ids.shape) > 1 and input_ids.shape[1] != 1:
                # 处理图像特征
                vision_feature_layer = kwargs.pop("vision_feature_layer", self.config.vision_feature_layer)
                vision_feature_select_strategy = kwargs.pop("vision_feature_select_strategy", self.config.vision_feature_select_strategy)
                
                # 提取图像特征
                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]
                
                # 处理图像特征
                selected_image_feature = torch.chunk(selected_image_feature, len(pixel_values), dim=1)
                selected_image_feature = torch.cat(selected_image_feature, dim=0)
                
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                
                # 投影图像特征
                image_features = self.multi_modal_projector(selected_image_feature)
                
                # 处理不同大小的图像
                if image_sizes is not None:
                    # 计算每个图像的patch数量
                    image_num_patches = [
                        image_size_to_num_patches(
                            image_size=imsize,
                            grid_pinpoints=self.config.image_grid_pinpoints,
                            patch_size=self.config.vision_config.image_size,
                        )
                        for imsize in image_sizes
                    ]
                    
                    # 分割图像特征
                    image_features = torch.split(image_features, image_num_patches, dim=0)
                    
                    # 打包图像特征
                    image_features, feature_lens = self.pack_image_features(
                        image_features,
                        image_sizes,
                        image_newline=self.image_newline,
                    )
                
                # 获取文本嵌入
                inputs_embeds = self.get_input_embeddings(input_ids)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                
                # 使用掩码替换图像标记
                special_image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            else:
                # 如果没有图像输入，直接使用文本嵌入
                inputs_embeds = self.get_input_embeddings(input_ids)
            
            input_ids = None

        # 调用语言模型进行前向传播
        hidden_states = self.language_model.model(input_ids,
                                                 positions,
                                                 intermediate_tensors,
                                                 inputs_embeds=inputs_embeds)
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        past_length = len(past_key_values)
        if past_length == 0:
            past_key_values = None
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            **kwargs
        )
        if past_length == 0:
            model_inputs.update(
                {
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                }
            )
        return model_inputs

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
    
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

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if self.config.vision_config.model_type == "pixtral":
                embed_is_patch = kwargs.pop("embed_is_patch")
                if not isinstance(embed_is_patch, (torch.Tensor, list)):
                    raise ValueError("Incorrect type of embed_is_patch. "
                                     f"Got type: {type(embed_is_patch)}")

                embed_is_patch = flatten_bn(embed_is_patch)

                return PixtralHFImagePixelInputs(
                    type="pixel_values_pixtral",
                    pixel_values=flatten_bn(pixel_values),
                    embed_is_patch=embed_is_patch,
                )

            return LlavaImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(
                    flatten_bn(pixel_values, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            if self.config.vision_config.model_type == "pixtral":
                raise ValueError("Pixtral-HF does not support image_embeds.")

            return LlavaImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds, concat=True),
            )

        raise AssertionError("This line should be unreachable.")
    
    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs):
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs)
