# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from https://github.com/amalad/vllm/blob/nemotron_parse/vllm/model_executor/models/nemotron_parse.py
# that's based on https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/hf_nemotron_parse_modeling.py
#
# Bart classes based on old vLLM codebase:
# https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/models/bart.py

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision import transforms as T
from transformers import (
    BartConfig,
    BatchFeature,
    PretrainedConfig,
    TensorType,
)

from vllm.attention.backends.abstract import AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.radio import RadioModel
from vllm.model_executor.models.whisper import WhisperAttention, WhisperCrossAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)
DEFAULT_FINAL_IMAGE_SIZE = (2048, 1648)


class BartScaledWordEmbedding(VocabParallelEmbedding):
    """
    This module overrides VocabParallelEmbedding's
    forward by multiplying with embeddings scale.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class BartParallelLMHead(ParallelLMHead):
    """
    This module overrides ParallelLMHead's
    forward by dividing by embeddings scale,
    yielding effectively the inverse of
    BartScaledWordEmbedding
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) / self.embed_scale


class BartDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        """
        afeldman-nm: personally I would call this "cross-attention",
        however I left the name as "encoder_attn" to maintain consistency
        with the name of the pretrained weights.
        """
        self.encoder_attn = WhisperCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_hidden_states: torch.Tensor of *decoder* input embeddings.
            encoder_hidden_states: torch.Tensor of *encoder* input embeddings.
        Returns:
            Decoder layer output torch.Tensor
        """
        residual = decoder_hidden_states

        # Self Attention
        hidden_states = self.self_attn(hidden_states=decoder_hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block

        residual = hidden_states

        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class MBartDecoderLayer(BartDecoderLayer):
    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states

        # Cross-Attention Block

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class MBartDecoderNoPos(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers.
    Each layer is a [`BartDecoderLayer`]
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.layers = nn.ModuleList(
            [
                MBartDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )

        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_input_ids: Indices of *decoder* input sequence tokens in the
                vocabulary. Padding will be ignored by default should you provide it.
            encoder_hidden_states: Tensor of encoder output embeddings
        Returns:
            Decoder output torch.Tensor
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(decoder_input_ids)

        hidden_states = self.layernorm_embedding(inputs_embeds)

        # decoder layers

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("embed_positions"):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class NemotronParsePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("b", 3, "h", "w")]


class NemotronParseImageProcessor:
    """
    NemotronParse Image Processor
    """

    def __init__(
        self,
        final_size: tuple = DEFAULT_FINAL_IMAGE_SIZE,
        **kwargs,
    ):
        # Ensure final_size is properly formatted
        if isinstance(final_size, (list, tuple)) and len(final_size) >= 2:
            self.final_size = (int(final_size[0]), int(final_size[1]))
        elif isinstance(final_size, (int, float)):
            self.final_size = (int(final_size), int(final_size))
        else:
            self.final_size = DEFAULT_FINAL_IMAGE_SIZE  # Default fallback

        self.norm_mean = torch.Tensor(OPENAI_CLIP_MEAN).reshape(1, 3, 1, 1)
        self.norm_std = torch.Tensor(OPENAI_CLIP_STD).reshape(1, 3, 1, 1)

        # Create transforms
        self._create_transforms()

    def _create_transforms(self):
        """Create transform objects."""
        try:
            import albumentations as A
        except ImportError as err:
            raise ImportError(
                "The package `albumentations` is required to use "
                "NemotronParse model. Please install it with `pip install "
                "albumentations`."
            ) from err

        # Ensure final_size is a tuple of integers
        if isinstance(self.final_size, (list, tuple)):
            self.target_height, self.target_width = (
                int(self.final_size[0]),
                int(self.final_size[1]),
            )
        else:
            self.target_height = self.target_width = int(self.final_size)

        self.transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.target_height,
                    min_width=self.target_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=[255, 255, 255],
                    p=1.0,
                ),
            ]
        )

        self.torch_transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio (exact replica of original
        LongestMaxSizeHW)."""
        height, width = image.shape[:2]
        max_size_height = self.target_height
        max_size_width = self.target_width

        # Original LongestMaxSizeHW algorithm from custom_augmentations.py
        aspect_ratio = width / height
        new_height = height
        new_width = width

        # If height too big then scale image down
        if height > max_size_height:
            new_height = max_size_height
            new_width = int(new_height * aspect_ratio)

        # If width too big, scale image down further
        if new_width > max_size_width:
            new_width = max_size_width
            new_height = int(new_width / aspect_ratio)

        # Use cv2.INTER_LINEAR like the original
        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def _pad_to_size(self, image: np.ndarray) -> np.ndarray:
        """Pad image to target size with white padding (matches A.PadIfNeeded
        behavior)."""
        h, w = image.shape[:2]
        min_height, min_width = self.target_height, self.target_width

        # Only pad if image is smaller than target (matches A.PadIfNeeded logic)
        pad_h = max(0, min_height - h)
        pad_w = max(0, min_width - w)

        if pad_h == 0 and pad_w == 0:
            return image

        # A.PadIfNeeded pads to bottom-right with constant value
        if len(image.shape) == 3:
            # Color image - pad bottom and right with white (255, 255, 255)
            padded = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        else:
            # Grayscale image - pad with white (255)
            padded = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255
            )

        return padded

    def preprocess(
        self,
        images: Image.Image | list[Image.Image],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Preprocess an image or batch of images for the NemotronParse model.

        Args:
            images: Input image(s)
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Convert PIL images to numpy arrays if needed
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.asarray(image)
            processed_images.append(image)

        # Apply NemotronParse-specific transforms
        pixel_values = []
        for image in processed_images:
            # Manual resize with aspect ratio preservation
            # (replaces LongestMaxSizeHW)
            processed_image = self._resize_with_aspect_ratio(image)

            # Apply remaining albumentations transforms if available
            if self.transform is not None:
                transformed = self.transform(image=processed_image)
                processed_image = transformed["image"]
            else:
                # Fallback: just pad to target size
                processed_image = self._pad_to_size(processed_image)

            # Convert to tensor
            pixel_values_tensor = self.torch_transform(processed_image)

            # Handle grayscale images
            if pixel_values_tensor.shape[0] == 1:
                pixel_values_tensor = pixel_values_tensor.expand(3, -1, -1)

            pixel_values.append(pixel_values_tensor)

        # Stack into batch
        pixel_values = torch.stack(pixel_values)

        # Normalize pixel values
        normalized_values = (pixel_values - self.norm_mean) / self.norm_std
        return {"pixel_values": normalized_values}

    def __call__(
        self, images: Image.Image | list[Image.Image], **kwargs
    ) -> dict[str, torch.Tensor]:
        return self.preprocess(images, **kwargs)


class NemotronParseProcessor:
    """
    NemotronParse Processor
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.image_processor = NemotronParseImageProcessor(final_size=config.image_size)

    def _make_batch_input(self, input_item=None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def __call__(
        self,
        text: str | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        text, images = [self._make_batch_input(x) for x in (text, images)]
        image_inputs = {} if len(images) == 0 else self.image_processor(images)

        text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
        combined_outputs = BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )
        return combined_outputs


class NemotronParseProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs) -> NemotronParseProcessor:
        return self.ctx.init_processor(
            NemotronParseProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        config = self.get_hf_config()
        final_size = config.image_size
        patch_size = config.encoder.patch_size

        return (final_size[0] // patch_size) * ((final_size[1] // patch_size) // 4) + 1

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        image_tokens = self.get_num_image_tokens()
        return {"image": image_tokens}


class NemotronParseDummyInputsBuilder(
    BaseDummyInputsBuilder[NemotronParseProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_hf_config().image_size

        return {
            "image": self._get_dummy_images(
                width=target_width, height=target_height, num_images=num_images
            )
        }


class NemotronParseMultiModalProcessor(
    EncDecMultiModalProcessor[NemotronParseProcessingInfo]
):
    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        return [0]

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs
            )
        else:
            hf_processor = self.info.get_hf_processor()
            tokenizer = hf_processor.tokenizer
            processed_outputs = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_image_tokens = self.info.get_num_image_tokens()

        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=[0] * num_image_tokens,
            )
        ]


class RadioWithNeck(nn.Module):
    """Vision encoder using RADIO model with custom neck."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config.encoder

        self.model_encoder = self.get_vit_model_from_radio_config(
            config, quant_config=quant_config
        )

        # Neck components
        last_hidden_state = 1024
        self.conv1 = nn.Conv1d(1280, last_hidden_state, 1)
        self.layer_norm1 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )
        self.conv2 = nn.Conv2d(
            last_hidden_state,
            last_hidden_state,
            kernel_size=(1, 4),
            stride=(1, 4),
            padding=0,
            bias=False,
        )
        self.layer_norm2 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )
        self.sum_proj = ColumnParallelLinear(
            3840,
            last_hidden_state,
            quant_config=quant_config,
            prefix=f"{prefix}.sum_proj",
        )
        self.layer_norm3 = nn.LayerNorm(
            last_hidden_state, eps=1e-06, elementwise_affine=True
        )

    def get_vit_model_from_radio_config(
        self,
        hf_config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> RadioModel:
        hf_config_vision = hf_config.encoder
        model_name = hf_config_vision.args.get("model")
        if model_name is None:
            raise ValueError(f"Unsupported vit model type: {model_name}")

        radio_config = RadioConfig(
            model_name=model_name,
            image_size=hf_config.image_size,
            **hf_config_vision.args,
        )

        return RadioModel(config=radio_config, quant_config=quant_config)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        summary, feature = self.model_encoder(pixel_values)

        output = self.conv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.layer_norm1(output)

        patch_size = self.config.patch_size
        output = rearrange(
            output,
            "b (h w) d -> b d h w",
            h=pixel_values.shape[-2] // patch_size,
            w=pixel_values.shape[-1] // patch_size,
        )

        output = self.conv2(output)
        output = rearrange(output, "b d h w -> b (h w) d")
        output = self.layer_norm2(output)
        summary = self.layer_norm3(self.sum_proj(summary)[0])
        output = torch.cat((output, summary.unsqueeze(1)), dim=1)

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_encoder_weights = []
        adaptor_dict = {
            name: param
            for name, param in dict(self.named_parameters()).items()
            if not name.startswith("model_encoder")
        }
        for name, w in weights:
            if name.startswith("model_encoder"):
                model_encoder_weights.append((".".join(name.split(".")[1:]), w))
            else:
                param = adaptor_dict[name]
                with torch.no_grad():
                    default_weight_loader(param, w)

        self.model_encoder.load_weights(model_encoder_weights)


@MULTIMODAL_REGISTRY.register_processor(
    NemotronParseMultiModalProcessor,
    info=NemotronParseProcessingInfo,
    dummy_inputs=NemotronParseDummyInputsBuilder,
)
class NemotronParseForConditionalGeneration(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vision_config = config.encoder
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.encoder = RadioWithNeck(
            config=config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )

        self.decoder = MBartDecoderNoPos(
            config.decoder,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.decoder",
        )

        self.vocab_size = config.decoder.vocab_size
        self.lm_head = ParallelLMHead(
            config.decoder.vocab_size, config.decoder.d_model, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(
            self.vocab_size, config.decoder.vocab_size
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> NemotronParsePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            h, w = self.config.image_size
            return NemotronParsePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={
                    "h": h,
                    "w": w,
                },
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self, image_input: NemotronParsePixelInputs
    ) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype)
        return self.encoder(pixel_values)

    def get_language_model(self) -> torch.nn.Module:
        return self.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids: torch.Tensor of *decoder* input token ids.
            positions: torch.Tensor of *decoder* position indices.
            encoder_outputs: List of encoder output tensors (vision embeddings).
                During profiling, this may be None or empty.
        Returns:
            Output torch.Tensor
        """
        inputs_embeds = None
        if encoder_outputs:
            inputs_embeds = torch.cat(encoder_outputs, dim=0)
        hidden_states = self.decoder(
            decoder_input_ids=input_ids, encoder_hidden_states=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        lm_head_dict = dict(self.lm_head.named_parameters())

        def is_encoder(name: str) -> bool:
            return name.startswith("encoder")

        def is_decoder(name: str) -> bool:
            return name.startswith("decoder")

        def is_lm_head(name: str):
            return name.startswith("lm_head")

        # Separate weights by component
        encoder_weights = []
        decoder_weights = []

        for name, w in weights:
            if is_encoder(name):
                encoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_decoder(name):
                decoder_weights.append((".".join(name.split(".")[1:]), w))
            elif is_lm_head(name):
                trimmed_name = ".".join(name.split(".")[1:])
                param = lm_head_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            else:
                logger.info("Found unexpected weight: %s", name)

        # Load encoder weights
        self.encoder.load_weights(encoder_weights)
        # Load decoder weights
        self.decoder.load_weights(decoder_weights)
