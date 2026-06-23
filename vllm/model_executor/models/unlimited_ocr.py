# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    DeepseekOCRProcessor,
)

from ...transformers_utils.processors.deepseek_ocr import count_tiles
from .deepencoder import ImageEncoderViT, build_sam_vit_b
from .deepencoder2 import build_qwen2_decoder_as_encoder
from .deepseek_ocr import DeepseekOCRImagePixelInputs
from .deepseek_vl2 import MlpProjector

# The image token id may be various
IMAGE_SIZE = 768  # different from deepseek-ocr
_IMAGE_TOKEN = "<image>"
_UNLIMITED_IMAGE_SIZE = 640
_UNLIMITED_BASE_SIZE = 1024


class DeepseekOCR2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        v2_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v2",
        )

        return self.ctx.get_hf_processor(
            DeepseekOCRProcessor,
            **{**v2_processor_config, **kwargs},
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 768 and image_height <= 768:
                crop_ratio = [1, 1]
            else:
                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(
                    image_width, image_height, image_size=IMAGE_SIZE
                )

            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * w
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024 * 2, height=1024 * 2)
        return ImageSize(width=768 * 2, height=768 * 2)


class DeepseekOCR2DummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekOCR2ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
            )
        }


class DeepseekOCR2MultiModalProcessor(
    BaseMultiModalProcessor[DeepseekOCR2ProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        images_spatial_crop = hf_inputs.get("images_spatial_crop", torch.empty((0, 2)))
        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            images_crop=MultiModalFieldConfig.flat_from_sizes(
                "image", patches_per_image
            ),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                size = images.get_image_size(item_idx)

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=DeepseekOCR2ProcessingInfo,
    dummy_inputs=DeepseekOCR2DummyInputsBuilder,
)
class DeepseekOCR2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # map prefix for language backbone
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
            # remove "model." prefix for other components
            "model.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config
        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        with self._mark_tower_model(vllm_config, "image"):
            self.sam_model = ImageEncoderViT(
                depth=12,
                embed_dim=768,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=256,
                last_conv_output=896,
            )
            self.qwen2_model = build_qwen2_decoder_as_encoder()

            self.projector = MlpProjector(self.projector_config)
            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos

            # special token for image token sequence format
            n_embed = self.projector_config.n_embed
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            if self.tile_tag == "2D":
                # This is a typo in original implementation
                self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
            else:
                raise ValueError(
                    f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
                )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DeepseekOCRImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)

        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        base_size = self.vision_config.image_size
        return DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
            },
        )

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        global_features_1 = self.sam_model(image_tensor)
        global_features_2 = self.qwen2_model(global_features_1)

        features = self.projector(global_features_2)

        _, hw, dim = features.shape

        return features.view(-1, dim)

    def _encode_local_features(self, patches: torch.Tensor) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        local_features = self.sam_model(patches)
        local_features = self.qwen2_model(local_features)

        features = self.projector(local_features)

        _, _, dim = features.shape

        return features.view(-1, dim)

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        images_in_this_batch = []

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        images_crop = images_crop.split(patches_per_image.tolist())
        for jdx in range(images_spatial_crop.size(0)):
            patches = images_crop[jdx]
            image_ori = pixel_values[[jdx]]

            global_features = self._encode_global_features(image_ori)
            local_features = self._encode_local_features(patches)

            if local_features is not None:
                combined = torch.cat(
                    [local_features, global_features, self.view_seperator[None, :]],
                    dim=0,
                )
            else:
                combined = torch.cat(
                    [global_features, self.view_seperator[None, :]], dim=0
                )

            images_in_this_batch.append(combined)

        return images_in_this_batch

    def _process_image_input(
        self, image_input: DeepseekOCRImagePixelInputs
    ) -> torch.Tensor:
        pixel_values = image_input.data
        images_crop = image_input.images_crop
        images_spatial_crop = image_input.images_spatial_crop.to(dtype=torch.long)

        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        return vision_features

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
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model=["sam_model", "qwen2_model"],
        )


class UnlimitedOCRProcessingInfo(DeepseekOCR2ProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        processor_config = dict(
            image_size=_UNLIMITED_IMAGE_SIZE,
            base_size=_UNLIMITED_BASE_SIZE,
            crop_mode=True,
            strategy="v1",
        )
        return self.ctx.get_hf_processor(
            DeepseekOCRProcessor,
            **{**processor_config, **kwargs},
        )

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        image_size = _UNLIMITED_IMAGE_SIZE
        base_size = _UNLIMITED_BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if cropping and (image_width > image_size or image_height > image_size):
            num_width_tiles, num_height_tiles = count_tiles(
                image_width, image_height, image_size=image_size
            )
        else:
            num_width_tiles = num_height_tiles = 1

        global_side = math.ceil((base_size // patch_size) / downsample_ratio)
        local_side = math.ceil((image_size // patch_size) / downsample_ratio)

        global_tokens = global_side * (global_side + 1)
        local_tokens = 0
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_tokens = (
                (local_side * num_width_tiles + 1) * local_side * num_height_tiles
            )
        return global_tokens + local_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=_UNLIMITED_IMAGE_SIZE * 2, height=_UNLIMITED_IMAGE_SIZE * 2)


class UnlimitedOCRDummyInputsBuilder(
    BaseDummyInputsBuilder[UnlimitedOCRProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_processor().image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
            )
        }


class UnlimitedOCRCLIPVisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(num_patches + 1).expand((1, -1))
        )

    @staticmethod
    def _resize_pos(abs_pos: torch.Tensor, tgt_size: int) -> torch.Tensor:
        dim = abs_pos.size(-1)
        abs_pos = abs_pos.squeeze(0)
        cls_token, old_pos_embed = abs_pos[:1], abs_pos[1:]
        src_size = int(math.sqrt(abs_pos.shape[0] - 1))
        tgt_size = int(math.sqrt(tgt_size))
        if src_size == tgt_size:
            return abs_pos.view(1, tgt_size * tgt_size + 1, dim)
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim)
        old_pos_embed = old_pos_embed.permute(0, 3, 1, 2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(abs_pos.dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        return torch.cat([cls_token, new_pos_embed], dim=0).view(
            1, tgt_size * tgt_size + 1, dim
        )

    def forward(
        self, pixel_values: torch.Tensor, patch_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        pos = self._resize_pos(
            self.position_embedding(self.position_ids), embeddings.size(1)
        )
        return embeddings + pos


class UnlimitedOCRCLIPAttention(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = torch.unbind(qkv, dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out_proj(out)


class UnlimitedOCRCLIPBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = UnlimitedOCRCLIPAttention()
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(1024, 4096, bias=True)
        self.mlp.fc2 = nn.Linear(4096, 1024, bias=True)
        self.layer_norm1 = nn.LayerNorm(1024, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(1024, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        y = self.mlp.fc1(self.layer_norm2(x))
        x = x + self.mlp.fc2(y * torch.sigmoid(1.702 * y))
        return x


class UnlimitedOCRCLIPTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(UnlimitedOCRCLIPBlock() for _ in range(24))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class UnlimitedOCRCLIPVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = UnlimitedOCRCLIPVisionEmbeddings()
        self.transformer = UnlimitedOCRCLIPTransformer()
        self.pre_layrnorm = nn.LayerNorm(1024, eps=1e-5)

    def forward(
        self, x: torch.Tensor, patch_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(hidden_states)
        return self.transformer(hidden_states)


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCR2MultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=UnlimitedOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCR2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        compat_config = DeepseekVLV2Config(**config.to_dict())
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.vision_config = compat_config.vision_config
        self.projector_config = compat_config.projector_config
        self.text_config = compat_config.text_config
        self.text_config.architectures = ["DeepseekV2ForCausalLM"]
        self.text_config.model_type = "deepseek_v2"

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        with self._mark_tower_model(vllm_config, "image"):
            self.sam_model = build_sam_vit_b()
            self.vision_model = UnlimitedOCRCLIPVisionModel()
            self.projector = MlpProjector(self.projector_config)
            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos

            n_embed = self.projector_config.n_embed
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            if self.tile_tag != "2D":
                raise ValueError(
                    f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
                )
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        sam_features = self.sam_model(image_tensor)
        clip_features = self.vision_model(image_tensor, sam_features)
        features = torch.cat(
            [clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)],
            dim=-1,
        )
        features = self.projector(features)
        _, hw, dim = features.shape
        side = int(hw**0.5)
        features = features.view(side, side, dim)
        features = torch.cat(
            [features, self.image_newline[None, None, :].expand(side, 1, dim)],
            dim=1,
        )
        return features.view(-1, dim)

    def _encode_local_features(
        self, patches: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        sam_features = self.sam_model(patches)
        clip_features = self.vision_model(patches, sam_features)
        features = torch.cat(
            [clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)],
            dim=-1,
        )
        features = self.projector(features)
        _, hw, dim = features.shape
        side = int(hw**0.5)
        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]
        features = features.view(
            height_crop_num, width_crop_num, side, side, dim
        ).permute(0, 2, 1, 3, 4)
        features = features.reshape(height_crop_num * side, width_crop_num * side, dim)
        features = torch.cat(
            [
                features,
                self.image_newline[None, None, :].expand(
                    height_crop_num * side, 1, dim
                ),
            ],
            dim=1,
        )
        return features.view(-1, dim)

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        images_in_this_batch = []

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        images_crop = images_crop.split(patches_per_image.tolist())
        for jdx in range(images_spatial_crop.size(0)):
            patches = images_crop[jdx]
            image_ori = pixel_values[[jdx]]

            global_features = self._encode_global_features(image_ori)
            local_features = self._encode_local_features(
                patches, images_spatial_crop[jdx]
            )

            if local_features is not None:
                combined = torch.cat(
                    [local_features, global_features, self.view_seperator[None, :]],
                    dim=0,
                )
            else:
                combined = torch.cat(
                    [global_features, self.view_seperator[None, :]], dim=0
                )

            images_in_this_batch.append(combined)

        return images_in_this_batch

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model=["sam_model", "vision_model"],
        )
