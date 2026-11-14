# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
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
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    DeepseekOCRProcessor,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphPathConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from ...transformers_utils.processors.deepseek_ocr import count_tiles
from .deepencoder import ImageEncoderViT
from .deepencoder2 import build_qwen2_decoder_as_encoder
from .deepseek_ocr import DeepseekOCRImagePixelInputs
from .deepseek_vl2 import MlpProjector

# The image token id may be various
IMAGE_SIZE = 768  # different from deepseek-ocr
_IMAGE_TOKEN = "<image>"


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
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        max_image_size = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=mm_counts.get("image", 0),
            )
        }


class DeepseekOCR2MultiModalProcessor(
    BaseMultiModalProcessor[DeepseekOCR2ProcessingInfo]
):
    def _get_hf_mm_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        if "text" in hf_inputs.hf_data:
            hf_inputs.hf_data["prompt"] = hf_inputs.hf_data.pop("text")

        return hf_inputs

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
            images_spatial_crop=MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
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
                assert isinstance(images, ImageProcessorItems)
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
class DeepseekOCR2ForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsEncoderCudaGraph
):
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
        self.model_config = vllm_config.model_config
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
                global_attn_indexes=(2, 5, 8, 11),
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

        if pixel_values is None:
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
        if patches.numel() == 0:
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
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model=["sam_model", "qwen2_model"],
        )

    @property
    def global_image_output_tokens(self) -> int:
        return (BASE_SIZE // 16 // 4) ** 2

    @property
    def single_patch_output_tokens(self) -> int:
        return (IMAGE_SIZE // 16 // 4) ** 2

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values"],
            out_hidden_size=self.projector_config.n_embed,
            paths={
                "global": EncoderCudaGraphPathConfig(
                    min_token_budget=self.global_image_output_tokens
                ),
                "local": EncoderCudaGraphPathConfig(
                    min_token_budget=self.single_patch_output_tokens,
                    allow_zero_tokens=True,
                ),
            },
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        return (
            self.global_image_output_tokens,
            min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.model_config.max_model_len,
            ),
        )

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        specs = []
        for crop in mm_kwargs["images_spatial_crop"]:
            is_tiled = crop[0] > 1 or crop[1] > 1
            num_patches = int(crop.prod().item()) if is_tiled else 0
            global_tokens = self.global_image_output_tokens
            local_tokens = num_patches * self.single_patch_output_tokens
            specs.append(
                EncoderItemSpec(
                    input_size=(BASE_SIZE // 16) ** 2
                    + num_patches * (IMAGE_SIZE // 16) ** 2,
                    output_tokens=global_tokens + local_tokens + 1,
                    path_output_tokens={
                        "global": global_tokens,
                        "local": local_tokens,
                    },
                )
            )
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        pixel_values = mm_kwargs["pixel_values"]
        images_crop = mm_kwargs["images_crop"]
        images_spatial_crop = mm_kwargs["images_spatial_crop"]
        if not indices:
            return {
                "pixel_values": pixel_values[:0],
                "images_crop": images_crop[:0],
                "images_spatial_crop": images_spatial_crop[:0],
            }

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(-1), 0)
        patch_offsets = [0]
        for count in patches_per_image:
            patch_offsets.append(patch_offsets[-1] + int(count))
        return {
            "pixel_values": pixel_values[indices],
            "images_crop": torch.cat(
                [images_crop[patch_offsets[i] : patch_offsets[i + 1]] for i in indices]
            ),
            "images_spatial_crop": images_spatial_crop[indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        if path == "global":
            count = min(max_batch_size, token_budget // self.global_image_output_tokens)
            key, side = "pixel_values", BASE_SIZE
        elif path == "local":
            count = token_budget // self.single_patch_output_tokens
            key, side = "images_crop", IMAGE_SIZE
        else:
            raise ValueError(f"Unknown encoder CUDA graph path: {path}")
        return EncoderCudaGraphCaptureInputs(
            values={key: torch.zeros(count, 3, side, side, device=device, dtype=dtype)}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        if path == "global":
            return EncoderCudaGraphReplayBuffers(
                values={"pixel_values": mm_kwargs["pixel_values"]}
            )
        if path == "local":
            return EncoderCudaGraphReplayBuffers(
                values={"images_crop": mm_kwargs["images_crop"]}
            )
        raise ValueError(f"Unknown encoder CUDA graph path: {path}")

    def _batched_encoder_forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projector(self.qwen2_model(self.sam_model(images)))
        return features.reshape(-1, features.shape[-1])

    def encoder_cudagraph_forward(
        self, values: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        if path == "global":
            return self._batched_encoder_forward(values["pixel_values"])
        if path == "local":
            return self._batched_encoder_forward(values["images_crop"])
        raise ValueError(f"Unknown encoder CUDA graph path: {path}")

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        if path == "default":
            image_input = self._parse_and_validate_image_input(**mm_kwargs)
            assert image_input is not None
            return torch.cat(self._process_image_input(image_input), dim=0)
        return self.encoder_cudagraph_forward(mm_kwargs, path=path)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert batch_mm_kwargs is not None
        crops = batch_mm_kwargs["images_spatial_crop"]
        is_tiled = (crops[:, 0] > 1) | (crops[:, 1] > 1)
        patch_counts = torch.where(is_tiled, crops.prod(-1), 0).tolist()
        hidden_size = outputs["global"].shape[-1]
        global_features = outputs["global"].reshape(
            -1, self.global_image_output_tokens, hidden_size
        )
        local_features = outputs.get("local")
        if local_features is not None:
            local_features = local_features.reshape(
                -1, self.single_patch_output_tokens, hidden_size
            )

        patch_offset = 0
        for i, index in enumerate(indices):
            count = patch_counts[i]
            parts = []
            if count:
                assert local_features is not None
                patches = local_features[patch_offset : patch_offset + count]
                parts.append(patches.flatten(0, 1))
                patch_offset += count
            parts.extend((global_features[i], self.view_seperator[None, :]))
            dest[index] = torch.cat(parts, dim=0)
