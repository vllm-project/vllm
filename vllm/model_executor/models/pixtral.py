# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Annotated, Any, ClassVar, Literal, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from packaging.version import Version
from transformers import PixtralVisionConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)
from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb

from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalConfig, MultiModalDummyOptions
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import SiluAndMul, get_act_and_mul_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    HFMultiModalInputs,
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonImageProcessor,
    MistralCommonPixtralProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    supports_eagle3,
)
from .module_mapping import MultiModelKeys
from .utils import StageMissingLayer, init_vllm_registered_model, maybe_prefix
from .vision import (
    VisionEncoderInfo,
    VisionFeatureSelectStrategy,
    get_fp8_padded_hidden_size,
    is_vit_use_data_parallel,
    resolve_visual_encoder_outputs,
)

PATCH_MERGE = "patch_merge"

# Transformers 5.17 renamed Pixtral's rotary embedding and switched it to axial
# RoPE, which takes 2D (height, width) positions instead of flattened grid ids.
TRANSFORMERS_VERSION = importlib.metadata.version("transformers")
TRANSFORMERS_WITH_AXIAL_ROPE = Version(TRANSFORMERS_VERSION) >= Version("5.17.0.dev0")

if TRANSFORMERS_WITH_AXIAL_ROPE:
    from transformers.models.pixtral.modeling_pixtral import (
        PixtralVisionRotaryEmbedding,
    )
else:
    from transformers.models.pixtral.modeling_pixtral import (
        PixtralRotaryEmbedding as PixtralVisionRotaryEmbedding,
    )


def _patch_grid_sizes(
    images: Sequence[torch.Tensor], patch_size: int
) -> list[tuple[int, int]]:
    return [
        (image.shape[-2] // patch_size, image.shape[-1] // patch_size)
        for image in images
    ]


def _patch_grid_positions(
    grid_sizes: Sequence[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    """(row, col) coordinates of every patch, packed as (num_patches, 2)."""
    positions = np.concatenate(
        [np.indices((h, w)).reshape(2, -1).T for h, w in grid_sizes]
    )
    return async_tensor_h2d(positions, device=device)


class PackedAttentionMetadata(TypedDict):
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor
    sequence_lengths: torch.Tensor | None


def _make_packed_sequence_metadata(
    sequence_lengths: list[int],
    attn_backend: AttentionBackendEnum,
    hidden_size: int,
    tp_size: int,
    device: torch.device,
    *,
    num_sequences: int | None = None,
    fp8_padded_hidden_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build varlen attention metadata for images packed in one sequence.

    `num_sequences` pads the batch with zero-length sequences, which keeps the
    metadata shapes fixed for encoder CUDA graphs while attention does no work
    for the padding rows.
    """
    lengths = np.array(sequence_lengths, dtype=np.int32)
    if num_sequences is not None:
        assert len(lengths) <= num_sequences
        lengths = np.pad(lengths, (0, num_sequences - len(lengths)))
    cu_seqlens = np.concatenate(
        [np.zeros(1, dtype=np.int32), lengths.cumsum(dtype=np.int32)]
    )
    sequence_lengths_tensor = MMEncoderAttention.maybe_compute_seq_lens(
        attn_backend, cu_seqlens, device
    )
    # Attention wrappers call .item() on max_seqlen; keeping it on the CPU
    # avoids a device sync (and a D2H copy recorded in encoder CUDA graphs).
    max_seqlen = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(attn_backend, cu_seqlens),
        dtype=torch.int32,
    )
    cu_seqlens_tensor = MMEncoderAttention.maybe_recompute_cu_seqlens(
        attn_backend,
        cu_seqlens,
        hidden_size,
        tp_size,
        device,
        fp8_padded_hidden_size=fp8_padded_hidden_size,
    )
    return cu_seqlens_tensor, max_seqlen, sequence_lengths_tensor


def _make_merge_indices(
    grid_sizes: Sequence[tuple[int, int]],
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Indices gathering each `spatial_merge_size`^2 patch block of the packed
    sequence into one row, matching the patch mergers' unfold order."""
    m = spatial_merge_size
    indices = []
    offset = 0
    for h, w in grid_sizes:
        grid = np.arange(offset, offset + h * w).reshape(h // m, m, w // m, m)
        indices.append(grid.transpose(0, 2, 1, 3).reshape(-1, m * m))
        offset += h * w
    return async_tensor_h2d(np.concatenate(indices), device=device)


def _merge_patches_by_index(
    features: torch.Tensor, merge_indices: torch.Tensor
) -> torch.Tensor:
    # (N, D) -> (N / m^2, D * m^2), channel-major like F.unfold.
    merged = features[merge_indices]
    return merged.transpose(1, 2).reshape(merge_indices.shape[0], -1)


def _is_layer_none_or_staged(layer: nn.Module) -> bool:
    return layer is None or isinstance(layer, StageMissingLayer)


class PixtralImagePixelInputs(TensorSchema):
    """Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image

    The result of stacking `ImageEncoding.tokens` from each prompt.
    """

    type: Literal["pixel_values"] = "pixel_values"

    images: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", 3, "h", "w", dynamic_dims={"h", "w"}),
    ]


class PixtralProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")

        return tokenizer

    def get_image_processor(self) -> MistralCommonImageProcessor:
        return MistralCommonImageProcessor(self.get_tokenizer().instruct.mm_encoder)

    def get_hf_processor(self, **kwargs) -> MistralCommonPixtralProcessor:
        return MistralCommonPixtralProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        max_image_size = image_processor.mm_encoder.mm_config.max_image_size

        return ImageSize(width=max_image_size, height=max_image_size)


class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class PixtralMultiModalProcessor(BaseMultiModalProcessor[PixtralProcessingInfo]):
    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
        # For test_common.py only
        mm_data: MultiModalDataDict | None = None,
    ) -> ProcessorInputs:
        builder = self.dummy_inputs
        tokenizer = self.info.get_tokenizer()

        dummy_text = builder.get_dummy_text(mm_counts)
        dummy_mm_data = (
            builder.get_dummy_mm_data(seq_len, mm_counts, mm_options)
            if mm_data is None
            else mm_data
        )
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data)
        dummy_images = (
            [] if "image" not in dummy_mm_data else dummy_mm_items["image"].get_all()
        )

        request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text=dummy_text),
                        *(ImageChunk(image=image) for image in dummy_images),
                    ]
                ),
            ]
        )
        res = tokenizer.mistral.encode_chat_completion(request)
        dummy_tokens = res.tokens

        return ProcessorInputs(prompt=dummy_tokens, mm_data_items=dummy_mm_items)

    # The tokens are already inserted by the chat template,
    # so we just double check that they exist
    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_res.kwargs, mm_item_counts)
        self._validate_mm_updates(mm_res.prompt_updates, mm_item_counts)

        mm_placeholders = self._find_mm_placeholders(
            mm_res.prompt_ids,
            mm_res.prompt_updates,
        )
        self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

        return mm_res.prompt_ids, mm_placeholders

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(images=MultiModalFieldConfig.batched("image"))

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)

        # Avoid padding issue
        return hf_inputs._replace(
            hf_kwargs=dict(hf_inputs.hf_kwargs, return_tensors=None)
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_break_id = processor.image_break_id
        image_token_id = processor.image_token_id
        image_end_id = processor.image_end_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            _, nrows, ncols = processor.image_processor.get_number_of_image_patches(
                image_size.height,
                image_size.width,
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[],  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]


class PixtralEncoderCudaGraphMixin(SupportsEncoderCudaGraph):
    """Encoder CUDA graph support for models built on a Pixtral vision tower.

    All images of a batch are packed into one patch sequence. Capture buffers
    are sized for the token budget, and the batch is padded with zero-length
    sequences up to `max_batch_size`, so replays only copy fresh values into
    fixed-shape buffers.
    """

    # mm_kwargs key holding the list of CHW images.
    encoder_cudagraph_pixel_key: ClassVar[str]

    config: Any
    multimodal_config: MultiModalConfig | None

    @property
    def encoder_cudagraph_vision_tower(
        self,
    ) -> "VisionTransformer | PixtralHFVisionModel":
        raise NotImplementedError

    @property
    def encoder_cudagraph_spatial_merge_size(self) -> int:
        """Patches merged per side by the projector, 1 when not merging."""
        raise NotImplementedError

    def project_packed_image_features(
        self, features: torch.Tensor, merge_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """Map (N_patches, D) tower features to language model embeddings."""
        raise NotImplementedError

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        tower = self.encoder_cudagraph_vision_tower
        attn = tower.encoder_attention
        # SDPA splits the packed batch with Python sequence lengths, and
        # dynamic FP8 scales update their state on the host.
        graph_safe = (
            tower.attn_backend != AttentionBackendEnum.TORCH_SDPA
            and not attn._fp8_dynamic_scale
        )
        # Graph replays only handle pixel inputs, not precomputed embeddings.
        mm_config = self.multimodal_config
        pixels_only = mm_config is None or not mm_config.enable_mm_embeds
        return EncoderCudaGraphConfig(
            modalities=["image"] if graph_safe and pixels_only else [],
            buffer_keys=[
                "patches",
                "positions",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
                "merge_indices",
            ],
            out_hidden_size=self.config.text_config.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        tower = self.encoder_cudagraph_vision_tower
        merged_patch_size = tower.patch_size * self.encoder_cudagraph_spatial_merge_size
        # Smallest budget: a 224x224 image.
        min_budget = math.ceil(224 / merged_patch_size) ** 2
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return min_budget, max_budget

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        m = self.encoder_cudagraph_spatial_merge_size
        grid_sizes = _patch_grid_sizes(
            mm_kwargs[self.encoder_cudagraph_pixel_key],
            self.encoder_cudagraph_vision_tower.patch_size,
        )
        specs = []
        for h, w in grid_sizes:
            # The processors resize images to a multiple of the merged patch
            # size; the capture buffers rely on it too.
            if h % m or w % m:
                raise ValueError(
                    f"Image patch grid {h}x{w} is not divisible by the "
                    f"spatial merge size {m}."
                )
            specs.append(
                EncoderItemSpec(input_size=h * w, output_tokens=(h // m) * (w // m))
            )
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        images = mm_kwargs[self.encoder_cudagraph_pixel_key]
        return {self.encoder_cudagraph_pixel_key: [images[i] for i in indices]}

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
        tower = self.encoder_cudagraph_vision_tower
        m = self.encoder_cudagraph_spatial_merge_size
        num_patches = token_budget * m * m
        input_norm = getattr(tower, "input_norm", None)
        # max_seqlen is baked into the graph, so capture a single sequence
        # spanning the whole buffer to cover any replay.
        metadata = tower.packed_attention_metadata(
            [num_patches], device, num_sequences=max_batch_size
        )
        values: dict[str, torch.Tensor | None] = {
            "patches": torch.zeros(
                num_patches,
                tower.patch_conv.weight[0].numel(),
                device=device,
                dtype=getattr(input_norm, "input_dtype", None) or dtype,
            ),
            # Replays zero-pad positions; (0, 0) is a valid rotary index.
            "positions": torch.zeros(num_patches, 2, dtype=torch.long, device=device),
            "cu_seqlens": metadata["cu_seqlens"],
            "max_seqlen": metadata["max_seqlen"],
            "sequence_lengths": metadata["sequence_lengths"],
        }
        if m > 1:
            values["merge_indices"] = torch.arange(num_patches, device=device).view(
                token_budget, m * m
            )
        return EncoderCudaGraphCaptureInputs(
            values={key: value for key, value in values.items() if value is not None}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        tower = self.encoder_cudagraph_vision_tower
        m = self.encoder_cudagraph_spatial_merge_size
        images = [
            image.to(device=tower.device)
            for image in mm_kwargs[self.encoder_cudagraph_pixel_key]
        ]
        patches, grid_sizes = pack_image_patches(
            images, (tower.patch_size, tower.patch_size)
        )
        metadata = tower.packed_attention_metadata(
            [h * w for h, w in grid_sizes],
            patches.device,
            num_sequences=max_batch_size,
        )
        # max_seqlen is not replayed, keeping the captured upper bound.
        values = {
            "patches": patches,
            "positions": _patch_grid_positions(grid_sizes, patches.device),
            "cu_seqlens": metadata["cu_seqlens"],
            "sequence_lengths": metadata["sequence_lengths"],
        }
        if m > 1:
            values["merge_indices"] = _make_merge_indices(grid_sizes, m, patches.device)
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        inputs: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        features = self.encoder_cudagraph_vision_tower.forward_packed(
            inputs["patches"],
            inputs["positions"],
            inputs["cu_seqlens"],
            inputs["max_seqlen"],
            inputs.get("sequence_lengths"),
        )
        return self.project_packed_image_features(features, inputs.get("merge_indices"))

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        embeddings = cast(SupportsMultiModal, self).embed_multimodal(**mm_kwargs)
        return torch.cat(tuple(embeddings))


@MULTIMODAL_REGISTRY.register_processor(
    PixtralMultiModalProcessor,
    info=PixtralProcessingInfo,
    dummy_inputs=PixtralDummyInputsBuilder,
)
class PixtralForConditionalGeneration(
    nn.Module,
    SupportsLoRA,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
    PixtralEncoderCudaGraphMixin,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_encoder.",
            "model.multi_modal_projector.": "vision_language_adapter.",
        },
        orig_to_new_substr={
            ".linear_1.": ".w_in.",
            ".linear_2.": ".w_out.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supports_tower_connector_lora = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in self.config.vision_config.to_dict().items()
            if key in dataclass_fields
        }

        self.vision_args = VisionEncoderArgs(**vision_args)

        # init MistralForCausalLM
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_encoder = VisionTransformer(
                self.vision_args,
                prefix=maybe_prefix(prefix, "vision_encoder"),
            )
            self.pre_mm_projector_norm = (
                RMSNorm(self.vision_args.hidden_size, eps=1e-5)
                if self.vision_args.add_pre_mm_projector_layer_norm
                else None
            )
            self.patch_merger = (
                PatchMerger(
                    vision_encoder_dim=self.vision_args.hidden_size,
                    spatial_merge_size=self.vision_args.spatial_merge_size,
                    use_mlp_bias=False,
                )
                if self.vision_args.mm_projector_id == PATCH_MERGE
                else None
            )
            self.vision_language_adapter = VisionLanguageAdapter(
                self.vision_args, dim=config.text_config.hidden_size
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> PixtralImagePixelInputs | None:
        images = kwargs.pop("images", None)
        if images is None:
            return None

        return PixtralImagePixelInputs(
            type="pixel_values",
            images=images,
        )

    def _process_image_input(
        self,
        image_input: PixtralImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        images = image_input["images"]
        image_features = self.vision_encoder(images)
        feature_sizes = [image_feature.shape[0] for image_feature in image_features]
        image_features = torch.cat(image_features)
        if self.pre_mm_projector_norm is not None:
            image_features = self.pre_mm_projector_norm(image_features)
        if self.patch_merger is not None:
            patch_size = self.vision_args.patch_size
            spatial_merge_size_square = self.vision_args.spatial_merge_size**2
            img_patch_dims = [
                (img.shape[1] // patch_size, img.shape[2] // patch_size)
                for img in images
            ]
            feature_sizes = [
                feature_size // spatial_merge_size_square
                for feature_size in feature_sizes
            ]
            image_features = self.patch_merger(
                image_features, image_sizes=img_patch_dims
            )
        image_embeds = self.vision_language_adapter(image_features)
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    # -- SupportsEncoderCudaGraph --

    encoder_cudagraph_pixel_key = "images"

    @property
    def encoder_cudagraph_vision_tower(self) -> "VisionTransformer":
        return self.vision_encoder

    @property
    def encoder_cudagraph_spatial_merge_size(self) -> int:
        if self.patch_merger is None:
            return 1
        return self.vision_args.spatial_merge_size

    def project_packed_image_features(
        self, features: torch.Tensor, merge_indices: torch.Tensor | None
    ) -> torch.Tensor:
        if self.pre_mm_projector_norm is not None:
            features = self.pre_mm_projector_norm(features)
        if self.patch_merger is not None:
            assert merge_indices is not None
            features = self.patch_merger.forward_packed(features, merge_indices)
        return self.vision_language_adapter(features)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for pixtral."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def _require_language_model_eagle3(self) -> None:
        if not supports_eagle3(self.language_model):
            raise RuntimeError(
                f"EAGLE-3 speculative decoding requires the language model to "
                f"support EAGLE-3, but {type(self.language_model).__name__} does not."
            )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self._require_language_model_eagle3()
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        self._require_language_model_eagle3()
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        _vision_encoder_stacked_params = [
            # (param_name, shard_name, shard_id)
            # HF format
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # Mistral native (consolidated) format
            (".qkv_proj", ".wq", "q"),
            (".qkv_proj", ".wk", "k"),
            (".qkv_proj", ".wv", "v"),
            (".gate_up_proj", ".w1", 0),
            (".gate_up_proj", ".w3", 1),
        ]

        # Remap Mistral native names to HF-style names
        # used by the vLLM vision encoder modules.
        _vision_encoder_name_remap = {
            ".wo.": ".o_proj.",
            ".w2.": ".down_proj.",
        }

        def is_vision_encoder_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith(("vision_encoder", "vision_tower"))

        def is_vision_lang_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith(
                ("vision_language_adapter", "multi_modal_projector")
            )

        def is_patch_merger(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("patch_merger")

        def is_pre_mm_projector_norm(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("pre_mm_projector_norm")

        vision_encoder_dict = (
            dict(self.vision_encoder.named_parameters())
            if self.vision_encoder is not None
            else {}
        )
        patch_merger_dict = (
            dict(self.patch_merger.named_parameters())
            if self.patch_merger is not None
            else {}
        )
        pre_mm_projector_norm_dict = (
            dict(self.pre_mm_projector_norm.named_parameters())
            if self.pre_mm_projector_norm is not None
            else {}
        )
        vision_lang_adapter_dict = (
            dict(self.vision_language_adapter.named_parameters())
            if self.vision_language_adapter is not None
            else {}
        )

        def llm_weights_generator():
            for name, w in weights:
                if is_vision_encoder_weights((name, w)):
                    if _is_layer_none_or_staged(self.vision_encoder):
                        continue
                    trimmed_name = ".".join(name.split(".")[1:])
                    for (
                        param_name,
                        weight_name,
                        shard_id,
                    ) in _vision_encoder_stacked_params:
                        if weight_name in trimmed_name:
                            trimmed_name = trimmed_name.replace(weight_name, param_name)
                            param = vision_encoder_dict[trimmed_name]
                            weight_loader = param.weight_loader
                            weight_loader(param, w, shard_id)
                            break
                    else:
                        for old, new in _vision_encoder_name_remap.items():
                            if old in trimmed_name:
                                trimmed_name = trimmed_name.replace(old, new)
                                break

                        param = vision_encoder_dict.get(trimmed_name)
                        if param is not None:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, w)
                elif is_patch_merger((name, w)):
                    if _is_layer_none_or_staged(self.patch_merger):
                        continue
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = patch_merger_dict[trimmed_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, w)
                elif is_pre_mm_projector_norm((name, w)):
                    if _is_layer_none_or_staged(self.pre_mm_projector_norm):
                        continue
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = pre_mm_projector_norm_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_vision_lang_adapter_weights((name, w)):
                    if _is_layer_none_or_staged(self.vision_language_adapter):
                        continue
                    trimmed_name = ".".join(name.split(".")[1:])
                    param = vision_lang_adapter_dict.get(trimmed_name)
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, w)
                else:
                    name = name.removeprefix("language_model.")
                    yield (name, w)

        self.language_model.load_weights(llm_weights_generator())

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="vision_language_adapter.",
            tower_model="vision_encoder",
        )

    def get_mm_lora_token_counts(
        self,
        *,
        modality: str,
        mm_kwargs: MultiModalKwargsItem | None,
        num_mm_embeds: int,
    ) -> tuple[int, int | None]:
        del modality, mm_kwargs
        if getattr(self, "patch_merger", None) is None:
            return num_mm_embeds, num_mm_embeds
        merge_size = self.vision_args.spatial_merge_size
        return num_mm_embeds * (merge_size**2), num_mm_embeds


# Vision encoder
@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
    adapter_bias: bool = True
    spatial_merge_size: int = 1
    add_pre_mm_projector_layer_norm: bool = False
    mm_projector_id: str = ""


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
    to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def _apply_rotary_emb_vit_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.view_as_complex(freqs_cis)[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _apply_rotary_emb_vit_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cis[None, :, None, :, 0]
    sin = freqs_cis[None, :, None, :, 1]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_re, x_im = x_[..., 0], x_[..., 1]
        rotated = torch.stack(
            (x_re * cos - x_im * sin, x_re * sin + x_im * cos), dim=-1
        )
        return rotated.flatten(3).type_as(x)

    return rotate(xq), rotate(xk)


def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate `xq`/`xk` of shape (bsz, seq_len, n_heads, head_dim).

    `freqs_cis` is the real view (seq_len, head_dim / 2, 2) of the complex
    frequencies.
    """
    assert freqs_cis.dtype == torch.float32
    assert freqs_cis.shape == (xq.shape[1], xq.shape[-1] // 2, 2), (
        freqs_cis.shape,
        xq.shape,
    )
    if torch.compiler.is_compiling():
        # Inductor does not generate code for complex operators and would
        # fall back to eager around RoPE.
        return _apply_rotary_emb_vit_real(xq, xk, freqs_cis)
    # In eager mode the complex multiply is a single kernel per tensor, while
    # the real formulation launches several elementwise kernels.
    return _apply_rotary_emb_vit_complex(xq, xk, freqs_cis)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.w13",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.w2",
        )

        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = maybe_fused_act_quant(self.act_fn, x, self.down_proj)
        x, _ = self.down_proj(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        disable_tp: bool = False,
    ):
        super().__init__()
        self.args = args
        assert not args.hidden_size % args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=args.hidden_size,
            head_size=self.head_dim,
            total_num_heads=args.num_attention_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=disable_tp,
        )
        self.o_proj = RowParallelLinear(
            input_size=args.hidden_size,
            output_size=args.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=disable_tp,
        )

        tp_size = 1 if disable_tp else get_tensor_model_parallel_world_size()
        self.n_heads = divide(args.num_attention_heads, tp_size)
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        out = self.attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


@support_torch_compile(
    dynamic_arg_dims={
        "x": 1,
        "freqs_cis": 0,
        "cu_seqlens": 0,
        "sequence_lengths": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class TransformerBlock(nn.Module):
    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        disable_tp: bool = False,
    ):
        super().__init__()
        self.attention = Attention(
            args,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
            disable_tp=disable_tp,
        )
        self.feed_forward = FeedForward(
            args.hidden_size,
            args.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
            disable_tp=disable_tp,
        )
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        r = self.attention.forward(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        disable_tp: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for idx in range(args.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    args,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                    disable_tp=disable_tp,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                freqs_cis=freqs_cis,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.args = args
        disable_tp = is_vit_use_data_parallel()
        self.patch_conv = Conv2dLayer(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(
            args,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer",
            disable_tp=disable_tp,
        )

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: torch.Tensor | None = None
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            args.num_attention_heads, head_dim
        )

    @property
    def patch_size(self) -> int:
        return self.args.patch_size

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.types.Device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = torch.view_as_real(
                precompute_freqs_cis_2d(
                    dim=self.args.hidden_size // self.args.num_attention_heads,
                    height=self.max_patches_per_side,
                    width=self.max_patches_per_side,
                    theta=self.args.rope_theta,
                )
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    @property
    def encoder_attention(self) -> MMEncoderAttention:
        """Attention of the first layer; all layers share its settings."""
        return cast(TransformerBlock, self.transformer.layers[0]).attention.attn

    @property
    def attn_backend(self) -> AttentionBackendEnum:
        return self.encoder_attention.attn_backend

    def packed_attention_metadata(
        self,
        sequence_lengths: list[int],
        device: torch.device,
        *,
        num_sequences: int | None = None,
    ) -> PackedAttentionMetadata:
        cu_seqlens, max_seqlen, packed_sequence_lengths = (
            _make_packed_sequence_metadata(
                sequence_lengths,
                self.attn_backend,
                self.args.hidden_size,
                1
                if is_vit_use_data_parallel()
                else get_tensor_model_parallel_world_size(),
                device,
                num_sequences=num_sequences,
                fp8_padded_hidden_size=self.fp8_padded_hidden_size,
            )
        )
        return {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "sequence_lengths": packed_sequence_lengths,
        }

    def forward_packed(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode images packed by `pack_image_patches`.

        `patches` holds (N_toks, C * patch_size**2) pixel patches and
        `positions` their (N_toks, 2) (row, col) coordinates; the attention
        arguments come from `packed_attention_metadata`. Returns (N_toks, D)
        token features.
        """
        weight = self.patch_conv.weight
        patch_embeds = F.linear(
            patches.to(weight.dtype), weight.reshape(weight.shape[0], -1)
        )
        patch_embeds = self.ln_pre(patch_embeds.unsqueeze(0))
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]
        out = self.transformer(
            patch_embeds,
            freqs_cis=freqs_cis,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        return out.squeeze(0)

    def forward(
        self,
        images: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            images: list of N_img images of variable sizes,
                each of shape (C, H, W)

        Returns:
            image_features: per-image token features, each of shape
                (N_toks_i, D)

        """
        patch_size = self.args.patch_size
        patches, grid_sizes = pack_image_patches(images, (patch_size, patch_size))
        embed_sizes = [h * w for h, w in grid_sizes]
        out = self.forward_packed(
            patches,
            _patch_grid_positions(grid_sizes, patches.device),
            **self.packed_attention_metadata(embed_sizes, patches.device),
        )
        return torch.split(out, embed_sizes)


class VisionLanguageAdapter(nn.Module):
    def __init__(self, args: VisionEncoderArgs, dim: int):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        self.w_in = ReplicatedLinear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
            return_bias=False,
        )
        self.gelu = nn.GELU()
        self.w_out = ReplicatedLinear(
            dim, dim, bias=args.adapter_bias, return_bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class PatchMerger(nn.Module):
    """Learned merging of spatial_merge_size ** 2 patches."""

    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
        use_mlp_bias: bool = False,
    ) -> None:
        super().__init__()

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = ReplicatedLinear(
            mlp_input_dim, vision_encoder_dim, bias=use_mlp_bias, return_bias=False
        )

    def forward(
        self, x: torch.Tensor, image_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == len(x)

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2,
        #       vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def forward_packed(
        self, x: torch.Tensor, merge_indices: torch.Tensor
    ) -> torch.Tensor:
        """Same as `forward`, gathering blocks with `_make_merge_indices`."""
        return self.merging_layer(_merge_patches_by_index(x, merge_indices))

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)

        """
        sub_grids = get_sub_grids(
            x=x, image_sizes=image_sizes, spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor: list[torch.Tensor] = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted_tensor.append(
                grid.view(-1, n_patches).t()
            )  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(
            permuted_tensor, dim=0
        )  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1)[
            None, :, :, :
        ]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(
            image_grid, kernel_size=sub_grid_size, stride=sub_grid_size
        )
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size, -1
        )  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids


#### HF Transformers version of Pixtral ####
# Based off https://github.com/huggingface/transformers/blob/d7950bff82b18c823193d17d72188c5e46d06c83/src/transformers/models/pixtral/modeling_pixtral.py
# This model follows the Llava family, meaning image embeddings are placed
# instead of the `[IMG]` token placeholders.
# The model uses [`PixtralVisionModel`] for its vision encoder,
# and [`MistralForCausalLM`] for its language decoder.


class PixtralHFEncoderInfo(VisionEncoderInfo[PixtralVisionConfig]):
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        ncols, nrows = self.get_patch_grid_size(
            image_width=image_width,
            image_height=image_height,
        )
        return ncols * nrows

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        # spatial_merge_size is needed for Mistral3
        spatial_merge_size = getattr(self.hf_config, "spatial_merge_size", 1)
        return self.vision_config.patch_size * spatial_merge_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()

        # Since interpolation is applied, the image size need not be divisible
        # assert image_size % patch_size == 0
        return image_size // patch_size

    # Adapted from: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/pixtral/image_processing_pixtral.py#L99
    def get_patch_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_width = max_height = self.get_image_size()
        patch_width = patch_height = self.get_patch_size()

        ratio = max(image_width / max_width, image_height / max_height)

        if ratio > 1:
            image_width = int(math.floor(image_width / ratio))
            image_height = int(math.floor(image_height / ratio))

        nrows, ncols = _get_pixtral_hf_num_image_tokens(
            (image_height, image_width),
            (patch_height, patch_width),
        )  # type: ignore

        return ncols, nrows


class PixtralHFMLP(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        use_data_parallel = is_vit_use_data_parallel()

        assert config.intermediate_size is not None
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_and_mul = get_act_and_mul_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = maybe_fused_act_quant(self.act_and_mul, gate_up, self.down_proj)
        x, _ = self.down_proj(x)
        return x


class PixtralHFAttention(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        assert not config.hidden_size % config.num_attention_heads
        self.total_num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.total_num_heads * self.head_dim == config.hidden_size

        use_data_parallel = is_vit_use_data_parallel()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.o_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=use_data_parallel,
        )

        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.n_heads = divide(config.num_attention_heads, self.tp_size)
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, patches, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv_states.chunk(3, dim=-1)

        # Transpose q and k to apply HF's Rotary Position Embedding.
        q = q.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, patches, self.n_heads, self.head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        out = self.attn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        attn_output, _ = self.o_proj(out)

        return attn_output, None


@support_torch_compile(
    dynamic_arg_dims={
        "hidden_states": 1,
        "rotary_pos_emb_cos": 0,
        "rotary_pos_emb_sin": 0,
        "cu_seqlens": 0,
        "sequence_lengths": 0,
    },
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class PixtralHFTransformerBlock(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.attention = PixtralHFAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.feed_forward = PixtralHFMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        r, _ = self.attention.forward(
            self.attention_norm(hidden_states),
            position_embeddings=(rotary_pos_emb_cos, rotary_pos_emb_sin),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        h = hidden_states + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class PixtralHFTransformer(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                PixtralHFTransformerBlock(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
        return_all_hidden_states: bool,
    ) -> torch.Tensor:
        hidden_states_pool = [x]
        rotary_pos_emb_cos, rotary_pos_emb_sin = position_embeddings

        for layer in self.layers:
            x = layer(
                x,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                cu_seqlens,
                max_seqlen,
                sequence_lengths,
            )
            if return_all_hidden_states:
                hidden_states_pool.append(x)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return x


def pack_image_patches(
    images: Sequence[torch.Tensor],
    patch_shape: tuple[int, int],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Pack CHW images into one (num_patches, C * ph * pw) tensor.

    Patches are laid out row-major per image, with each row flattened in the
    (C, ph, pw) order of a patch convolution weight. Returns the packed tensor
    and the (rows, cols) patch grid of every image.
    """
    patch_height, patch_width = patch_shape
    in_channels = images[0].shape[0]
    grid_sizes = []
    for image in images:
        assert image.ndim == 3 and image.shape[0] == in_channels
        _, height, width = image.shape
        grid_sizes.append((height // patch_height, width // patch_width))

    patch_counts = [rows * cols for rows, cols in grid_sizes]
    packed = torch.empty(
        (sum(patch_counts), in_channels * patch_height * patch_width),
        dtype=images[0].dtype,
        device=images[0].device,
    )
    offset = 0
    for image, (rows, cols), count in zip(images, grid_sizes, patch_counts):
        source = (
            image[:, : rows * patch_height, : cols * patch_width]
            .reshape(in_channels, rows, patch_height, cols, patch_width)
            .permute(1, 3, 0, 2, 4)
        )
        packed[offset : offset + count].view(
            rows, cols, in_channels, patch_height, patch_width
        ).copy_(source)
        offset += count
    return packed, grid_sizes


def pixtral_patch_embed(
    images: list[torch.Tensor],
    weight: torch.Tensor,
    input_norm: nn.Module,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Normalize and project all CHW images in one matrix multiplication."""
    out_channels, in_channels, patch_height, patch_width = weight.shape
    assert images[0].shape[0] == in_channels
    packed, grid_sizes = pack_image_patches(images, (patch_height, patch_width))
    normalized = input_norm(packed, weight.dtype)
    projected = F.linear(normalized, weight.reshape(out_channels, -1))
    patch_counts = [rows * cols for rows, cols in grid_sizes]
    embeddings = [
        tokens.transpose(0, 1).reshape(1, out_channels, rows, cols)
        for tokens, (rows, cols) in zip(projected.split(patch_counts), grid_sizes)
    ]
    return projected.unsqueeze(0), embeddings


class PixtralHFVisionModel(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        input_norm: nn.Module | None = None,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.input_norm = input_norm

        self.patch_conv = Conv2dLayer(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralHFTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.transformer",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} "
                "layers."
            )

        if require_post_norm is True:
            msg = "PixtralHFVisionModel does not have post-layernorm"
            raise ValueError(msg)

        self.dtype = next(self.parameters()).dtype
        self.device = next(self.parameters()).device
        self.patch_positional_embedding = PixtralVisionRotaryEmbedding(config).to(
            self.device
        )
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            config.num_attention_heads, config.hidden_size // config.num_attention_heads
        )

    @property
    def patch_size(self) -> int:
        return cast(int, self.config.patch_size)

    @property
    def encoder_attention(self) -> MMEncoderAttention:
        """Attention of the first layer; all layers share its settings."""
        return self._first_attention.attn

    @property
    def _first_attention(self) -> "PixtralHFAttention":
        return cast(PixtralHFTransformerBlock, self.transformer.layers[0]).attention

    @property
    def attn_backend(self) -> AttentionBackendEnum:
        return self.encoder_attention.attn_backend

    def packed_attention_metadata(
        self,
        sequence_lengths: list[int],
        device: torch.device,
        *,
        num_sequences: int | None = None,
    ) -> PackedAttentionMetadata:
        cu_seqlens, max_seqlen, packed_sequence_lengths = (
            _make_packed_sequence_metadata(
                sequence_lengths,
                self.attn_backend,
                self.config.hidden_size,
                self._first_attention.tp_size,
                device,
                num_sequences=num_sequences,
                fp8_padded_hidden_size=self.fp8_padded_hidden_size,
            )
        )
        return {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "sequence_lengths": packed_sequence_lengths,
        }

    def forward_packed(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        """Encode images packed by `pack_image_patches`.

        `patches` holds (N_toks, C * patch_size**2) pixel patches (raw pixels
        when `input_norm` normalizes on device) and `positions` their
        (N_toks, 2) (row, col) coordinates; the attention arguments come from
        `packed_attention_metadata`. Returns (N_toks, D) token features.
        """
        weight = self.patch_conv.weight
        if self.input_norm is not None:
            patches = self.input_norm(patches, weight.dtype)
        else:
            patches = patches.to(weight.dtype)
        patch_embeds = F.linear(patches, weight.reshape(weight.shape[0], -1))
        patch_embeds = self.ln_pre(patch_embeds.unsqueeze(0))

        if not TRANSFORMERS_WITH_AXIAL_ROPE:
            max_width = self.config.image_size // self.config.patch_size
            positions = positions[:, 0] * max_width + positions[:, 1]
        position_embedding = self.patch_positional_embedding(patch_embeds, positions)

        out = self.transformer(
            patch_embeds,
            position_embedding,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
            return_all_hidden_states=select_layers is not None,
        )

        out = resolve_visual_encoder_outputs(
            out,
            None,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
            feature_select_strategy=feature_select_strategy,
        )
        return out.squeeze(0)

    def forward(
        self,
        pixel_values: list[torch.Tensor],
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This means it will be a list of tensors
                because multiple requests batched can have multiple images,
                each with their own shape potentially
            select_layers: Layer indices whose features should be
                concatenated and used as the visual encoder output. If none
                are provided, the last layer is used.

        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)

        """
        images = [image.to(device=self.device) for image in pixel_values]
        patches, grid_sizes = pack_image_patches(
            images, (self.patch_size, self.patch_size)
        )
        embed_sizes = [h * w for h, w in grid_sizes]
        out = self.forward_packed(
            patches,
            _patch_grid_positions(grid_sizes, self.device),
            **self.packed_attention_metadata(embed_sizes, self.device),
            select_layers=select_layers,
            feature_select_strategy=feature_select_strategy,
        )

        # split into separate tensors for each image
        return torch.split(out, embed_sizes)

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.transformer.layers)

        for name, loaded_weight in weights:
            # omit layers when num_hidden_layers_override is set
            if name.startswith("transformer.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
