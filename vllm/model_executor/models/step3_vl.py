# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.step3_vl import Step3VisionEncoderConfig
from vllm.transformers_utils.processors.step3_vl import (
    MAX_IMAGE_SIZE,
    Step3VLImageProcessor,
    Step3VLProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import is_vit_use_data_parallel, run_dp_sharded_vision_model


class Step3VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height
        - w: Width
        - bnp: Batch size * number of images * number of patches
        - hp: Height of patch
        - wp: Width of patch
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]
    patch_pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "hp", "wp")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class Step3VLImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor, TensorShape("bn", "f", "h")]


Step3VLImageInputs: TypeAlias = Step3VLImagePixelInputs | Step3VLImageEmbeddingInputs


class Step3VLProcessingInfo(BaseProcessingInfo):
    def get_image_processor(self, **kwargs):
        config = self.get_hf_config()

        kwargs.setdefault(
            "enable_patch",
            getattr(config.vision_config, "enable_patch", True),
        )

        return Step3VLImageProcessor(**kwargs)

    def get_hf_processor(self) -> Step3VLProcessor:
        return Step3VLProcessor(
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_max_image_tokens(self) -> int:
        hf_processor = self.get_hf_processor()
        target_width, target_height = self.get_image_size_with_most_features()

        return hf_processor.image_processor.get_num_image_tokens(
            target_width, target_height
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)


class Step3VLDummyInputsBuilder(BaseDummyInputsBuilder[Step3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<im_patch>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class Step3VLMultiModalProcessor(BaseMultiModalProcessor[Step3VLProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_placeholder_token_id = hf_processor.image_token_id

        def get_replacement_step1o(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            num_patches = int(out_item["num_patches"].data)
            if num_patches > 0:
                patch_newline_mask = out_item["patch_newline_mask"].data
                image_repl_ids = hf_processor._get_image_repl_feature_ids(
                    1, num_patches, patch_newline_mask.tolist()
                )
            else:
                image_repl_ids = hf_processor._get_image_repl_feature_ids(1, 0, None)

            return PromptUpdateDetails.select_token_id(
                seq=image_repl_ids,
                embed_token_id=image_placeholder_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[image_placeholder_token_id],
                replacement=get_replacement_step1o,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            patch_pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
            num_patches=MultiModalFieldConfig.batched("image"),
            patch_newline_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
        )


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = (
            old_pos_embed.view(1, src_size, src_size, dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


class Step3VisionEmbeddings(nn.Module):
    def __init__(self, config: Step3VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, self.embed_dim))

        self.patch_embedding = Conv2dLayer(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.pad_tp_size = 4  # hard code for padding
        # To load the pretrained weights, we still use P+1 as the seqlen
        self.position_embedding = torch.nn.Embedding(
            self.num_patches + 1, self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches + 1).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # pad
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + get_abs_pos(
            self.position_embedding(self.position_ids), patch_embeds.size(1)
        )
        embeddings = torch.cat(
            [
                embeddings[:, 0, :].unsqueeze(1).repeat(1, self.pad_tp_size - 1, 1),
                embeddings,
            ],
            dim=1,
        )
        return embeddings


class Step3VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.total_num_heads

        self.scale = self.head_dim**-0.5

        use_data_parallel = is_vit_use_data_parallel()
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.q_size = self.num_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        # Use unified MMEncoderAttention with automatic backend selection
        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        # Use unified MMEncoderAttention with automatic backend selection
        attn_output = self.attn(q, k, v)

        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class Step3VisionMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        use_data_parallel = is_vit_use_data_parallel()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Step3VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Step3VisionAttention(
            config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Step3VisionMLP(
            config,
            quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states + self.layer_norm1(self.self_attn(hidden_states))
        hidden_states = hidden_states + self.layer_norm2(self.mlp(hidden_states))
        return hidden_states


class Step3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Step3VisionEncoderLayer(
                    config,
                    quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class Step3VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Step3VisionEncoderConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.use_data_parallel = is_vit_use_data_parallel()
        self.image_size = config.image_size
        self.embeddings = Step3VisionEmbeddings(config)
        self.transformer = Step3VisionEncoder(
            config,
            quant_config,
            prefix=f"{prefix}.transformer",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        hidden_states = self.embeddings(pixel_values)
        if self.use_data_parallel:
            hidden_states = run_dp_sharded_vision_model(hidden_states, self.transformer)
        else:
            hidden_states = self.transformer(inputs_embeds=hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Step3VLMultiModalProcessor,
    info=Step3VLProcessingInfo,
    dummy_inputs=Step3VLDummyInputsBuilder,
)
class Step3VLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<im_patch>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # NOTE: This behavior is consistent with the previous OOV handling,
        # but does not currently handle the start/stop toks around the
        # image features (<patch_start> <patch_end> <im_start> <im_end>)
        # See: https://huggingface.co/stepfun-ai/step3/blob/main/processing_step3v.py#L323
        #
        # If this becomes an issue or we refactor to handle this using the
        # processor info in the future, it would probably be best to handle
        # those too.
        self.configure_mm_token_handling(
            self.config.text_config.vocab_size,
            [self.config.image_token_id],
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = Step3VisionTransformer(
                config.vision_config,
                None,
                prefix=maybe_prefix(prefix, "vision_model"),
            )
            self.vit_downsampler = Conv2dLayer(
                config.vision_config.hidden_size,
                config.vision_config.output_hidden_size,
                kernel_size=2,
                stride=config.understand_projector_stride,
            )
            self.vit_downsampler2 = Conv2dLayer(
                config.vision_config.output_hidden_size,
                config.vision_config.output_hidden_size * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.vit_large_projector = nn.Linear(
                config.vision_config.output_hidden_size * 2,
                config.hidden_size,
                bias=config.projector_bias,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Step3VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and patch_pixel_values is not None:
            return Step3VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values.to(self.dtype),
                patch_pixel_values=patch_pixel_values.to(self.dtype),
                num_patches=num_patches,
            )

        if image_embeds is not None:
            return Step3VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds.to(self.dtype),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(sqrt(P))
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler(image_features)
        image_features = self.vit_downsampler2(image_features)
        n_dim = image_features.size(1)
        image_features = image_features.view(B, n_dim, -1).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def _get_vision_model_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)[:, 4:]

    def _process_image_input(
        self, image_input: Step3VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            image_features = image_input["image_embeds"]
        else:
            image_features = self._get_vision_model_output(image_input["pixel_values"])
            patch_image_features = (
                self._get_vision_model_output(image_input["patch_pixel_values"])
                if len(image_input["patch_pixel_values"]) > 0
                else None
            )
            num_patches = image_input["num_patches"]

        image_features = self._process_image_features(image_features)
        patch_image_features = (
            self._process_image_features(patch_image_features)
            if patch_image_features is not None
            else None
        )

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx : cur_patch_idx + num_patch
                ]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]
            )
        return merged_image_features

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
