# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/zai-org/CogAgent
"""Inference-only CogAgent model compatible with THUDM weights."""

import itertools
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import torch
from torch import nn
from torch.nn import LayerNorm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature, PretrainedConfig, PreTrainedTokenizer, TensorType
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import TextInput

from vllm.attention.layer import MultiHeadAttention
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import ChatGLMConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .chatglm import ChatGLMBaseModel, ChatGLMModel
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)


class GLMVImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - c: Number of channels (3)
        - h: Height of image
        - w: Width of image
    """

    type: Literal["pixel_values"] = "pixel_values"
    data: Annotated[torch.Tensor, TensorShape("b", 3, "h", "w")]


class EVA2CLIPPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        images = images.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class EVA2CLIPAttention(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_rank = config.num_heads // self.tp_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value",
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.attn = MultiHeadAttention(
            self.num_heads_per_rank, self.head_dim, self.scale
        )
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.query_key_value(x)  # B, L, 3 * H * D
        q, k, v = qkv.chunk(3, dim=-1)

        out = self.attn(q, k, v)
        output, _ = self.dense(out)
        output = self.output_dropout(output)
        return output


class EVA2CLIPMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x, _ = self.fc2(x)
        return x


class EVA2CLIPTransformerLayer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = EVA2CLIPAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.attention"
        )
        self.mlp = EVA2CLIPMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class EVA2CLIPTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EVA2CLIPTransformerLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class EVA2CLIPGLU(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """
        The original implementation is the same as:
        ```python
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        ```
        ```
        gate_proj_output, _ = self.gate_proj(x)
        dense_h_to_4h_output, _ = self.dense_h_to_4h(x)
        x = torch.cat([gate_proj_output, dense_h_to_4h_output], dim=-1)
        ```

        We merge two ColumnParallelLinear into one MergedColumnParallelLinear:
        ```
        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        ```
        ```
        x, _ = self.merged_proj(x)
        ```
        """
        super().__init__()
        self.linear_proj = ReplicatedLinear(
            in_features,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_proj",
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = SiluAndMul()

        self.merged_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.merged_proj",
        )

        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
        )

    def forward(self, x):
        x, _ = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x, _ = self.merged_proj(x)
        x = self.act2(x)
        x, _ = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = EVA2CLIPPatchEmbedding(vision_config)
        self.transformer = EVA2CLIPTransformer(
            vision_config, quant_config=quant_config, prefix=f"{prefix}.transformer"
        )
        self.linear_proj = EVA2CLIPGLU(
            config,
            in_features=config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_proj",
        )
        self.conv = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=2,
            stride=2,
        )
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        images : torch.Tensor
            Input image tensor with shape (B, C, H, W)

        Returns:
        torch.Tensor
            Transformed tensor with shape (B, L, D)
        """
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x


class GLM4VModel(ChatGLMModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = vllm_config.quant_config

        self.vision = EVA2CLIPModel(
            self.config, quant_config, prefix=f"{prefix}.vision"
        )


class GLM4VProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.
    """

    def __init__(
        self,
        config: ChatGLMConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        vision_config = config.vision_config
        image_size = vision_config["image_size"]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | list[ImageInput] | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        text_inputs = self.tokenizer(text)

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values = [self.image_transform(image) for image in images]
            image_inputs = {"pixel_values": torch.stack(pixel_values)}

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


class GLM4VProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(ChatGLMConfig)

    def get_hf_processor(self, **kwargs: object) -> GLM4VProcessor:
        return self.ctx.init_processor(
            GLM4VProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        image_size = vision_config["image_size"]
        patch_size = vision_config["patch_size"]
        grid_length = image_size // patch_size // 2
        return grid_length * grid_length

    def get_num_image_feature_tokens(self) -> int:
        # EVA2CLIPModel has embeddings for boi and eoi tokens as well
        return self.get_num_image_tokens() + 2


class GLM4VDummyInputsBuilder(BaseDummyInputsBuilder[GLM4VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        base_text = "<|begin_of_image|><|endoftext|><|end_of_image|>"

        return base_text * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config

        target_width = target_height = vision_config["image_size"]
        num_images = mm_counts.get("image", 0)

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class GLM4VMultiModalProcessor(BaseMultiModalProcessor[GLM4VProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

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
        hf_config = self.info.get_hf_config()

        boi_token_id = hf_config.boi_token_id
        image_token_id = hf_config.pad_token_id
        eoi_token_id = hf_config.eoi_token_id

        def get_replacement(item_idx: int):
            num_image_tokens = self.info.get_num_image_tokens()
            image_tokens = [image_token_id] * num_image_tokens

            return [boi_token_id] + image_tokens + [eoi_token_id]

        return [
            PromptReplacement(
                modality="image",
                target=[boi_token_id, image_token_id, eoi_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GLM4VMultiModalProcessor,
    info=GLM4VProcessingInfo,
    dummy_inputs=GLM4VDummyInputsBuilder,
)
class GLM4VForCausalLM(
    ChatGLMBaseModel, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    merge_by_field_config = True

    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"],
        "merged_proj": ["gate_proj", "dense_h_to_4h"],
    }

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="transformer.encoder",
            connector="transformer.vision.linear_proj",
            tower_model="transformer.vision.transformer",
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|begin_of_image|><|endoftext|><|end_of_image|>"

        raise ValueError("Only image modality is supported")

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        transformer_type: type[GLM4VModel] = GLM4VModel,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            transformer_type=transformer_type,
        )

        self.transformer: GLM4VModel

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> GLMVImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is not None:
            expected_h = expected_w = self.config.vision_config["image_size"]
            return GLMVImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        return None

    def _process_image_input(self, image_input: GLMVImagePixelInputs) -> torch.Tensor:
        pixel_values = image_input["data"].to(dtype=self.config.dtype)

        return self.transformer.vision(pixel_values)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor,
        video_grid_thw: list[list[int]] | torch.Tensor,
        context_len: int = 0,
        seq_len: int | None = None,
        second_per_grid_ts: list[float] | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value for GLM4V."""

        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        llm_pos_ids_list: list = []

        if not (image_grid_thw is None and video_grid_thw is None):
            if isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = image_grid_thw.tolist()

            input_token_type: list[str] = []
            video_check_flg = False
            for token in input_tokens:
                if token == video_start_token_id:
                    video_check_flg = True
                elif token == video_end_token_id:
                    video_check_flg = False

                if (token == image_token_id) and (video_check_flg is False):
                    input_token_type.append("image")
                elif (token == image_token_id) and (video_check_flg is True):
                    input_token_type.append("video")
                else:
                    input_token_type.append("text")

            input_type_group: list[tuple[str, int, int]] = []
            for key, group_iter in itertools.groupby(
                enumerate(input_token_type), lambda x: x[1]
            ):
                group_list = list(group_iter)
                start_index = group_list[0][0]
                end_index = group_list[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            video_frame_num = 1
            mm_data_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                if modality_type == "image":
                    t, h, w = (
                        image_grid_thw[mm_data_idx][0],
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + st_idx
                    )
                    mm_data_idx += 1

                elif modality_type == "video":
                    t, h, w = (
                        video_frame_num,
                        image_grid_thw[mm_data_idx][1],
                        image_grid_thw[mm_data_idx][2],
                    )
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )

                    for t_idx in range(llm_grid_t):
                        t_index = (
                            torch.tensor(t_idx)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            .expand(1, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            .expand(1, llm_grid_h, -1)
                            .flatten()
                        )
                        llm_pos_ids_list.append(
                            torch.stack([t_index, h_index, w_index]) + st_idx
                        )

                    mm_data_idx += 1
                    video_frame_num += 1

                else:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    video_frame_num = 1

        else:
            text_len = len(input_tokens)
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        llm_positions = llm_positions[:, context_len:seq_len]
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return llm_positions, mrope_position_delta

    def get_language_model(self) -> torch.nn.Module:
        return self.transformer

    get_input_embeddings = SupportsMultiModal.get_input_embeddings

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states
