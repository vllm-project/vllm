# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from collections.abc import Iterable, Mapping
from itertools import tee
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature, Llama4Config, Llama4VisionConfig
from transformers.image_utils import SizeDict
from transformers.models.llama4 import Llama4Processor
from transformers.models.llama4.image_processing_llama4_fast import (
    find_supported_resolutions,
    get_best_fit,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.vision import should_torch_compile_mm_vit
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .llama4 import Llama4ForCausalLM
from .utils import AutoWeightsLoader, StageMissingLayer, maybe_prefix
from .vision import is_vit_use_data_parallel, run_dp_sharded_vision_model


class Llama4ImagePatchInputs(TensorSchema):
    """
    Dimensions:
        - batch_size: Batch size
        - total_num_chunks: Batch size * number of chunks
        - num_channels: Number of channels
        - image_size: Size of each image
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("total_num_chunks", "num_channels", "image_size", "image_size"),
    ]

    patches_per_image: Annotated[torch.Tensor, TensorShape("batch_size")]
    """
    The number of total patches for each image in the batch.
    
    This is used to split the embeddings which has the first two dimensions
    flattened just like `pixel_values`.
    """

    aspect_ratios: Annotated[torch.Tensor, TensorShape("batch_size", 2)]
    """
    A list of aspect ratios corresponding to the number of tiles
    in each dimension that each image in the batch corresponds to.
    Each aspect ratio is a pair (ratio_h, ratio_w).
    """


class Llama4VisionMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        bias: bool,
        output_activation: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.fc1 = ColumnParallelLinear(
            input_size=input_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )
        self.activation_fn = nn.GELU()
        self.output_activation = output_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        if self.output_activation:
            return self.activation_fn(hidden_states)
        return hidden_states


class Llama4MultiModalProjector(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            input_size=config.vision_config.vision_output_dim,
            output_size=config.text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            gather_output=True,
            prefix=f"{prefix}.linear_1",
        )

    def forward(self, image_features):
        hidden_states, _ = self.linear_1(image_features)
        return hidden_states


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.size()

    reshaped_tensor = input_tensor.view(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(
            config.projector_input_dim // (self.pixel_shuffle_ratio**2)
        )
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP(
            input_size=config.intermediate_size,
            intermediate_size=config.projector_input_dim,
            output_size=config.projector_output_dim,
            bias=config.multi_modal_projector_bias,
            output_activation=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


class Llama4VisionAttention(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        assert self.num_heads % self.tp_size == 0
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.attn = MMEncoderAttention(
            self.num_local_heads,
            self.head_dim,
            self.scaling,
            prefix=prefix,
        )

        if use_data_parallel:
            self.qkv_proj = ReplicatedLinear(
                self.embed_dim,
                self.q_size + 2 * self.kv_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.head_dim,
                self.embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                self.embed_dim,
                self.head_dim,
                self.num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = RowParallelLinear(
                self.num_heads * self.head_dim,
                self.embed_dim,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

        rope_parameters = {
            "rope_type": "mllama4",
            "rope_theta": config.rope_parameters["rope_theta"],
            "partial_rotary_factor": 0.5,
        }

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            # number of image patches
            max_position=(config.image_size // config.patch_size) ** 2,
            rope_parameters=rope_parameters,
            is_neox_style=False,
            dtype=torch.complex64,  # important
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(q.shape[0], q.shape[1], self.num_local_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_local_heads, self.head_dim)
        q, k = self.rotary_emb(q, k)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output, _ = self.o_proj(attn_output)

        return attn_output


class Llama4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.self_attn = Llama4VisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Llama4VisionMLP(
            input_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            output_activation=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_state: torch.Tensor,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state)
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)
        return outputs


class Llama4VisionEncoder(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states: Input tensor of shape
                (batch_size, sequence_length, hidden_size).
                Hidden states from the model embeddings, representing
                the input tokens.
                associated vectors than the model's internal embedding
                lookup matrix.
        """

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]

        return hidden_states


class Llama4UnfoldConvolution(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        use_data_parallel = is_vit_use_data_parallel()
        self.linear = ColumnParallelLinear(
            input_size=config.num_channels * kernel_size[0] * kernel_size[1],
            output_size=config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states, _ = self.linear(hidden_states)
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={"images_flattened": 0}, enable_if=should_torch_compile_mm_vit
)
class Llama4VisionModel(nn.Module):
    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.patch_embedding",
        )

        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.hidden_size)
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # encoders
        self.model = Llama4VisionEncoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.model",
        )

        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            config,
            quant_config,
            prefix=f"{prefix}.vision_adapter",
        )

    def forward(
        self,
        images_flattened: torch.Tensor,
    ) -> torch.Tensor:
        # Patch embedding
        hidden_state = self.patch_embedding(images_flattened)
        num_tiles, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        class_embedding = self.class_embedding.expand(
            hidden_state.shape[0], 1, hidden_state.shape[-1]
        )
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            num_tiles,
            1,
            num_patches,
            hidden_dim,
        )
        positional_embedding = self.positional_embedding_vlm.to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )
        hidden_state = hidden_state + positional_embedding
        hidden_state = self.layernorm_pre(hidden_state)
        hidden_state = hidden_state.view(num_tiles, -1, hidden_dim)

        # Apply encoder
        hidden_state = self.model(hidden_state)
        hidden_state = self.layernorm_post(hidden_state)

        # Remove CLS token output
        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        return hidden_state


class Mllama4ProcessingInfo(BaseProcessingInfo):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

    def get_hf_config(self) -> Llama4Config:
        return self.ctx.get_hf_config(Llama4Config)

    def get_hf_processor(self, **kwargs: object) -> Llama4Processor:
        return self.ctx.get_hf_processor(
            Llama4Processor, use_fast=kwargs.pop("use_fast", True), **kwargs
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Although vLLM can support more images from an infra capability
        # perspective, we do not recommend using >10 images in practice.
        return {"image": None}

    @staticmethod
    def get_patch_per_chunk(vision_config: Llama4VisionConfig) -> int:
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size

        assert image_size % patch_size == 0, (
            f"chunk size {image_size} should be multiple of "
        )
        f"patch_size {patch_size}"

        ds_ratio = int(round(1.0 / (vision_config.pixel_shuffle_ratio**2)))
        return (image_size // patch_size) ** 2 // ds_ratio

    def get_max_num_tiles(self) -> int:
        image_processor = self.get_hf_processor().image_processor
        return image_processor.max_patches

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_config = self.get_hf_config().vision_config
        image_size = vision_config.image_size
        # Result in the max possible feature size (h:w = 16:1)
        return ImageSize(height=self.get_max_num_tiles() * image_size, width=image_size)


class Mllama4MultiModalProcessor(BaseMultiModalProcessor[Mllama4ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        if mm_data is None:
            return tokenizer(prompt, add_special_tokens=False)  # exclude bos
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        processor = self.info.get_hf_processor(**mm_kwargs)
        image_processor = processor.image_processor
        vision_config = self.info.get_hf_config().vision_config

        if processed_outputs.get("pixel_values") is not None:
            assert "images" in mm_data, (
                "images expected to be in mm_data when pixel_values is present"
            )

            images = mm_data["images"]
            mm_items = self.info.parse_mm_data({"image": images}, validate=False)
            parsed_images = mm_items.get_items("image", ImageProcessorItems)

            tile_size = vision_config.image_size
            possible_resolutions = find_supported_resolutions(
                max_num_chunks=self.info.get_max_num_tiles(),
                patch_size=SizeDict(height=tile_size, width=tile_size),
            )
            best_fit_sizes = [
                get_best_fit(
                    (image.size[1], image.size[0]),
                    torch.tensor(possible_resolutions),
                    resize_to_max_canvas=image_processor.resize_to_max_canvas,
                )
                for image in parsed_images
            ]
            # TODO tile height/width do not necessarily need to match
            aspect_ratios = [
                (image_size[0] // tile_size, image_size[1] // tile_size)
                for image_size in best_fit_sizes
            ]
            patches_per_image = [
                1 if r_h * r_w == 1 else 1 + r_h * r_w for (r_h, r_w) in aspect_ratios
            ]

            processed_outputs["aspect_ratios"] = torch.tensor(aspect_ratios)
            processed_outputs["patches_per_image"] = torch.tensor(patches_per_image)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        patches_per_image = hf_inputs.get("patches_per_image", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", patches_per_image
            ),
            patches_per_image=MultiModalFieldConfig.batched("image"),
            aspect_ratios=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        config = self.info.get_hf_config()
        vision_config = config.vision_config

        num_patches_per_chunk = self.info.get_patch_per_chunk(vision_config)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token
        img_patch_token = hf_processor.img_patch_token

        def get_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            aspect_ratio = out_item["aspect_ratios"].data

            repl = hf_processor._prompt_split_image(
                aspect_ratio=aspect_ratio,
                num_patches_per_chunk=num_patches_per_chunk,
            )

            return PromptUpdateDetails.select_text(repl, img_patch_token)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


class Mllama4DummyInputsBuilder(BaseDummyInputsBuilder[Mllama4ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.fake_image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        (target_width, target_height) = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    Mllama4MultiModalProcessor,
    info=Mllama4ProcessingInfo,
    dummy_inputs=Mllama4DummyInputsBuilder,
)
class Llama4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, "image"):
            from vllm.compilation.backends import set_model_tag

            with (
                set_current_vllm_config(vllm_config),
                set_model_tag("Llama4VisionModel", is_encoder=True),
            ):
                self.vision_model = Llama4VisionModel(
                    config=config.vision_config,
                    quant_config=None,
                    prefix=maybe_prefix(prefix, "vision_model"),
                )

            self.multi_modal_projector = Llama4MultiModalProjector(
                config=self.config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = initialize_model(
                vllm_config=vllm_config.with_hf_config(
                    config.text_config, ["LlamaForCausalLM"]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
                model_class=Llama4ForCausalLM,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.num_expert_groups = 1
        self.num_logical_experts = self.language_model.num_logical_experts
        self.num_physical_experts = self.language_model.num_physical_experts
        self.num_local_physical_experts = self.language_model.num_local_physical_experts
        self.num_routed_experts = self.language_model.num_routed_experts
        self.num_shared_experts = self.language_model.num_shared_experts
        self.num_redundant_experts = self.language_model.num_redundant_experts
        self.moe_layers = self.language_model.moe_layers
        self.num_moe_layers = len(self.moe_layers)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Set which layers should output auxiliary hidden states for EAGLE3."""
        # Delegate to underlying language model (Llama4ForCausalLM)
        assert hasattr(self.language_model, "set_aux_hidden_state_layers")
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Get the layer indices for auxiliary hidden state outputs.

        Note: The GPU model runner will override this with layers from
        the speculative config if available, providing dynamic configuration.
        """
        # Delegate to underlying language model (Llama4ForCausalLM)
        assert hasattr(self.language_model, "get_eagle3_aux_hidden_state_layers")
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ):
        self.language_model.set_eplb_state(
            expert_load_view, logical_to_physical_map, logical_replica_count
        )
        self.expert_weights = self.language_model.expert_weights

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ):
        self.language_model.update_physical_experts_metadata(
            num_physical_experts, num_local_physical_experts
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Llama4ImagePatchInputs | None:
        # num_images, 1, num_chunks, channel, image_size, image_size
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        patches_per_image = kwargs.pop("patches_per_image")
        aspect_ratios = kwargs.pop("aspect_ratios")

        return Llama4ImagePatchInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            patches_per_image=patches_per_image,
            aspect_ratios=aspect_ratios,
        )

    def _process_image_input(
        self, image_input: Llama4ImagePatchInputs
    ) -> MultiModalEmbeddings:
        assert self.vision_model and self.multi_modal_projector
        pixel_values = image_input["pixel_values"]
        patches_per_image = image_input["patches_per_image"].tolist()

        # shard image input
        if self.use_data_parallel:
            vision_embeddings_flat = run_dp_sharded_vision_model(
                pixel_values, self.vision_model
            )
        else:
            vision_embeddings_flat = self.vision_model(pixel_values)

        vision_embeddings_flat = self.multi_modal_projector(vision_embeddings_flat)

        return [
            img.flatten(0, 1)
            for img in vision_embeddings_flat.split(patches_per_image, dim=0)
        ]

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        with (
            set_forward_context(None, self.vllm_config),
        ):
            return self._process_image_input(image_input)

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

        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def separate_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        prefix: str,
    ) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str, torch.Tensor]]]:
        weights1, weights2 = tee(weights, 2)

        def get_prefix_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, data in weights1:
                if name.startswith(prefix):
                    yield (name, data)

        def get_other_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, data in weights2:
                if not name.startswith(prefix):
                    yield (name, data)

        return get_prefix_weights(), get_other_weights()

    def _consolidate_qkv_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        qkv_idx_mappings = {
            ".self_attn.q_proj": 0,
            ".self_attn.k_proj": 1,
            ".self_attn.v_proj": 2,
        }
        qkv_weights = {}
        for name, loaded_weight in weights:
            for weight_name, idx in qkv_idx_mappings.items():
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, ".self_attn.qkv_proj")
                if new_name not in qkv_weights:
                    qkv_weights[new_name] = [None] * 3
                qkv_weights[new_name][idx] = loaded_weight
                break
            else:
                yield name, loaded_weight
        for key, weight in qkv_weights.items():
            qkv_weight = torch.cat(weight, dim=0)
            yield key, qkv_weight

    def _rename_weight_for_modelopt_checkpoint(self, name: str) -> str:
        """Rename weights from ModelOpt llama4 fp8 checkpoints to vLLM
        format."""
        if name.startswith("model.") or name.startswith("language_model.model."):
            renamed = (
                name.replace("model.", "language_model.model.", 1)
                if name.startswith("model.")
                else name
            )
            # Handle expert scale parameters with flat naming
            if "feed_forward.experts." in name and (
                "_input_scale" in name or "_weight_scale" in name
            ):
                # Map checkpoint naming to vLLM's expected naming
                if "down_proj_input_scale" in renamed:
                    return renamed.replace("down_proj_input_scale", "w2_input_scale")
                elif "down_proj_weight_scale" in renamed:
                    return renamed.replace("down_proj_weight_scale", "w2_weight_scale")
                elif "gate_up_proj_input_scale" in renamed:
                    return renamed.replace(
                        "gate_up_proj_input_scale", "w13_input_scale"
                    )
                elif "gate_up_proj_weight_scale" in renamed:
                    return renamed.replace(
                        "gate_up_proj_weight_scale", "w13_weight_scale"
                    )
                return renamed

            # Handle attention scale parameters
            elif "self_attn." in name and (".k_scale" in name or ".v_scale" in name):
                if ".k_proj.k_scale" in renamed:
                    return renamed.replace(".k_proj.k_scale", ".attn.k_scale")
                elif ".v_proj.v_scale" in renamed:
                    return renamed.replace(".v_proj.v_scale", ".attn.v_scale")
                return renamed

            # Standard model.* to language_model.model.* renaming
            return renamed

        elif name.startswith("lm_head.weight"):
            return name.replace("lm_head.weight", "language_model.lm_head.weight")

        return name

    def _separate_and_rename_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Rename weights and separate them into language_model and other
        weights."""
        language_model_weights = []
        other_weights = []

        for name, weight in weights:
            renamed = self._rename_weight_for_modelopt_checkpoint(name)

            attr = renamed.split(".", 1)[0]
            if isinstance(getattr(self, attr), StageMissingLayer):
                continue

            if renamed.startswith("language_model."):
                language_model_weights.append((renamed, weight))
            else:
                other_weights.append((renamed, weight))

        return language_model_weights, other_weights

    def _handle_expert_scale_broadcasting(
        self, weights: list[tuple[str, torch.Tensor]], params_dict: dict
    ) -> tuple[list[tuple[str, torch.Tensor]], set[str]]:
        """Handle expert scale parameters that need broadcasting.

        ModelOpt checkpoints use a single value tensor scalar for BMM style
        experts, vLLM expects the scale to be broadcasted across all experts.
        """
        regular_weights = []
        expert_scale_weights = []
        updated_params = set()

        for name, weight in weights:
            # Check if this is an expert scale parameter that needs broadcasting
            if (
                "feed_forward.experts." in name
                and "scale" in name
                and ".shared_expert" not in name
            ):
                if name in params_dict:
                    param = params_dict[name]
                    if (
                        hasattr(param, "data")
                        and param.data.numel() > 1
                        and weight.numel() == 1
                    ):
                        # Broadcast single value to all experts
                        param.data.fill_(weight.item())
                        updated_params.add(name)
                        continue

                expert_scale_weights.append((name, weight))
            else:
                regular_weights.append((name, weight))

        return regular_weights, expert_scale_weights, updated_params

    def _load_other_weights(
        self,
        other_weights: Iterable[tuple[str, torch.Tensor]],
        params_dict: dict,
        stacked_params_mapping: list,
    ) -> set[str]:
        """Load non-language-model weights with stacking support."""
        updated_params = set()

        if self.use_data_parallel:
            other_weights = self._consolidate_qkv_weights(other_weights)

        for name, loaded_weight in other_weights:
            # Try stacked parameter mapping first
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or self.use_data_parallel:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Use regular weight loading
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                updated_params.add(name)

        return updated_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.text_config.num_local_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            # Shared expert gate_up_proj stacking
            (".shared_expert.gate_up_proj", ".shared_expert.gate_proj", 0),
            (".shared_expert.gate_up_proj", ".shared_expert.up_proj", 1),
            # Feed forward gate_up_proj stacking (for non-MoE layers if any)
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: set[str] = set()

        # Separate and rename weights
        language_model_weights, other_weights = self._separate_and_rename_weights(
            weights
        )

        # Handle expert scale parameters
        regular_weights, expert_scale_weights, updated_params_from_experts = (
            self._handle_expert_scale_broadcasting(language_model_weights, params_dict)
        )
        updated_params.update(updated_params_from_experts)

        loader = AutoWeightsLoader(self)
        loaded_language_model_params = loader.load_weights(regular_weights)
        assert loaded_language_model_params is not None
        updated_params.update(loaded_language_model_params)

        if expert_scale_weights:
            loaded_expert_scale_params = loader.load_weights(expert_scale_weights)
            if loaded_expert_scale_params:
                updated_params.update(loaded_expert_scale_params)

        updated_params.update(
            self._load_other_weights(other_weights, params_dict, stacked_params_mapping)
        )

        return updated_params

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector.",
            tower_model="vision_model.",
        )
