# SPDX-License-Identifier: Apache-2.0
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
from typing import Literal, Optional, TypedDict, Union

import torch
from torch import nn
from transformers import BatchFeature, Llama4Config, Llama4VisionConfig
from transformers.image_utils import SizeDict
from transformers.models.llama4 import Llama4Processor
from transformers.models.llama4.image_processing_llama4_fast import (
    find_supported_resolutions, get_best_fit)

from vllm.attention.layer import MultiHeadAttention
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import InputProcessingContext
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llama4 import Llama4ForCausalLM
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)


class Llama4ImagePatchInputs(TypedDict):
    type: Literal["pixel_values"]
    flat_data: torch.Tensor
    """
    Shape:
    `(batch_size * num_chunks, num_channels, image size, image size)`
    """
    patches_per_image: torch.Tensor
    """
    The number of total patches for each image in the batch.

    This is used to split the embeddings which has the first two dimensions
    flattened just like `flat_data`.
    """

    aspect_ratios: Union[torch.Tensor, list[torch.Tensor]]
    """
    A list of aspect ratios corresponding to the number of tiles
    in each dimension that each image in the batch corresponds to.

    Shape:
    `(batch_size, ratio)` where ratio is a pair `(ratio_h, ratio_w)`
    """


class Llama4VisionMLP(nn.Module):

    def __init__(self,
                 input_size: int,
                 intermediate_size: int,
                 output_size: int,
                 bias: bool,
                 output_activation: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=input_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
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
        quant_config: Optional[QuantizationConfig] = None,
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

    reshaped_tensor = input_tensor.view(batch_size, height,
                                        int(width * shuffle_ratio),
                                        int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(batch_size,
                                           int(height * shuffle_ratio),
                                           int(width * shuffle_ratio),
                                           int(channels / (shuffle_ratio**2)))
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1,
                                         reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.projector_input_dim //
                             (self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP(
            input_size=config.intermediate_size,
            intermediate_size=config.projector_input_dim,
            output_size=config.projector_output_dim,
            bias=config.multi_modal_projector_bias,
            output_activation=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp")

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches,
                                        self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


class Llama4VisionAttention(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        assert self.num_heads % self.tp_size == 0
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_size = self.num_local_heads * self.head_dim
        self.kv_size = self.num_local_heads * self.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.attn = MultiHeadAttention(self.num_local_heads, self.head_dim,
                                       self.scaling)
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

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=config.hidden_size // config.num_attention_heads // 2,
            # number of image patches
            max_position=(config.image_size // config.patch_size)**2,
            base=config.rope_theta,
            rope_scaling={"rope_type": "mllama4"},
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
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        self.self_attn = Llama4VisionAttention(config,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.self_attn")
        self.mlp = Llama4VisionMLP(input_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   output_size=config.hidden_size,
                                   bias=True,
                                   output_activation=False,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

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

        outputs = (hidden_state, )
        return outputs


class Llama4VisionEncoder(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Llama4VisionEncoderLayer(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            ) for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape
                    `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to
                directly pass an embedded representation. This is useful if you
                want more control over how to convert `input_ids` indices into
                associated vectors than the model's internal embedding
                lookup matrix.
        """

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs[0]

        return hidden_states


class Llama4UnfoldConvolution(nn.Module):

    def __init__(self,
                 config: Llama4VisionConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,
                                      stride=config.patch_size)
        self.linear = ColumnParallelLinear(config.num_channels *
                                           kernel_size[0] * kernel_size[1],
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           gather_output=True,
                                           prefix=f"{prefix}.linear")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states, _ = self.linear(hidden_states)
        return hidden_states


class Llama4VisionModel(nn.Module):

    def __init__(
        self,
        config: Llama4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size)**2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.patch_embedding")

        self.class_embedding = nn.Parameter(self.scale *
                                            torch.randn(self.hidden_size))
        self.positional_embedding_vlm = nn.Parameter(
            self.scale * torch.randn(self.num_patches, self.hidden_size))

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # encoders
        self.model = Llama4VisionEncoder(config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.model")
        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            config, quant_config, prefix=f"{prefix}.vision_adapter")

    def forward(
        self,
        images_flattened: torch.Tensor,
    ) -> torch.Tensor:
        # Patch embedding
        hidden_state = self.patch_embedding(images_flattened)
        num_tiles, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        class_embedding = self.class_embedding.expand(hidden_state.shape[0], 1,
                                                      hidden_state.shape[-1])
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
            dtype=hidden_state.dtype, device=hidden_state.device)
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
        return self.ctx.get_hf_processor(Llama4Processor,
                                         use_fast=True,
                                         **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Although vLLM can support more images from an infra capability
        # perspective, we do not recommend using >10 images in practice.
        return {"image": None}

    @staticmethod
    def get_patch_per_chunk(vision_config: Llama4VisionConfig) -> int:
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size

        assert (
            image_size %
            patch_size == 0), f"chunk size {image_size} should be multiple of "
        f"patch_size {patch_size}"

        ds_ratio = int(round(1.0 / (vision_config.pixel_shuffle_ratio**2)))
        return (image_size // patch_size)**2 // ds_ratio

    def get_max_num_tiles(self) -> int:
        image_processor = self.get_hf_processor().image_processor
        return image_processor.max_patches

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_config = self.get_hf_config().vision_config
        image_size = vision_config.image_size
        # Result in the max possible feature size (h:w = 16:1)
        return ImageSize(height=self.get_max_num_tiles() * image_size,
                         width=image_size)


class Mllama4MultiModalProcessor(BaseMultiModalProcessor[Mllama4ProcessingInfo]
                                 ):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        if mm_data is None:
            return tokenizer(prompt, add_special_tokens=False)  # exclude bos
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        processor = self.info.get_hf_processor(**mm_kwargs)
        image_processor = processor.image_processor
        vision_config = self.info.get_hf_config().vision_config

        if processed_outputs.get("pixel_values") is not None:
            assert "images" in mm_data, \
                "images expected to be in mm_data when pixel_values is present"

            images = mm_data["images"]
            parsed_images = (self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", ImageProcessorItems))

            tile_size = vision_config.image_size
            possible_resolutions = find_supported_resolutions(
                max_num_chunks=self.info.get_max_num_tiles(),
                patch_size=SizeDict(height=tile_size, width=tile_size),
            )
            best_fit_sizes = [
                get_best_fit(
                    (image.size[1], image.size[0]),
                    torch.tensor(possible_resolutions),
                    resize_to_max_canvas=image_processor.resize_to_max_canvas)
                for image in parsed_images
            ]
            # TODO tile height/width do not necessarily need to match
            aspect_ratios = [(image_size[0] // tile_size,
                              image_size[1] // tile_size)
                             for image_size in best_fit_sizes]
            patches_per_image = [
                1 if r_h * r_w == 1 else 1 + r_h * r_w
                for (r_h, r_w) in aspect_ratios
            ]

            processed_outputs["aspect_ratios"] = aspect_ratios
            processed_outputs["patches_per_image"] = torch.tensor(
                patches_per_image)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        patches_per_image = hf_inputs.get("patches_per_image", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", patches_per_image),
            patches_per_image=MultiModalFieldConfig.batched("image"),
            aspect_ratios=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        assert (
            mm_items.get_count("image", strict=False) == 0
            or "aspect_ratios" in out_mm_kwargs
        ), "Transformers expect to include aspect_ratios in out_mm_kwargs"

        config = self.info.get_hf_config()
        vision_config = config.vision_config

        num_patches_per_chunk = self.info.get_patch_per_chunk(vision_config)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token
        img_patch_token = hf_processor.img_patch_token

        def get_replacement(item_idx: int):
            aspect_ratio = out_mm_kwargs["aspect_ratios"][item_idx]

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
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        (target_width,
         target_height) = self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


@MULTIMODAL_REGISTRY.register_processor(
    Mllama4MultiModalProcessor,
    info=Mllama4ProcessingInfo,
    dummy_inputs=Mllama4DummyInputsBuilder,
)
class Llama4ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                     SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self.vision_model = Llama4VisionModel(config.vision_config,
                                              None,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_model"))
        self.multi_modal_projector = Llama4MultiModalProjector(
            self.config,
            None,
            prefix=maybe_prefix(prefix, "multi_modal_projector"))
        self.language_model = initialize_model(
            vllm_config=vllm_config.with_hf_config(config.text_config,
                                                   ["LlamaForCausalLM"]),
            prefix=maybe_prefix(prefix, "language_model"),
            model_class=Llama4ForCausalLM,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Llama4ImagePatchInputs]:
        # num_images, 1, num_chunks, channel, image_size, image_size
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        # num_images x num_chunks, channel, image_size, image_size
        # TODO: confirm handling for variable lengths
        flat_pixel_values = flatten_bn(pixel_values, concat=True)
        patches_per_image = flatten_bn(kwargs.pop("patches_per_image"))

        aspect_ratios = kwargs.pop("aspect_ratios", None)
        if not isinstance(aspect_ratios, (torch.Tensor, list)):
            raise ValueError("Incorrect type of aspect_ratios. "
                             f"Got type: {type(aspect_ratios)}")

        return Llama4ImagePatchInputs(
            type="pixel_values",
            flat_data=flat_pixel_values,
            patches_per_image=patches_per_image,
            aspect_ratios=aspect_ratios,
        )

    def _process_image_input(
            self, image_input: Llama4ImagePatchInputs) -> MultiModalEmbeddings:
        flat_data = image_input["flat_data"]
        patches_per_image = image_input["patches_per_image"].tolist()

        vision_embeddings_flat = self.vision_model(flat_data)
        vision_embeddings_flat = self.multi_modal_projector(
            vision_embeddings_flat)

        return [
            img.flatten(0, 1)
            for img in vision_embeddings_flat.split(patches_per_image, dim=0)
        ]

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_index,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner,
        # this condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        return self.language_model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def separate_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        prefix: str,
    ) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[
            str, torch.Tensor]]]:
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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        updated_params: set[str] = set()

        # language_model is an Llama4ForCausalLM instance. We load it's
        # using llama4's load_weights routine.
        language_model_weights, other_weights = self.separate_weights(
            weights, prefix="language_model.")
        loader = AutoWeightsLoader(self)
        loaded_language_model_params = loader.load_weights(
            language_model_weights)
        assert loaded_language_model_params is not None
        updated_params.update(loaded_language_model_params)

        for name, loaded_weight in other_weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                updated_params.add(name)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)

                weight_loader(param, loaded_weight)
                updated_params.add(name)
        return updated_params
