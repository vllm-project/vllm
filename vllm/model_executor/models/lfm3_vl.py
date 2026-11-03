# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3-VL model.

The vision encoder interleaves gated depthwise 2D convolutions and
bidirectional attention with axial 2D RoPE over the patches of each frame.
The projector merges each 2x2 block of vision tokens into one LFM3 token.
"""

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed import utils as dist_utils
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from .qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_vit_attn_backend


class Lfm3VlImagePixelInputs(TensorSchema):
    """Dimensions:
    - np: Number of patches over all images
    - cps: Number of channels * temporal_patch_size * patch_size**2
    - ni: Number of images
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class Lfm3VlProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class Lfm3VlDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
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


class Lfm3VlVisionMLP(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.act_fn = SiluAndMul()
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w13(x)
        x, _ = self.w2(self.act_fn(x))
        return x


class Lfm3VlVisionAttention(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = dist_utils.divide(config.num_attention_heads, tp_size)
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            enable_fp32_compute=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_layernorm(q.reshape(seq_len, self.num_heads, self.head_dim))
        k = self.k_layernorm(k.reshape(seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape(1, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = rotary_pos_emb.cos(), rotary_pos_emb.sin()
        q = self.apply_rotary_emb(q.unsqueeze(0), cos, sin)
        k = self.apply_rotary_emb(k.unsqueeze(0), cos, sin)

        out = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        out, _ = self.out_proj(out.reshape(seq_len, self.q_size))
        return out


class Lfm3VlVisionConv(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.merge_size = config.spatial_merge_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        num_channels = dist_utils.divide(config.hidden_size, tp_size)
        self.spatial_conv = nn.Module()
        self.spatial_conv.weight = nn.Parameter(
            torch.empty(num_channels, 1, self.kernel_size, self.kernel_size)
        )
        set_weight_attrs(
            self.spatial_conv.weight, {"weight_loader": sharded_weight_loader(0)}
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def _conv(self, frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Patches come in merge-block order; the convolution needs raster order.
        m = self.merge_size
        num_frames = frames.shape[0]
        frames = frames.view(num_frames, height // m, width // m, m, m, -1)
        frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(num_frames, -1, height, width)
        frames = F.conv2d(
            frames,
            self.spatial_conv.weight,
            padding=self.kernel_size // 2,
            groups=frames.shape[1],
        )
        frames = frames.view(num_frames, -1, height // m, m, width // m, m)
        return frames.permute(0, 2, 4, 3, 5, 1).reshape(num_frames * height * width, -1)

    def forward(
        self, hidden_states: torch.Tensor, frame_shapes: list[tuple[int, int]]
    ) -> torch.Tensor:
        projected, _ = self.in_proj(hidden_states)
        b, c, x = projected.chunk(3, dim=-1)
        gated = b * x
        if len(set(frame_shapes)) == 1:
            height, width = frame_shapes[0]
            frames = gated.view(len(frame_shapes), height * width, -1)
            conv_out = self._conv(frames, height, width)
        else:
            lengths = [height * width for height, width in frame_shapes]
            conv_out = torch.cat(
                [
                    self._conv(frame.unsqueeze(0), height, width)
                    for frame, (height, width) in zip(
                        gated.split(lengths), frame_shapes
                    )
                ]
            )
        out, _ = self.out_proj(c * conv_out)
        return out


class Lfm3VlVisionLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(
                config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
            )
        else:
            self.conv = Lfm3VlVisionConv(
                config, quant_config=quant_config, prefix=f"{prefix}.conv"
            )
        self.feed_forward = Lfm3VlVisionMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        frame_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        normed = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            hidden_states = hidden_states + self.self_attn(
                normed, cu_seqlens, rotary_pos_emb, max_seqlen
            )
        else:
            hidden_states = hidden_states + self.conv(normed, frame_shapes)
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlVisionModel(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        self.patchifier = ReplicatedLinear(
            patch_dim,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.patchifier",
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        spatial_dim = head_dim // 2
        theta = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )

        self.layers = nn.ModuleList(
            [
                Lfm3VlVisionLayer(
                    config,
                    layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patchifier.weight.dtype

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor:
        m = self.spatial_merge_size
        pos_ids = []
        for t, h, w in grid_thw:
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            hw = torch.stack([hpos, wpos], dim=-1).reshape(h // m, m, w // m, m, 2)
            pos_ids.append(hw.transpose(1, 2).reshape(-1, 2).repeat(t, 1))
        pos_ids = torch.cat(pos_ids).to(self.inv_freq.device)
        freqs = pos_ids[..., None].float() * self.inv_freq
        return freqs.flatten(1)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        frame_shapes = [(h, w) for t, h, w in grid_thw for _ in range(t)]
        lengths = torch.tensor([h * w for h, w in frame_shapes], dtype=torch.int32)
        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        cu_seqlens = cu_seqlens.to(pixel_values.device, non_blocking=True)
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        hidden_states, _ = self.patchifier(pixel_values.to(self.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens, rotary_pos_emb, max_seqlen, frame_shapes
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv", ".q_proj", "q"),
            (".qkv", ".k_proj", "k"),
            (".qkv", ".v_proj", "v"),
            (".w13", ".w1", 0),
            (".w13", ".w3", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Lfm3VlMultiModalProjector(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        self.in_features = vision_hidden_size * config.downsample_factor**2
        self.layer_norm = nn.LayerNorm(
            vision_hidden_size, eps=config.projector_layer_norm_eps
        )
        self.linear_1 = ReplicatedLinear(
            self.in_features,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        assert config.projector_hidden_act == "gelu"
        self.act = nn.GELU()
        self.linear_2 = ReplicatedLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # Consecutive patches form one merge block.
        hidden_states = self.layer_norm(vision_features).reshape(-1, self.in_features)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states, _ = self.linear_2(self.act(hidden_states))
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=Lfm3VlProcessingInfo,
    dummy_inputs=Lfm3VlDummyInputsBuilder,
)
class Lfm3VlForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, IsHybrid
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image|><|vision_end|>"
        raise ValueError("Only image modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        text_config = vllm_config.model_config.hf_config.text_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=text_config.hidden_size,
            conv_kernel=text_config.conv_L_cache,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None
        self.config = config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = Lfm3VlVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = Lfm3VlMultiModalProjector(
                config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Lfm3ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Lfm3VlImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        return Lfm3VlImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=kwargs.pop("image_grid_thw", None),
        )

    def _process_image_input(
        self, image_input: Lfm3VlImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        grid_thw_list = grid_thw.tolist()
        vision_features = self.vision_tower(image_input["pixel_values"], grid_thw_list)
        image_embeds = self.multi_modal_projector(vision_features)
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size**2).tolist()
        return image_embeds.split(sizes)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
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
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
