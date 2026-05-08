# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MOSS-VL model compatible with HuggingFace weights.

MOSS-VL uses a Qwen3-VL-style vision tower and a Qwen3-style text tower with
configured cross-attention layers. Vision outputs are consumed through vLLM's
encoder-decoder cross-attention KV cache instead of being merged into the text
embedding stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import CrossAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .qwen3 import Qwen3DecoderLayer
from .qwen3_vl import Qwen3_VisionBlock, Qwen3_VisionPatchEmbed
from .utils import WeightsMapper, maybe_prefix
from .vision import get_vit_attn_backend

logger = init_logger(__name__)


def _get_text_rope_parameters(config) -> dict[str, Any]:
    set_default_rope_theta(config, default_theta=1000000)
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return rope_parameters

    rope_parameters = {"rope_theta": getattr(config, "rope_theta", 1000000)}
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_parameters.update(rope_scaling)
    return rope_parameters


class MossVLVisionPatchMerger(nn.Module):
    """MOSS-VL merger: concatenate final and deepstack features, then one MLP."""

    def __init__(
        self,
        config,
        num_deepstack_features: int,
        norm_layer: type[nn.Module] | partial,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        base_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.input_hidden_size = base_hidden_size * (1 + num_deepstack_features)

        self.norms = nn.ModuleList(
            [norm_layer(config.hidden_size) for _ in range(1 + num_deepstack_features)]
        )
        self.linear_fc1 = ColumnParallelLinear(
            self.input_hidden_size,
            self.input_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.input_hidden_size,
            config.out_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
        )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        deepstack_features: list[torch.Tensor],
    ) -> torch.Tensor:
        features = [last_hidden_state] + deepstack_features
        hidden_states = [norm(feat) for norm, feat in zip(self.norms, features)]
        hidden_states = torch.cat(hidden_states, dim=-1)
        hidden_states = hidden_states.view(-1, self.input_hidden_size)
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class MossVLVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = getattr(
            vision_config, "deepstack_visual_indexes", []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{idx}",
                )
                for idx in range(vision_config.depth)
            ]
        )
        self.merger = MossVLVisionPatchMerger(
            vision_config,
            num_deepstack_features=len(self.deepstack_visual_indexes),
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        wpos_ids = wpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()
        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids_t = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
        return cos[pos_ids_t].flatten(1), sin[pos_ids_t].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        from .qwen3_vl import HAS_TRITON, pos_embed_interpolate_native

        if HAS_TRITON:
            from .qwen3_vl import triton_pos_embed_interpolate as interpolate_fn
        else:
            interpolate_fn = pos_embed_interpolate_native

        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                interpolate_fn(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    def prepare_encoder_metadata(
        self, grid_thw: list[list[int]]
    ) -> dict[str, torch.Tensor | None]:
        from vllm.model_executor.layers.attention import MMEncoderAttention

        metadata: dict[str, torch.Tensor | None] = {}
        metadata["pos_embeds"] = self.fast_pos_embed_interpolate(grid_thw)
        cos, sin = self.rot_pos_emb(grid_thw)
        metadata["rotary_pos_emb_cos"] = cos
        metadata["rotary_pos_emb_sin"] = sin

        grid_np = np.array(grid_thw, dtype=np.int32)
        patches_per_frame = grid_np[:, 1] * grid_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_np[:, 0]).cumsum(dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, self.device
        )
        metadata["max_seqlen"] = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
            dtype=torch.int32,
        )
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            get_tensor_model_parallel_world_size(),
            self.device,
        )
        return metadata

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        grid_thw_list = grid_thw.tolist()
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)
        hidden_states = hidden_states + encoder_metadata["pos_embeds"]
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_features: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata["sequence_lengths"],
            )
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(hidden_states)

        return self.merger(hidden_states, deepstack_features)


class MossVLTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError("Only silu is supported for MossVLTextMLP.")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MossVLTextCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 32768),
            rope_parameters=_get_text_rope_parameters(config),
        )
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _apply_rotary_one(
        self, positions: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        states, _ = self.rotary_emb(positions, states, states)
        return states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        vision_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.view(-1, self.head_dim)).view(q.shape)
        q = self._apply_rotary_one(positions, q)

        if encoder_hidden_states is not None:
            k, _ = self.k_proj(encoder_hidden_states)
            v, _ = self.v_proj(encoder_hidden_states)
            k = self.k_norm(k.view(-1, self.head_dim)).view(k.shape)
            if vision_positions is not None:
                k = self._apply_rotary_one(vision_positions, k)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MossVLCrossAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cross_attn = MossVLTextCrossAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))
        self.mlp = MossVLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
        vision_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None

        residual_states = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            positions=positions,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            vision_positions=vision_positions,
        )
        hidden_states = (
            residual_states + self.cross_attn_attn_gate.tanh() * hidden_states
        )

        residual_states = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = (
            residual_states + self.cross_attn_mlp_gate.tanh() * hidden_states
        )
        return hidden_states, residual


class MossVLTextModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        cross_attention_layers = set(getattr(config, "cross_attention_layers", []))
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{layer_idx}"
            if layer_idx in cross_attention_layers:
                layer = MossVLCrossAttentionDecoderLayer(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=layer_prefix,
                )
            else:
                layer = Qwen3DecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=layer_prefix,
                )
            self.layers.append(layer)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
            residual = None
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
            residual = None

        for layer in self.layers:
            if isinstance(layer, MossVLCrossAttentionDecoderLayer):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    encoder_hidden_states=encoder_hidden_states,
                    vision_positions=vision_positions,
                )
            else:
                hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MossVLForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = MossVLTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            vision_positions=vision_positions,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)


class MossVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # This is a conservative placeholder for profiling. Real token counts
        # come from the HF processor's grid_thw at request-processing time.
        max_tokens = min(seq_len, self.ctx.model_config.max_model_len)
        return {"image": max_tokens, "video": max_tokens}


class MossVLDummyInputsBuilder(BaseDummyInputsBuilder[MossVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        image_token = getattr(processor, "image_placeholder", "<|image|>")
        video_token = getattr(processor, "video_placeholder", "<|video|>")
        return image_token * mm_counts.get("image", 0) + video_token * mm_counts.get(
            "video", 0
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ):
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")
        return {
            "image": self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=224,
                height=224,
                num_frames=2,
                num_videos=num_videos,
                overrides=video_overrides,
            ),
        }


class MossVLMultiModalProcessor(BaseMultiModalProcessor[MossVLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_key = "grid_thw"
        if grid_key not in hf_inputs:
            grid_key = "image_grid_thw"
        if grid_key not in hf_inputs:
            grid_key = "video_grid_thw"
        grid_thw = hf_inputs.get(grid_key, torch.empty((0, 3), dtype=torch.long))

        grid_sizes = grid_thw.prod(-1)
        fields = {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_key: MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }
        if "pixel_values_videos" in hf_inputs:
            fields["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "image", grid_sizes
            )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        image_placeholder = getattr(processor, "image_placeholder", "<|image|>")
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        image_token_id = vocab.get(
            image_token, getattr(self.info.get_hf_config(), "image_token_id", None)
        )
        if image_token_id is None:
            raise ValueError("MOSS-VL requires an image token in the tokenizer.")

        merge_square = self.info.get_hf_config().vision_config.spatial_merge_size ** 2

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            out_item = out_mm_kwargs["image"][item_idx]
            grid_key = "grid_thw"
            if grid_key not in out_item:
                grid_key = "image_grid_thw"
            if grid_key not in out_item:
                grid_key = "video_grid_thw"
            grid_thw = out_item[grid_key].data
            assert isinstance(grid_thw, torch.Tensor)

            # MOSS-VL keeps a single image token in the decoder text stream,
            # but the cross-attention KV cache must be sized for the full
            # encoder output, including one separator token per frame.
            num_encoder_tokens = int(grid_thw.prod().item()) // merge_square
            num_encoder_tokens += int(grid_thw[0].item())

            def get_num_encoder_tokens(*_: object) -> torch.Tensor:
                return torch.tensor([num_encoder_tokens], dtype=torch.int32)

            return PromptUpdateDetails(
                full=[image_token_id],
                is_embed=get_num_encoder_tokens,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder,
                replacement=get_image_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MossVLMultiModalProcessor,
    info=MossVLProcessingInfo,
    dummy_inputs=MossVLDummyInputsBuilder,
)
class MossVLForConditionalGeneration(nn.Module, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv": ["qkv"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model.separator_token": "separator_token",
        }
    )

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("video"):
            return "<|video|>"
        raise ValueError("Only image/video inputs are supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.spatial_merge_size = int(
            getattr(config.vision_config, "spatial_merge_size", 2)
        )
        self.visual = MossVLVisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )
        self.language_model = MossVLForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.separator_token = nn.Parameter(
            torch.zeros(config.vision_config.out_hidden_size)
        )

    def _compute_vision_positions(
        self,
        grid_thw: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        positions: list[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            eff_h = h // self.spatial_merge_size
            eff_w = w // self.spatial_merge_size
            y = torch.arange(eff_h, device=device).view(1, eff_h, 1)
            x = torch.arange(eff_w, device=device).view(1, 1, eff_w)
            for frame_idx in range(t):
                t_grid = torch.full((1, eff_h, eff_w), frame_idx, device=device)
                frame_pos = torch.stack(
                    [
                        t_grid.expand(1, eff_h, eff_w),
                        frame_idx + y.expand(1, eff_h, eff_w),
                        frame_idx + x.expand(1, eff_h, eff_w),
                    ],
                    dim=0,
                ).reshape(3, -1)
                sep = torch.full((3, 1), frame_idx + max(eff_h, eff_w), device=device)
                positions.append(torch.cat([frame_pos, sep], dim=1))
        return torch.cat(positions, dim=1).long()

    def _insert_separator_tokens(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        merge_square = self.spatial_merge_size**2
        tokens_per_media = grid_thw.prod(dim=1) // merge_square
        hidden_size = hidden_states.shape[-1]
        separator = self.separator_token.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        output_parts = []
        src_offset = 0
        for media_idx in range(grid_thw.shape[0]):
            num_tokens = int(tokens_per_media[media_idx].item())
            num_frames = int(grid_thw[media_idx, 0].item())
            tokens_per_frame = num_tokens // num_frames
            media_states = hidden_states[src_offset : src_offset + num_tokens].view(
                num_frames, tokens_per_frame, hidden_size
            )
            separators = separator.view(1, 1, hidden_size).expand(
                num_frames, 1, hidden_size
            )
            output_parts.append(
                torch.cat([media_states, separators], dim=1).flatten(0, 1)
            )
            src_offset += num_tokens
        return torch.cat(output_parts, dim=0)

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            pixel_values = kwargs.pop("pixel_values_videos", None)
        grid_thw = kwargs.pop("grid_thw", None)
        if grid_thw is None:
            grid_thw = kwargs.pop("image_grid_thw", None)
        if grid_thw is None:
            grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values is None:
            return None
        return pixel_values, grid_thw

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        parsed = self._parse_and_validate_image_input(**kwargs)
        if parsed is None:
            return None
        pixel_values, grid_thw = parsed
        assert grid_thw is not None
        assert grid_thw.ndim == 2

        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        grid_thw = grid_thw.to(device=self.visual.device, dtype=torch.long)
        vision_hidden_states = self.visual(pixel_values, grid_thw=grid_thw)
        vision_hidden_states = self._insert_separator_tokens(
            vision_hidden_states, grid_thw
        )
        vision_positions = self._compute_vision_positions(
            grid_thw, device=vision_hidden_states.device
        ).T.to(dtype=vision_hidden_states.dtype)

        packed = torch.cat([vision_hidden_states, vision_positions], dim=-1)
        merge_square = self.spatial_merge_size**2
        output_sizes = (grid_thw.prod(dim=1) // merge_square + grid_thw[:, 0]).tolist()
        return tuple(packed.split(output_sizes))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        encoder_hidden_states = None
        vision_positions = None
        if encoder_outputs:
            packed = torch.cat(encoder_outputs, dim=0)
            encoder_hidden_states = packed[:, :-3]
            vision_positions = packed[:, -3:].T.long()

        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            vision_positions=vision_positions,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkv.", ".q.", "q"),
            (".qkv.", ".k.", "k"),
            (".qkv.", ".v.", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name == "lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith("model.visual."):
                name = "visual." + name[len("model.visual.") :]
            if name.startswith("model.separator_token"):
                name = name.replace("model.", "", 1)

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".cross_attn." in name and weight_name in (
                    ".q_proj",
                    ".k_proj",
                    ".v_proj",
                ):
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    handled = True
                    break
                if mapped_name in params_dict:
                    param = params_dict[mapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, weight, shard_id)
                    loaded_params.add(mapped_name)
                    handled = True
                    break

            if handled:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.debug("Skipping weight: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded_params.add(name)

        return loaded_params

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["visual.merger"],
            tower_model="visual.",
        )
