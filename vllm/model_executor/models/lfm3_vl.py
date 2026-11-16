# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3-VL model: images and videos in, text out."""

from collections.abc import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.lfm3_vl import Lfm3VlConfig, Lfm3VlVisionConfig
from vllm.transformers_utils.processors.lfm3_vl import Lfm3VlProcessor
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from .lfm3 import Lfm3MLP, _short_conv_state_dtype, _short_conv_state_shape
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_fp8_padded_hidden_size, get_vit_attn_backend


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame, with
    RMSNorm on the query and key heads and axial 2D RoPE."""

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.out_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_layernorm(q.reshape(seq_len, self.num_heads, self.head_dim))
        k = self.k_layernorm(k.reshape(seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape(1, seq_len, self.num_kv_heads, self.head_dim)
        q = self.apply_rotary_emb(q[None], cos, sin)
        k = self.apply_rotary_emb(k[None], cos, sin)
        if self.attn.attn_backend == AttentionBackendEnum.FLASHINFER:
            # The cuDNN wrapper shares physical offsets between Q, K and O, so
            # K and V take the query head count and V the same contiguous layout.
            if self.num_kv_heads != self.num_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=2)
                v = v.repeat_interleave(repeats, dim=2)
            v = v.contiguous()
        output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output, _ = self.out_proj(output.reshape(seq_len, -1))
        return output


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = MergedColumnParallelLinear(
            dim,
            [dim] * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        # One (k, k) filter per channel, sharded over channels with the
        # projections; the parameter keeps the Conv2d weight shape for loading.
        self.spatial_conv = ColumnParallelLinear(
            self.kernel_size**2, dim, bias=False, prefix=f"{prefix}.spatial_conv"
        )
        self.spatial_conv.weight.data = self.spatial_conv.weight.data.view(
            -1, 1, self.kernel_size, self.kernel_size
        )
        self.out_proj = RowParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj"
        )

    def _convolve(self, frames: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            frames,
            self.spatial_conv.weight,
            padding=self.kernel_size // 2,
            groups=frames.shape[1],
        )

    def forward(
        self, hidden_states: torch.Tensor, frame_shapes: list[tuple[int, int]]
    ) -> torch.Tensor:
        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)
        gated = B * x
        merge = self.spatial_merge_size
        if len(set(frame_shapes)) == 1:
            # frames of one shape convolve as one batch: merge-block order ->
            # raster grid, convolve, back
            height, width = frame_shapes[0]
            frames = gated.view(
                len(frame_shapes), height // merge, width // merge, merge, merge, -1
            )
            frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(
                len(frame_shapes), -1, height, width
            )
            frames = self._convolve(frames)
            frames = frames.view(
                len(frame_shapes), -1, height // merge, merge, width // merge, merge
            )
            conv_output = frames.permute(0, 2, 4, 3, 5, 1).reshape(gated.shape[0], -1)
        else:
            outputs = []
            lengths = [height * width for height, width in frame_shapes]
            for frame, (height, width) in zip(gated.split(lengths), frame_shapes):
                frame = frame.view(height // merge, width // merge, merge, merge, -1)
                frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
                frame = self._convolve(frame)
                frame = frame.view(-1, height // merge, merge, width // merge, merge)
                outputs.append(frame.permute(1, 3, 2, 4, 0).reshape(height * width, -1))
            conv_output = torch.cat(outputs)
        output, _ = self.out_proj(C * conv_output)
        return output


class Lfm3VlVisionLayer(nn.Module):
    def __init__(
        self,
        config: Lfm3VlVisionConfig,
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
            self.conv = Lfm3VlVisionConv2d(
                config, quant_config=quant_config, prefix=f"{prefix}.conv"
            )
        self.feed_forward = Lfm3MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
        frame_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        normed = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            operator_output = self.self_attn(
                normed, cu_seqlens, max_seqlen, cos, sin, sequence_lengths
            )
        else:
            operator_output = self.conv(normed, frame_shapes)
        hidden_states = hidden_states + operator_output
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlVisionModel(nn.Module):
    """The LFM3-VL vision encoder: gated convolution and bidirectional attention
    layers over the patches of each image or video frame, embedded by a linear
    patchifier."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".w1.": (".w13.", 0),
            ".w3.": (".w13.", 1),
        }
    )

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_attention_heads
        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        self.patchifier = ReplicatedLinear(
            patch_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.patchifier",
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        # Axial 2D RoPE: half the head rotates with the row, half with the column.
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={
                "partial_rotary_factor": 0.5,
                "rope_theta": config.rope_parameters["rope_theta"],
            },
        )
        self.layers = nn.ModuleList(
            Lfm3VlVisionLayer(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )
        # FlashInfer expands KV to the TP-local query head count before attention,
        # so Q/K/V share one contiguous stride.
        num_heads = divide(
            config.num_attention_heads, get_tensor_model_parallel_world_size()
        )
        self.flashinfer_hidden_size = (
            get_fp8_padded_hidden_size(num_heads, head_dim) or num_heads * head_dim
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patchifier.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patchifier.weight.device

    def position_embeddings(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos and sin of every patch's (row, column), in merge-block order."""
        merge = self.spatial_merge_size
        positions = []
        for t, h, w in grid_thw:
            rows = torch.arange(h, device=self.device)[:, None].expand(h, w)
            cols = torch.arange(w, device=self.device)[None, :].expand(h, w)
            rows = rows.reshape(h // merge, merge, w // merge, merge).transpose(1, 2)
            cols = cols.reshape(h // merge, merge, w // merge, merge).transpose(1, 2)
            positions.append(
                torch.stack([rows.flatten(), cols.flatten()], -1).repeat(t, 1)
            )
        positions = torch.cat(positions)
        cos, sin = self.rotary_pos_emb.get_cos_sin(
            max(max(h, w) for _, h, w in grid_thw)
        )
        return cos[positions].flatten(1), sin[positions].flatten(1)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        hidden_states, _ = self.patchifier(pixel_values.to(self.device, self.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        cos, sin = self.position_embeddings(grid_thw)
        frame_shapes = [(h, w) for t, h, w in grid_thw for _ in range(t)]
        cu_seqlens = np.cumsum([0] + [h * w for h, w in frame_shapes], dtype=np.int32)
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
            dtype=torch.int32,
        )
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, self.device
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            hidden_size=hidden_states.shape[-1],
            tp_size=1,  # flashinfer_hidden_size is already TP-local
            device=self.device,
            fp8_padded_hidden_size=self.flashinfer_hidden_size,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=cos,
                sin=sin,
                sequence_lengths=sequence_lengths,
                frame_shapes=frame_shapes,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )


class Lfm3VlMultiModalProjector(nn.Module):
    """Merges each block of `spatial_merge_size**2` vision tokens, normalized
    before the merge, into one language token."""

    def __init__(
        self,
        config: Lfm3VlConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision_config = config.vision_config
        self.layer_norm = nn.LayerNorm(vision_config.hidden_size, eps=1e-6)
        self.linear_1 = ColumnParallelLinear(
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(image_features)
        hidden_states = hidden_states.reshape(-1, self.linear_1.input_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states, _ = self.linear_2(self.act(hidden_states))
        return hidden_states


class Lfm3VlProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Lfm3VlConfig:
        return self.ctx.get_hf_config(Lfm3VlConfig)

    def get_hf_processor(self, **kwargs: object) -> Lfm3VlProcessor:
        return self.ctx.get_hf_processor(Lfm3VlProcessor, **kwargs)


class Lfm3VlDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        image = "<|vision_start|><|image|><|vision_end|>"
        video = "<|vision_start|><|video|><|vision_end|>"
        return image * mm_counts.get("image", 0) + video * mm_counts.get("video", 0)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Lfm3VlProcessingInfo,
    dummy_inputs=Lfm3VlDummyInputsBuilder,
)
class Lfm3VlForConditionalGeneration(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsPP
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return _short_conv_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        return _short_conv_state_shape(
            vllm_config, vllm_config.model_config.hf_config.text_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Lfm3VlConfig = vllm_config.model_config.hf_config
        self.config = config
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = Lfm3VlVisionModel(
                config.vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = Lfm3VlMultiModalProjector(
                config,
                quant_config=vllm_config.quant_config,
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

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        merge = self.vision_tower.spatial_merge_size**2
        embeddings: list[torch.Tensor] = []
        for pixels_key, grid_key, embeds_key in (
            ("pixel_values", "image_grid_thw", "image_embeds"),
            ("pixel_values_videos", "video_grid_thw", "video_embeds"),
        ):
            grid = kwargs.get(grid_key)
            if (pixels := kwargs.get(pixels_key)) is not None:
                assert isinstance(pixels, torch.Tensor)
                assert isinstance(grid, torch.Tensor)
                grid_thw = grid.tolist()
                features = self.multi_modal_projector(
                    self.vision_tower(pixels, grid_thw)
                )
                embeddings += features.split(
                    [t * h * w // merge for t, h, w in grid_thw]
                )
            elif (embeds := kwargs.get(embeds_key)) is not None:
                assert isinstance(embeds, torch.Tensor)
                assert isinstance(grid, torch.Tensor)
                embeddings += embeds.split((grid.prod(-1) // merge).tolist())
        return embeddings

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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
