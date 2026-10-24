# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MuseGlimmer multimodal model for vLLM.

Native port of the MuseGlimmer text decoder (``MuseGlimmerForCausalLM``). The text
stack is a Gemma2 derivative with the
following MuseGlimmer-specific deltas, each of which is handled explicitly here:

  * SiLU-gated MLP (``hidden_activation="silu"``), not Gemma's gelu-tanh.
  * Scaleless RMSNorm on the token embeddings (no sqrt(hidden) scaling).
  * Per-layer sandwich RMSNorms with a baked ``+1`` weight offset
    (``x * (1 + w)``), matching Gemma, but with distinct eps for the
    pre/post norms (``rms_norm_eps`` vs ``post_norm_eps``).
  * QK-norm (weightless, fp32) applied *before* RoPE, followed by a query
    pre-scale of ``qk_scale_factor / sqrt(head_dim)``.
  * A per-head sigmoid attention output gate.
  * iRoPE layout: NoPE layers use full attention, RoPE layers use sliding
    window attention. RoPE is applied NEOX-style (``is_neox_style=True``):
    the HF converter (``convert_muse_glimmer_weights_to_hf.py``, 20260806+) permutes
    q/k into the half-split (NEOX) layout via ``_permute_for_rope`` so they
    pair with ``rotate_half`` — matching the reference's interleaved rotation
    on the *native* (unpermuted) weights. Serving the permuted HF weights with
    ``is_neox_style=False`` scrambles q/k and causes token-repetition collapse.
  * Final logits are pre-scaled by ``output_multiplier`` and then tanh
    soft-capped at ``final_logit_softcapping``.
  * Untied lm_head.

The vision path supports variable-resolution images and temporally patched
videos. It mirrors the checkpoint's native vision encoder, including sparse
block attention, 2-D RoPE, pixel-shuffle downsampling, and the two-layer
adapter/projection stack.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature, MuseGlimmerConfig, MuseGlimmerProcessor

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from .vision import get_fp8_padded_hidden_size, is_vit_use_data_parallel

logger = init_logger(__name__)


def _text_config(config):
    """MuseGlimmer checkpoints may nest the text config under ``text_config``
    (multimodal ``MuseGlimmerConfig``) or expose it directly
    (``MuseGlimmerTextConfig``)."""
    return getattr(config, "text_config", config)


def _muse_glimmer_has_vision(config) -> bool:
    return hasattr(config, "vision_config")


IMAGE_TOKEN = "<|patch|>"
VIDEO_TOKEN = "<|video|>"


class MuseGlimmerImagePixelInputs(TensorSchema):
    """Dimensions:
    - np: Total number of patches over all images
    - cps: Temporal patch size * channels * patch size * patch size
    - ni: Number of images
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class MuseGlimmerVideoPixelInputs(TensorSchema):
    """Dimensions:
    - np: Total number of patches over all videos
    - ctps: Temporal patch size * channels * patch size * patch size
    - nv: Number of videos
    """

    type: Literal["pixel_values_videos"]
    pixel_values_videos: Annotated[torch.Tensor, TensorShape("np", "ctps")]
    video_grid_thw: Annotated[torch.Tensor, TensorShape("nv", 3)]


class MuseGlimmerProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        if not _muse_glimmer_has_vision(self.get_hf_config()):
            return {}
        return {"image": None, "video": None}

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_hf_processor(self, **kwargs: object) -> MuseGlimmerProcessor:
        return self.ctx.get_hf_processor(MuseGlimmerProcessor, **kwargs)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        video_processor = processor.video_processor
        num_frames = self.get_num_frames_with_most_features(seq_len, mm_counts)
        return {
            "image": int(image_processor.max_image_tokens),
            "video": (
                num_frames
                // int(video_processor.temporal_patch_size)
                * int(video_processor.max_video_frame_tokens)
            ),
        }

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_hf_processor().image_processor
        grid_size = math.isqrt(int(image_processor.max_image_tokens))
        side = (
            int(image_processor.patch_size)
            * int(image_processor.merge_size)
            * grid_size
        )
        return ImageSize(width=side, height=side)

    def get_num_frames_with_most_features(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> int:
        video_processor = self.get_hf_processor().video_processor
        num_videos = max(mm_counts.get("video", 0), 1)
        groups = max(
            1,
            seq_len // num_videos // int(video_processor.max_video_frame_tokens),
        )
        temporal_patch_size = int(video_processor.temporal_patch_size)
        max_groups = max(1, int(video_processor.num_frames) // temporal_patch_size)
        return min(groups, max_groups) * temporal_patch_size


class MuseGlimmerDummyInputsBuilder(BaseDummyInputsBuilder[MuseGlimmerProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0) + VIDEO_TOKEN * mm_counts.get(
            "video", 0
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        processor = self.info.get_hf_processor()
        video_processor = processor.video_processor
        image_width, image_height = self.info.get_image_size_with_most_features()
        video_grid = math.isqrt(int(video_processor.max_video_frame_tokens))
        video_size = (
            int(video_processor.patch_size)
            * int(video_processor.merge_size)
            * video_grid
        )
        return {
            "image": self._get_dummy_images(
                width=image_width,
                height=image_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=video_size,
                height=video_size,
                num_frames=self.info.get_num_frames_with_most_features(
                    seq_len, mm_counts
                ),
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


class MuseGlimmerMultiModalProcessor(
    BaseMultiModalProcessor[MuseGlimmerProcessingInfo]
):
    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # vLLM samples video frames before processing
        hf_kwargs = {**hf_kwargs, "do_sample_frames": False}
        return super()._call_hf_processor(hf_data, hf_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thw.prod(-1)
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_thw.prod(-1)
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        config = self.info.get_hf_config()
        merge_length = config.vision_config.merge_size**2
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[IMAGE_TOKEN],
            "video": vocab[VIDEO_TOKEN],
        }
        image_start_id = vocab["<|image_start|>"]
        image_end_id = vocab["<|image_end|>"]
        video_start_id = vocab["<|vid_start|>"]
        video_end_id = vocab["<|vid_end|>"]
        video_separator_id = vocab["<|vid_frame_separator|>"]

        def image_replacement(item_idx: int) -> PromptUpdateDetails:
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            replacement = (
                [image_start_id] + [config.image_token_id] * num_tokens + [image_end_id]
            )
            return PromptUpdateDetails.select_token_id(
                replacement, config.image_token_id
            )

        def video_replacement(item_idx: int) -> PromptUpdateDetails:
            grid_thw = out_mm_kwargs["video"][item_idx]["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_groups = int(grid_thw[0])
            tokens_per_group = int(grid_thw[1:].prod()) // merge_length

            video_processor = processor.video_processor
            temporal_patch_size = int(video_processor.temporal_patch_size)
            sampling_fps = float(video_processor.fps)
            if sampling_fps <= 0:
                raise ValueError("MuseGlimmer video fps must be positive")

            replacement = [video_start_id]
            for group_idx in range(num_groups):
                timestamp = group_idx * temporal_patch_size / sampling_fps
                replacement.extend(
                    tokenizer.encode(
                        f"Time: {timestamp:.1f}s", add_special_tokens=False
                    )
                )
                replacement.extend([config.video_token_id] * tokens_per_group)
                replacement.append(
                    video_separator_id if group_idx < num_groups - 1 else video_end_id
                )
            return PromptUpdateDetails.select_token_id(
                replacement, config.video_token_id
            )

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder["image"]],
                replacement=image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[placeholder["video"]],
                replacement=video_replacement,
            ),
        ]


class MuseGlimmerRMSNorm(nn.Module):
    """RMSNorm mirroring HF MuseGlimmer exactly (fp32 compute, cast at the end).

    ``normed = _norm(x.float()) * (w.float() + weight_offset)`` cast back to the
    input dtype. When ``with_scale`` is False the layer is weightless (used for
    QK-norm and the token-embedding norm).
    """

    def __init__(
        self,
        dim: int | None = None,
        eps: float = 1e-6,
        with_scale: bool = True,
        weight_offset: int = 0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        self.weight_offset = weight_offset
        if with_scale:
            assert dim is not None
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self._norm(hidden_states.float())
        if self.with_scale:
            out = out * (self.weight.float() + self.weight_offset)
        return out.type_as(hidden_states)


class MuseGlimmerVisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("MuseGlimmer vision hidden size must divide num heads")

        self.total_num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_data_parallel = is_vit_use_data_parallel(num_heads)
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_heads = divide(num_heads, self.tp_size)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=self.use_data_parallel,
        )
        self.o_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            prefix=f"{prefix}.o_proj",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=True,
            enable_fp32_compute=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            hidden_states.shape[0],
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(1)
        query, key = self.apply_rotary_emb(
            torch.stack([query, key]).contiguous(),
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        ).unbind(0)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        output = self.attn(
            query=query,
            key=key,
            value=value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output = output.reshape(hidden_states.shape[0], -1)
        output, _ = self.o_proj(output)
        return output


class MuseGlimmerVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden_states)))


class MuseGlimmerVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = MuseGlimmerVisionAttention(
            hidden_size,
            num_heads,
            prefix=f"{prefix}.attn",
        )
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = MuseGlimmerVisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flattened = hidden_states.view(batch_size * seq_len, hidden_size)
        flattened = flattened + self.attn(
            self.ln_1(flattened),
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
        )
        flattened = flattened + self.mlp(self.ln_2(flattened))
        return flattened.view(batch_size, seq_len, hidden_size)


class MuseGlimmerVisionEncoder(nn.Module):
    def __init__(self, config: MuseGlimmerConfig, prefix: str = "") -> None:
        super().__init__()
        out_hidden_size = config.out_hidden_size
        config = config.vision_config
        hidden_size = config.hidden_size
        patch_dim = config.patch_temporal * 3 * config.patch_size**2
        self.hidden_size = hidden_size
        self.patch_size = config.patch_size
        self.patch_temporal = config.patch_temporal
        self.merge_size = config.merge_size
        self.pos_emb_height = config.pos_emb_height
        self.pos_emb_width = config.pos_emb_width
        self.head_dim = hidden_size // config.num_attention_heads
        self.layer_types = list(config.layer_types)
        if len(self.layer_types) != config.num_hidden_layers:
            raise ValueError(
                "MuseGlimmer vision layer_types must match num_hidden_layers"
            )
        if self.head_dim % 4:
            raise ValueError("MuseGlimmer vision head dimension must be divisible by 4")

        self.conv1_linear = nn.Linear(patch_dim, hidden_size, bias=False)
        self.positional_embedding_vlm = nn.Parameter(
            torch.zeros(config.pos_emb_height * config.pos_emb_width, hidden_size)
        )
        self.ln_pre = nn.LayerNorm(hidden_size)
        self.transformer = nn.ModuleList(
            [
                MuseGlimmerVisionBlock(
                    hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    prefix=f"{prefix}.transformer.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        vision_attention = self.transformer[0].attn
        self.tp_size = vision_attention.tp_size
        self.attn_backend = vision_attention.attn.attn_backend
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            config.num_attention_heads, self.head_dim
        )
        self.ln_post = nn.LayerNorm(hidden_size)

        expected_output = hidden_size * self.merge_size**2
        if out_hidden_size != expected_output:
            raise ValueError(
                f"MuseGlimmer out_hidden_size={out_hidden_size} does not match "
                f"pixel-shuffle output {expected_output}"
            )

    def _make_2d_rope(
        self, grid_height: int, grid_width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=device)
                / spatial_dim
            )
        )
        height = torch.arange(1, grid_height + 1, dtype=torch.float32, device=device)
        width = torch.arange(1, grid_width + 1, dtype=torch.float32, device=device)
        height = height.unsqueeze(1).expand(-1, grid_width).reshape(-1)
        width = width.unsqueeze(0).expand(grid_height, -1).reshape(-1)
        freq_w = torch.outer(width, inv_freq)
        freq_h = torch.outer(height, inv_freq)
        freqs = torch.cat([freq_w, freq_h], dim=-1)
        return torch.cos(freqs), torch.sin(freqs)

    def _get_pos_emb(
        self, grid_height: int, grid_width: int, device: torch.device
    ) -> torch.Tensor:
        h_grid = (
            torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5
        ) * (self.pos_emb_height / grid_height) - 0.5
        w_grid = (
            torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5
        ) * (self.pos_emb_width / grid_width) - 0.5
        h_floor = torch.floor(h_grid).long()
        w_floor = torch.floor(w_grid).long()
        h_ceil = h_floor + 1
        w_ceil = w_floor + 1
        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_valid = (h_floor >= 0) & (h_floor < self.pos_emb_height)
        h_ceil_valid = (h_ceil >= 0) & (h_ceil < self.pos_emb_height)
        w_floor_valid = (w_floor >= 0) & (w_floor < self.pos_emb_width)
        w_ceil_valid = (w_ceil >= 0) & (w_ceil < self.pos_emb_width)
        h_floor = h_floor.clamp(0, self.pos_emb_height - 1)
        h_ceil = h_ceil.clamp(0, self.pos_emb_height - 1)
        w_floor = w_floor.clamp(0, self.pos_emb_width - 1)
        w_ceil = w_ceil.clamp(0, self.pos_emb_width - 1)

        h_floor_offset = h_floor * self.pos_emb_width
        h_ceil_offset = h_ceil * self.pos_emb_width
        indices = torch.stack(
            [
                (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
                (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
                (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
            ]
        )
        weights = torch.stack(
            [
                (
                    (1 - h_frac)[:, None]
                    * (1 - w_frac)[None, :]
                    * (h_floor_valid[:, None] & w_floor_valid[None, :])
                ).flatten(),
                (
                    (1 - h_frac)[:, None]
                    * w_frac[None, :]
                    * (h_floor_valid[:, None] & w_ceil_valid[None, :])
                ).flatten(),
                (
                    h_frac[:, None]
                    * (1 - w_frac)[None, :]
                    * (h_ceil_valid[:, None] & w_floor_valid[None, :])
                ).flatten(),
                (
                    h_frac[:, None]
                    * w_frac[None, :]
                    * (h_ceil_valid[:, None] & w_ceil_valid[None, :])
                ).flatten(),
            ]
        )
        return (self.positional_embedding_vlm[indices] * weights[..., None]).sum(0)

    def _pixel_shuffle_downsample(
        self, hidden_states: torch.Tensor, grid_height: int, grid_width: int
    ) -> torch.Tensor:
        factor = self.merge_size
        output_tokens = (grid_height // factor) * (grid_width // factor)
        permutation = torch.arange(
            grid_height * grid_width, device=hidden_states.device
        )
        permutation = permutation.view(
            grid_height // factor, factor, grid_width // factor, factor
        )
        permutation = permutation.permute(0, 2, 1, 3).reshape(-1)
        hidden_states = hidden_states.squeeze(0)[permutation]
        hidden_size = hidden_states.shape[-1]
        hidden_states = (
            hidden_states.view(output_tokens, factor * factor, hidden_size)
            .permute(0, 2, 1)
            .contiguous()
            .view(output_tokens, hidden_size * factor * factor)
        )
        return hidden_states.unsqueeze(0)

    def _get_sparse_permutation(
        self, grid_height: int, grid_width: int, device: torch.device
    ) -> tuple[torch.Tensor, list[int]]:
        block_height = self.pos_emb_height
        block_width = self.pos_emb_width
        padded_height = math.ceil(grid_height / block_height) * block_height
        padded_width = math.ceil(grid_width / block_width) * block_width
        indices = torch.arange(grid_height * grid_width, device=device).view(
            grid_height, grid_width
        )
        indices = F.pad(
            indices,
            (0, padded_width - grid_width, 0, padded_height - grid_height),
            value=-1,
        ).flatten()
        indices = indices.view(
            padded_height // block_height,
            block_height,
            padded_width // block_width,
            block_width,
        )
        indices = indices.permute(0, 2, 1, 3).reshape(-1)
        valid = (indices != -1).view(-1, block_height * block_width)
        return indices[indices != -1], valid.sum(dim=1).tolist()

    def _get_attention_metadata(
        self,
        seq_lens: Sequence[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cu_seqlens_np = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.asarray(seq_lens, dtype=np.int32).cumsum(dtype=np.int32),
            ]
        )
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend,
            cu_seqlens_np,
            device,
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(
                self.attn_backend,
                cu_seqlens_np,
            ),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            self.hidden_size,
            self.tp_size,
            device,
            fp8_padded_hidden_size=self.fp8_padded_hidden_size,
        )
        return cu_seqlens, max_seqlen, sequence_lengths

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        has_sparse_layers = any(
            layer_type != "full_attention" for layer_type in self.layer_types
        )

        # Each temporal group of an image or video is encoded independently
        grids = [
            (grid_height, grid_width)
            for num_groups, grid_height, grid_width in grid_thw.tolist()
            for _ in range(num_groups)
        ]
        patch_groups = pixel_values.to(device=device, dtype=dtype).split(
            [grid_height * grid_width for grid_height, grid_width in grids]
        )

        all_hidden_states = []
        all_rotary_pos_emb_cos = []
        all_rotary_pos_emb_sin = []
        sparse_seq_lens: list[int] = []
        global_seq_lens: list[int] = []
        metadata = []
        for patches, (grid_height, grid_width) in zip(patch_groups, grids):
            if grid_height % self.merge_size or grid_width % self.merge_size:
                raise ValueError(
                    "MuseGlimmer vision grid must be divisible by the merge size"
                )
            num_tokens = grid_height * grid_width
            hidden_states = self.conv1_linear(patches.unsqueeze(0))
            hidden_states = hidden_states + self._get_pos_emb(
                grid_height, grid_width, device
            ).unsqueeze(0).to(dtype)
            hidden_states = self.ln_pre(hidden_states.view(-1, self.hidden_size)).view(
                1, -1, self.hidden_size
            )
            rotary_pos_emb_cos, rotary_pos_emb_sin = self._make_2d_rope(
                grid_height, grid_width, device
            )

            permutation = None
            if has_sparse_layers:
                permutation, seq_lens = self._get_sparse_permutation(
                    grid_height, grid_width, device
                )
                hidden_states = hidden_states[:, permutation]
                rotary_pos_emb_cos = rotary_pos_emb_cos[permutation]
                rotary_pos_emb_sin = rotary_pos_emb_sin[permutation]
                sparse_seq_lens.extend(seq_lens)

            all_hidden_states.append(hidden_states.squeeze(0))
            all_rotary_pos_emb_cos.append(rotary_pos_emb_cos)
            all_rotary_pos_emb_sin.append(rotary_pos_emb_sin)
            global_seq_lens.append(num_tokens)
            metadata.append((grid_height, grid_width, num_tokens, permutation))

        hidden_states = torch.cat(all_hidden_states).unsqueeze(0)
        rotary_pos_emb_cos = torch.cat(all_rotary_pos_emb_cos)
        rotary_pos_emb_sin = torch.cat(all_rotary_pos_emb_sin)
        global_attention_metadata = self._get_attention_metadata(
            global_seq_lens,
            device,
        )
        sparse_attention_metadata = (
            self._get_attention_metadata(sparse_seq_lens, device)
            if sparse_seq_lens
            else None
        )
        for layer_type, block in zip(self.layer_types, self.transformer):
            attention_metadata = (
                global_attention_metadata
                if layer_type == "full_attention"
                else sparse_attention_metadata
            )
            if attention_metadata is None:
                raise ValueError("MuseGlimmer sparse attention metadata is missing")
            hidden_states = block(
                hidden_states,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                *attention_metadata,
            )

        features = []
        offset = 0
        for grid_height, grid_width, num_tokens, permutation in metadata:
            item = hidden_states[:, offset : offset + num_tokens]
            offset += num_tokens
            if permutation is not None:
                inverse = torch.empty_like(permutation)
                inverse[permutation] = torch.arange(len(permutation), device=device)
                item = item[:, inverse]
            item = self.ln_post(item.view(-1, self.hidden_size)).view(
                1, -1, self.hidden_size
            )
            features.append(
                self._pixel_shuffle_downsample(item, grid_height, grid_width).squeeze(0)
            )
        return torch.cat(features)


class MuseGlimmerVisionAdapter(nn.Module):
    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(
            config.out_hidden_size, config.projector_hidden_size, bias=False
        )
        self.c_proj = nn.Linear(
            config.projector_hidden_size, config.projector_hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.c_proj(F.gelu(self.c_fc(hidden_states))))


class MuseGlimmerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
        if hidden_activation != "silu":
            raise ValueError(
                f"MuseGlimmer uses `silu` as the hidden activation; "
                f"got `{hidden_activation}`."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = extract_layer_index(prefix)

        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # MuseGlimmer overrides Gemma2's query_pre_attn_scalar scaling with the
        # standard 1/sqrt(head_dim). The query is *additionally* pre-scaled by
        # scale_query_by after QK-norm (see forward).
        self.scaling = self.head_dim**-0.5

        # iRoPE: layers with a zero RoPE theta are NoPE
        self.use_rope = bool(config.layer_rope_theta[self.layer_idx])

        # Weightless, computed in fp32, applied per head over head_dim.
        self.qk_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)
        self.scale_query_by = config.qk_scale_factor

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.output_gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate_proj",
        )

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
                # HF converter permutes q/k to NEOX (half-split)
                # layout via _permute_for_rope
                is_neox_style=True,
            )
            if self.use_rope
            else None
        )

        is_sliding = config.layer_types[self.layer_idx] == "sliding_attention"
        sliding_window = config.sliding_window if is_sliding else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=None,  # MuseGlimmer sets attn_logit_softcapping = None
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm over head_dim, fp32, applied BEFORE RoPE; then pre-scale q.
        q = q.reshape(-1, self.head_dim)
        q = self.qk_norm(q).reshape(-1, self.q_size) * self.scale_query_by
        k = k.reshape(-1, self.head_dim)
        k = self.qk_norm(k).reshape(-1, self.kv_size)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        # Gate reads the layer input hidden states (not the attn output).
        gate, _ = self.output_gate_proj(hidden_states)
        attn_output = torch.sigmoid(gate) * attn_output

        output, _ = self.o_proj(attn_output)
        return output


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = MuseGlimmerAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MuseGlimmerMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Sandwich norms with baked +1 offset. Pre-norms use rms_norm_eps; the
        # post-norms use the (typically smaller) post_norm_eps.
        self.input_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_offset=1
        )
        self.post_attention_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, eps=config.post_norm_eps, weight_offset=1
        )
        self.pre_feedforward_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_offset=1
        )
        self.post_feedforward_layernorm = MuseGlimmerRMSNorm(
            config.hidden_size, eps=config.post_norm_eps, weight_offset=1
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gemma2-style sandwich, replicated explicitly (matches HF MuseGlimmer).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, residual


@support_torch_compile
class MuseGlimmerModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # MuseGlimmer normalizes token embeddings with a weightless RMSNorm instead of
        # Gemma's sqrt(hidden_size) multiplier.
        self.embed_norm = MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MuseGlimmerDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        # Final norm: weight-as-scale, no offset.
        self.norm = MuseGlimmerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, None
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, None
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    MuseGlimmerMultiModalProcessor,
    info=MuseGlimmerProcessingInfo,
    dummy_inputs=MuseGlimmerDummyInputsBuilder,
)
class MuseGlimmerForCausalLM(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    # Weight-name normalization. Two checkpoint conventions are supported:
    #
    #   * HF MuseGlimmer export (``convert_muse_glimmer_weights_to_hf.py``): the
    #     multimodal ``MuseGlimmerConfig`` prefixes the language model with
    #     ``model.language_model.``; the per-layer sandwich norms are already
    #     named ``input_layernorm`` / ``post_attention_layernorm`` /
    #     ``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``.
    #
    #   * Legacy HF export (an earlier checkpoint convention): uses ``model.``
    #     and a different sandwich-norm naming where ``post_attn_norm`` is the
    #     true post-attention norm and ``post_attention_layernorm`` is actually
    #     the pre-feedforward norm. We remap those to MuseGlimmer's names.
    #
    # CONVENTION DISAMBIGUATION (critical): the two checkpoint families use
    # DIFFERENT sandwich-norm names, and they must not be conflated:
    #
    #   * Canonical MuseGlimmer export (current
    #     ``convert_muse_glimmer_weights_to_hf.py`` — what partners ship):
    #     keys are ``model.language_model.layers.N.*`` and
    #     the norms are ALREADY named ``input_layernorm`` /
    #     ``post_attention_layernorm`` / ``pre_feedforward_layernorm`` /
    #     ``post_feedforward_layernorm``. No norm rename needed — pass through.
    #
    #   * Legacy HF export (an earlier checkpoint convention): keys are
    #     ``model.layers.N.*`` and the sandwich norms are
    #     named ``input_layernorm`` / ``post_attention_layernorm`` (this one is
    #     actually the PRE-feedforward norm) / ``post_attn_norm`` (the true
    #     post-attention norm) / ``post_ffn_norm``. These must be remapped.
    #
    # The unambiguous discriminator is the PREFIX: legacy keys start with
    # ``model.layers.`` while canonical keys start with
    # ``model.language_model.layers.``. ``orig_to_new_regex`` runs BEFORE the
    # prefix strip (see WeightsMapper._map_name_with_shard), so we anchor the
    # legacy renames on ``^model\.layers\.`` — they fire ONLY on legacy keys and
    # leave canonical/partner checkpoints untouched. Rule order within the regex
    # dict matters: the ``post_attention_layernorm`` -> ``pre_feedforward_...``
    # rule must precede the ``post_attn_norm`` -> ``post_attention_layernorm``
    # rule so the latter's output is not re-captured by the former (regex rules
    # apply as a single forward pass).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.vision_tower.patch_embedder.position_embedding_table.weight": (
                "model.vision_tower.positional_embedding_vlm"
            ),
            "model.vision_tower.layers.": "model.vision_tower.transformer.",
            ".norm1.": ".ln_1.",
            ".norm2.": ".ln_2.",
            ".attn.proj.": ".attn.o_proj.",
            ".mlp.fc1.": ".mlp.c_fc.",
            ".mlp.fc2.": ".mlp.c_proj.",
            ".self_attn.gate_proj": ".self_attn.output_gate_proj",
        },
        orig_to_new_prefix={
            "model.rotary_emb.": None,
            "model.language_model.": "model.",
            "language_model.": "model.",
            "model.vision_tower.patch_embedder.patch_embedding.": (
                "model.vision_tower.conv1_linear."
            ),
            "model.vision_tower.": "vision_encoder.",
            "vision_tower.": "vision_encoder.",
            "model.vision_encoder.": "vision_encoder.",
            "model.vision_adapter.fc1.": "model.vision_adapter.c_fc.",
            "model.vision_adapter.fc2.": "model.vision_adapter.c_proj.",
            "model.vision_adapter.": "vision_adapter.",
            "model.vision_projection.": "vision_projection.",
            "model.perception_emb_norm.": "perception_emb_norm.",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        if modality.startswith("video"):
            return VIDEO_TOKEN
        raise ValueError(f"Unsupported MuseGlimmer modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = _text_config(config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        self.has_vision = _muse_glimmer_has_vision(config)

        with self._mark_language_model(vllm_config):
            self.model = MuseGlimmerModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.vision_encoder: MuseGlimmerVisionEncoder | None
        self.vision_adapter: MuseGlimmerVisionAdapter | None
        self.perception_emb_norm: MuseGlimmerRMSNorm | None
        if self.has_vision:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.vision_encoder = MuseGlimmerVisionEncoder(
                    config,
                    prefix=maybe_prefix(prefix, "vision_encoder"),
                )
                self.vision_adapter = MuseGlimmerVisionAdapter(config)
                self.vision_projection = nn.Linear(
                    config.projector_hidden_size,
                    text_config.hidden_size,
                    bias=False,
                )
                self.perception_emb_norm = MuseGlimmerRMSNorm(
                    eps=text_config.rms_norm_eps, with_scale=False
                )
        else:
            self.vision_encoder = None
            self.vision_adapter = None
            self.vision_projection = None
            self.perception_emb_norm = None

        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.output_multiplier = text_config.output_multiplier
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.logits_processor = LogitsProcessor(text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.configure_mm_token_handling(
            text_config.vocab_size,
            [
                getattr(config, "image_token_id", 200092),
                getattr(config, "video_token_id", 200091),
            ],
        )

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> MuseGlimmerImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None
        return MuseGlimmerImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=kwargs.pop("image_grid_thw", None),
        )

    def _parse_and_validate_video_input(
        self,
        **kwargs: object,
    ) -> MuseGlimmerVideoPixelInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            return None
        return MuseGlimmerVideoPixelInputs(
            type="pixel_values_videos",
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=kwargs.pop("video_grid_thw", None),
        )

    def _encode_pixel_values(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if (
            self.vision_encoder is None
            or self.vision_adapter is None
            or self.vision_projection is None
            or self.perception_emb_norm is None
        ):
            raise ValueError("This MuseGlimmer checkpoint has no vision tower")
        features = self.vision_encoder(pixel_values, grid_thw)
        features = self.vision_adapter(features)
        features = self.vision_projection(features)
        features = self.perception_emb_norm(features)
        merge_length = self.vision_encoder.merge_size**2
        return features.split((grid_thw.prod(-1) // merge_length).tolist())

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        video_input = self._parse_and_validate_video_input(**kwargs)

        embeddings: list[torch.Tensor] = []
        for key in kwargs:
            if key == "pixel_values" and image_input is not None:
                embeddings.extend(
                    self._encode_pixel_values(
                        image_input["pixel_values"], image_input["image_grid_thw"]
                    )
                )
            elif key == "pixel_values_videos" and video_input is not None:
                embeddings.extend(
                    self._encode_pixel_values(
                        video_input["pixel_values_videos"],
                        video_input["video_grid_thw"],
                    )
                )
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models"""
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["vision_adapter.", "vision_projection."],
            tower_model="vision_encoder.",
        )
