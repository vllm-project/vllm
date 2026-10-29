# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/NVlabs/RADIO and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/radio/modeling_radio.py
# --------------------------------------------------------
# RADIO
# Copyright (c) 2023-2026, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# --------------------------------------------------------
"""RADIO vision encoder, matching the ``RadioModel`` in the Transformers library.

The module tree mirrors the Transformers ``RadioModel`` (``embeddings`` and an
``encoder.layer`` stack of ``norm1``/``attention``/``layer_scale1``/``norm2``/
``mlp``/``layer_scale2`` blocks). The vLLM-specific deviations are the fused
``attention.qkv`` (a :class:`~vllm.model_executor.layers.linear.QKVParallelLinear`
for TP/quantization, vs the split ``query``/``key``/``value`` in HF) and running
input normalization in the processor instead of an ``input_conditioner`` module;
the attention backend and MLP reuse vLLM's :class:`MMEncoderAttention` and
:class:`~vllm.model_executor.models.intern_vit.InternMLP`. ``load_weights`` maps
both native Transformers and legacy remote-code checkpoint keys onto this tree
via :data:`hf_to_vllm_mapper`.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, repeat
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedConfig

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.intern_vit import InternMLP
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.models.vision import is_vit_use_data_parallel

logger = init_logger(__name__)

input_dim_t: TypeAlias = int | tuple[int, int]
norm_t: TypeAlias = tuple[float, float, float] | torch.Tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def calc_seq_len(size: tuple[int, int], patch_size: int) -> int:
    h, w = size
    return (h // patch_size) * (w // patch_size)


def calc_seq_lens(sizes: list[tuple[int, int]], patch_size: int) -> list[int]:
    return [calc_seq_len(size, patch_size) for size in sizes]


class RadioPatchEmbeddings(nn.Module):
    """Cropped Position Embedding (CPE) patch generator, matching Transformers'
    ``RadioPatchEmbeddings``, with additional support for dynamic multi-resolution
    packing and video temporal compression.
    """

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        patch_size = config.patch_size
        embed_dim = config.hidden_size
        num_channels = config.num_channels

        max_input_dims = to_2tuple(config.max_img_size)
        max_input_dims = tuple(
            int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims
        )
        input_dims = to_2tuple(config.image_size)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cls_tokens = config.num_cls_tokens
        self.num_registers = config.num_registers
        self.temporal_patch_size = config.video_temporal_patch_size

        self.cpe_mode = max_input_dims != input_dims
        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols

        self.im_to_patches = Im2Patches(patch_size)
        self.patch_projection = ViTPatchLinear(
            patch_size, embed_dim, num_channels=num_channels, bias=False
        )

        if self.temporal_patch_size > 1:
            if not config.separate_video_embedder:
                raise NotImplementedError(
                    "Only separate_video_embedder=True is supported for"
                    " temporal compression (video_temporal_patch_size > 1)"
                )
            self.video_patch_projection = ViTPatchLinear(
                patch_size,
                embed_dim,
                num_channels=num_channels,
                bias=False,
                temporal_patch_size=self.temporal_patch_size,
            )

        scale = embed_dim**-0.5
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * scale
        )
        self.cls_register_token = nn.Parameter(
            torch.randn(self.num_cls_tokens + self.num_registers, embed_dim) * scale
        )
        self.patch_normalizer = nn.Identity()

    @property
    def num_skip(self) -> int:
        return self.num_cls_tokens + self.num_registers

    def forward(
        self, x: torch.Tensor, imgs_sizes: list[tuple[int, int]] | None = None
    ) -> torch.Tensor:
        if imgs_sizes is not None:
            # Dynamic multi-resolution: ``x`` is already patchified and packed
            # to [B, total_patches, C*P*P].
            patches = self.patch_projection(x)
            patches = self._apply_pos_enc_dynamic(patches, imgs_sizes=imgs_sizes)
            patches = self._cls_token_dynamic(patches, imgs_sizes=imgs_sizes)
        else:
            patches = self.patch_projection(self.im_to_patches(x))
            patches = self._apply_pos_enc(patches, input_size=x.shape[2:])
            patches = self._prepend_cls_token(patches)
        return self.patch_normalizer(patches)

    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """Process video frames with temporal compression.

        Groups T consecutive frames into tubelets before embedding.

        Args:
            x: [num_frames, 3, H, W] tensor of video frames

        Returns:
            Embedded patches with temporal compression applied.

        """
        assert self.temporal_patch_size > 1
        T = self.temporal_patch_size
        input_size = x.shape[2:]

        patches = self.im_to_patches(x)  # [N, num_patches, 3*P*P]
        num_frames, num_spatial, feat_dim = patches.shape

        # Pad to a multiple of T by repeating the last frame so that
        # all tubelets have exactly T frames.
        num_pad_frames = (-num_frames) % T
        if num_pad_frames > 0:
            last_frame_dup = patches[-1:].expand(num_pad_frames, -1, -1)
            patches = torch.cat([patches, last_frame_dup], dim=0)

        # Group T frames per tubelet: for each spatial position, concatenate
        #   features across T consecutive frames; order follows Megatron training
        num_frames_padded = patches.shape[0]
        num_tublets = num_frames_padded // T
        patches = rearrange(
            patches,
            "(tubelets frames) spatial feat -> tubelets spatial (frames feat)",
            tubelets=num_tublets,
            frames=T,
            spatial=num_spatial,
            feat=feat_dim,
        )

        patches = self.video_patch_projection(patches)
        patches = self._apply_pos_enc(patches, input_size=input_size)
        patches = self._prepend_cls_token(patches)
        return self.patch_normalizer(patches)

    def _prepend_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        token = self.cls_register_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)

    def _cls_token_dynamic(
        self, patches: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        out = []
        current_length = 0
        for seq_len in calc_seq_lens(imgs_sizes, self.patch_size):
            class_token = self.cls_register_token.unsqueeze(0).expand(
                patches.shape[0], -1, -1
            )
            out.append(class_token)
            out.append(patches[:, current_length : current_length + seq_len, :])
            current_length += seq_len
        return torch.cat(out, dim=1)

    def _apply_pos_enc(
        self,
        patches: torch.Tensor,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        pos_enc = self._get_pos_enc(patches.shape[0], input_size=input_size)
        return patches + pos_enc

    def _apply_pos_enc_dynamic(
        self, patches: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        current_length = 0
        for size in imgs_sizes:
            seq_length = calc_seq_len(size, self.patch_size)
            img_patches = patches[:, current_length : current_length + seq_length, :]
            pos_enc = self._get_pos_enc(patches.shape[0], input_size=size)
            img_patches_with_pos = img_patches + pos_enc
            patches = torch.cat(
                [
                    patches[:, :current_length, :],
                    img_patches_with_pos,
                    patches[:, current_length + seq_length :, :],
                ],
                dim=1,
            )
            current_length += seq_length
        return patches

    def _get_pos_enc(
        self,
        batch_size: int,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)
        return self._get_pos_embeddings(batch_size, input_dims)

    def _get_pos_embeddings(self, batch_size: int, input_dims: tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.position_embedding

        pos_embed = self.position_embedding.reshape(
            1, self.num_rows, self.num_cols, -1
        ).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            max_dim = max(input_dims)
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=(max_dim, max_dim),
                align_corners=False,
                mode="bilinear",
            ).to(pos_embed.dtype)

            pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


class Im2Patches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_size,
            px=px,
            xx=self.patch_size,
        )
        return patches


class ViTPatchLinear(nn.Linear):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        num_channels: int = 3,
        bias: bool = False,
        temporal_patch_size: int = 1,
        **factory,
    ):
        super().__init__(
            num_channels * temporal_patch_size * (patch_size**2),
            embed_dim,
            bias=bias,
            **factory,
        )
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size


@dataclass(frozen=True, kw_only=True)
class MaskMetadata:
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor


class RadioAttention(nn.Module):
    """RADIO self-attention: a fused ``qkv`` projection feeding the vLLM
    attention backend, with optional QK-normalization and dynamic-resolution
    packed attention (via ``mask_meta``). The fused counterpart of the
    Transformers ``RadioAttention`` (which keeps ``query``/``key``/``value`` and
    ``output.dense`` split); ``load_weights`` bridges the two."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim="
                f"{self.embed_dim} and num_heads={self.num_heads})."
            )

        use_data_parallel = is_vit_use_data_parallel()
        # Disable attention TP when the head count is not divisible by tp_size.
        tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        use_data_parallel = (
            use_data_parallel or (self.num_heads + num_dummy_heads) % tp_size != 0
        )
        self.tp_size = 1 if use_data_parallel else tp_size
        self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()

        # Dummy heads pad the head count so TP divides evenly on common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.num_heads_per_partition = divide(
            num_dummy_heads + self.num_heads, self.tp_size
        )
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.qk_normalization = config.qk_normalization
        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
            self.k_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            self.num_heads_per_partition,
            self.head_dim,
            self.scale,
            prefix=f"{prefix}.attn",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(
        self, hidden_states: torch.Tensor, mask_meta: MaskMetadata | None = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        cu_seqlens, max_seqlen = None, None
        if mask_meta is not None:
            cu_seqlens = mask_meta.cu_seqlens
            max_seqlen = mask_meta.max_seqlen
        out = self.attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out, _ = self.proj(out)
        return out


class RadioLayerScale(nn.Module):
    """LayerScale matching the Transformers ``RadioLayerScale``: a ``lambda1``
    parameter scaling the residual branch."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(
            config.layerscale_value * torch.ones(config.hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class RadioLayer(nn.Module):
    """RADIO transformer block, mirroring the Transformers ``RadioLayer``
    (``norm1`` / ``attention`` / ``layer_scale1`` / ``norm2`` / ``mlp`` /
    ``layer_scale2``). ``drop_path`` is an inference-time ``Identity``; the MLP
    reuses :class:`InternMLP` (``fc1``/``fc2``, matching ``RadioMLP``)."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = RadioAttention(
            config,
            quant_config=quant_config,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.attention",
        )
        self.layer_scale1 = RadioLayerScale(config)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = InternMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.layer_scale2 = RadioLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_meta: MaskMetadata | None = None,
    ) -> torch.Tensor:
        attn_out = self.attention(self.norm1(hidden_states), mask_meta=mask_meta)
        hidden_states = self.drop_path(self.layer_scale1(attn_out)) + hidden_states
        mlp_out = self.mlp(self.norm2(hidden_states))
        hidden_states = self.drop_path(self.layer_scale2(mlp_out)) + hidden_states
        return hidden_states


class RadioVisionEncoder(nn.Module):
    """RADIO encoder stack, mirroring the Transformers ``RadioEncoder`` module
    tree (a ``layer`` ModuleList of :class:`RadioLayer`)."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override
        self.layer = nn.ModuleList(
            [
                RadioLayer(
                    config,
                    quant_config=quant_config,
                    num_dummy_heads=num_dummy_heads,
                    prefix=f"{prefix}.layer.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        mask_meta: MaskMetadata | None = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layer:
            hidden_states = encoder_layer(hidden_states, mask_meta=mask_meta)
        return hidden_states


class RadioModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    # Map both RADIO checkpoint layouts onto this module tree, which mirrors the
    # Transformers ``RadioModel`` (``embeddings.*`` and ``encoder.layer.N.*``):
    #   * native Transformers checkpoints load onto it directly; only the split
    #     attention projections fuse into ``attention.qkv`` via
    #     ``orig_to_new_stacked`` and the output projection is renamed;
    #   * legacy remote-code checkpoints nest weights under
    #     ``radio_model.model.{patch_generator,blocks}.*`` with a fused
    #     ``attn.qkv`` (renamed to ``attention.qkv`` and loaded directly).
    # ``video_patch_projection`` must precede ``embedder`` (applied in order);
    # keys mapping to ``None`` are intentionally not loaded (input normalization
    # runs in the processor and ``summary_idxs`` comes from the config).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # legacy remote-code layout onto the native module tree
            "radio_model.model.patch_generator.video_embedder": (
                "embeddings.video_patch_projection"
            ),
            "radio_model.model.patch_generator.embedder": (
                "embeddings.patch_projection"
            ),
            "radio_model.model.patch_generator.pos_embed": (
                "embeddings.position_embedding"
            ),
            "radio_model.model.patch_generator.cls_token.token": (
                "embeddings.cls_register_token"
            ),
            "radio_model.model.blocks": "encoder.layer",
            # legacy names onto the native tree: ``attn`` -> ``attention`` (its
            # fused ``qkv``/``proj`` load directly) and bare ``ls1``/``ls2`` ->
            # the ``layer_scale{1,2}`` LayerScale modules.
            ".attn.": ".attention.",
            ".ls1": ".layer_scale1.lambda1",
            ".ls2": ".layer_scale2.lambda1",
            # native attention output projection: ``attention.output.dense`` is
            # current at transformers 5.x (Dinov2 naming); ``attention.o_proj``
            # covers the post-refactor spelling on transformers main.
            "attention.output.dense": "attention.proj",
            "attention.o_proj": "attention.proj",
            # dropped in both layouts: normalization runs in the processor and
            # ``summary_idxs`` comes from the config.
            "input_conditioner": None,
            "summary_idxs": None,
        },
        orig_to_new_stacked={
            # Native split q/k/v projections fuse into the ``qkv``
            # QKVParallelLinear. ``attention.attention.{query,key,value}`` is
            # current at transformers 5.x; ``attention.{q,k,v}_proj`` covers the
            # post-refactor spelling on transformers main.
            "attention.attention.query": ("attention.qkv", "q"),
            "attention.attention.key": ("attention.qkv", "k"),
            "attention.attention.value": ("attention.qkv", "v"),
            "attention.q_proj": ("attention.qkv", "q"),
            "attention.k_proj": ("attention.qkv", "k"),
            "attention.v_proj": ("attention.qkv", "v"),
        },
    )

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.temporal_patch_size = config.video_temporal_patch_size

        self.embeddings = RadioPatchEmbeddings(config)
        self.encoder = RadioVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
        )

        # summary_idxs selects which class tokens to gather: None (no teachers)
        # keeps all class tokens; a list (possibly empty) gathers those indices.
        if config.summary_idxs is not None:
            self.register_buffer(
                "summary_idxs", torch.tensor(config.summary_idxs, dtype=torch.long)
            )
        else:
            self.summary_idxs = None

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        *,
        imgs_sizes: list[tuple[int, int]] | None = None,
        num_frames: int | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        T = self.temporal_patch_size

        mask_meta = None
        packed_batch_size = None  # Original batch size before packing.

        if num_frames is not None and T > 1:
            # Conv3d video: all tubelets have the same sequence length.
            # Pack [num_tubelets, seq_per_tubelet, hidden] -> [1, total, hidden].
            hidden_states = self.embeddings.forward_video(pixel_values)
            packed_batch_size, seq_per_tubelet, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.reshape(1, -1, hidden_dim)
            mask_meta = self._mask_metadata_from_seq_lens(
                [seq_per_tubelet] * packed_batch_size, device=hidden_states.device
            )
        else:
            hidden_states = self.embeddings(pixel_values, imgs_sizes=imgs_sizes)
            if imgs_sizes is not None and len(imgs_sizes) > 1:
                # Dynamic resolution w/ > 1 image, create attn mask.
                mask_meta = self._inter_image_mask_metadata(
                    imgs_sizes, device=hidden_states.device
                )

        encoder_outputs = self.encoder(hidden_states, mask_meta=mask_meta)

        # Unpack back to original batch shape if we packed for video.
        if packed_batch_size is not None:
            encoder_outputs = encoder_outputs.reshape(
                packed_batch_size, seq_per_tubelet, -1
            )

        return self._extract_final(encoder_outputs, imgs_sizes=imgs_sizes)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load RADIO checkpoint weights, remapping legacy remote-code and
        native Transformers keys onto this module tree via
        :data:`hf_to_vllm_mapper`."""
        if isinstance(weights, dict):
            weights = weights.items()
        # Keep only tensors belonging to the RADIO tower (legacy ``radio_model.*``
        # or native top-level keys); ignore any auxiliary keys a fuller checkpoint
        # may carry (an unmapped tower key still errors).
        prefixes = (
            "radio_model.",
            "embeddings.",
            "encoder.",
            "input_conditioner.",
            "summary_idxs",
        )
        weights = ((name, w) for name, w in weights if name.startswith(prefixes))
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if not loaded:
            logger.warning(
                "RadioModel.load_weights matched no checkpoint tensors; the "
                "vision tower will keep its random initialization. Expected "
                "legacy 'radio_model.*' or native 'embeddings.*'/'encoder.*' "
                "keys."
            )
        return loaded

    def _extract_final(
        self, y: torch.Tensor, imgs_sizes: list[tuple[int, int]] | None = None
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # Remove CLS + REGISTER tokens.
        num_skip = self.embeddings.num_skip
        patch_size = self.embeddings.patch_size
        num_cls_tokens = self.embeddings.num_cls_tokens
        if imgs_sizes is None:
            all_summary = y[:, :num_cls_tokens]
            all_feat = y[:, num_skip:]
        else:
            all_patches = []
            summaries = []
            current_pos = 0
            for num_patches in calc_seq_lens(imgs_sizes, patch_size):
                patches = y[
                    :, current_pos + num_skip : current_pos + num_skip + num_patches, :
                ]
                all_patches.append(patches)
                summary = y[:, current_pos : current_pos + num_cls_tokens, :]
                summaries.append(summary)
                current_pos += num_skip + num_patches
            all_summary = torch.cat(summaries, dim=1)
            all_feat = torch.cat(all_patches, dim=1)

        if self.summary_idxs is not None:
            bb_summary = all_summary[:, self.summary_idxs]
        else:
            bb_summary = all_summary
        return bb_summary.flatten(1), all_feat

    def get_input_embeddings(self):
        return self.embeddings

    def _inter_image_mask_metadata(
        self, imgs_sizes: list[tuple[int, int]], device: torch.device
    ) -> MaskMetadata:
        """Build mask metadata from image pixel sizes. Adds ``num_skip`` to each
        sequence length (cls/register tokens) to match patch generator output."""
        patch_size = self.embeddings.patch_size
        num_skip = self.embeddings.num_skip
        seq_lens = calc_seq_lens(imgs_sizes, patch_size)
        adjusted = [s + num_skip for s in seq_lens]
        return self._mask_metadata_from_seq_lens(adjusted, device=device)

    def _mask_metadata_from_seq_lens(
        self, seq_lens: list[int], device: torch.device
    ) -> MaskMetadata:
        """Build mask metadata from sequence lengths (already including
        cls/register tokens, i.e. ``patch_count + num_skip`` per item)."""
        assert len(seq_lens) > 0
        cu_seqlens = torch.tensor(
            list(accumulate(seq_lens, initial=0)), dtype=torch.int32, device=device
        )
        # Keep max_seqlen on CPU to avoid .item() sync.
        # See: https://github.com/vllm-project/vllm/blob/20b6b01/vllm/v1/attention/ops/vit_attn_wrappers.py#L48
        max_seqlen = torch.tensor(max(seq_lens), dtype=torch.int32)
        return MaskMetadata(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
