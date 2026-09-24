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

A RADIO-specific patch generator (:class:`RadioPatchEmbeddings`) feeds an encoder
built from the shared InternViT blocks (:class:`InternVisionEncoder` /
:class:`InternVisionEncoderLayer` / :class:`InternParallelAttention`); the RADIO
subclasses extend ``forward`` for the dynamic-resolution and video features.
``load_weights`` maps existing RADIO checkpoint keys onto this module tree via
:data:`hf_to_vllm_mapper` so previously-saved checkpoints continue to load.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate, repeat
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.intern_vit import (
    InternParallelAttention,
    InternVisionEncoder,
    InternVisionEncoderLayer,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

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


class RadioParallelAttention(InternParallelAttention):
    def forward(
        self, x: torch.Tensor, mask_meta: MaskMetadata | None = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv(x)
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


class RadioVisionEncoderLayer(InternVisionEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, attn_cls=RadioParallelAttention, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_meta: MaskMetadata | None = None,
    ):
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states), mask_meta=mask_meta) * self.ls1
        )

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2

        return hidden_states


class RadioVisionEncoder(InternVisionEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, layer_cls=RadioVisionEncoderLayer, **kwargs)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        mask_meta: MaskMetadata | None = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, mask_meta=mask_meta)
        return hidden_states


class RadioModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    # Map existing RADIO checkpoint keys onto the current module tree so that
    # checkpoints saved for the previous modeling code still load.
    # ``video_patch_projection`` must precede ``embedder`` (applied in order);
    # keys mapping to ``None`` are intentionally not loaded (input normalization
    # runs in the processor, ``summary_idxs`` comes from the config, and
    # LayerScale stays at its identity init).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
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
            "radio_model.model.blocks": "encoder.layers",
            "radio_model.input_conditioner": None,
            "radio_model.summary_idxs": None,
            ".ls1": None,
            ".ls2": None,
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
        """Load RADIO checkpoint weights, remapping keys from the previous
        modeling code via :data:`hf_to_vllm_mapper`."""
        if isinstance(weights, dict):
            weights = weights.items()
        # Only radio_model.* tensors are ours; ignore any auxiliary keys a fuller
        # checkpoint may carry (an unmapped radio_model.* key still errors).
        weights = ((name, w) for name, w in weights if name.startswith("radio_model."))
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

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
