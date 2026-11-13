# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 vision encoder: a PE-Lang-style ViT with a 2x2 merge adapter.

Module names mirror the checkpoint, so `model.visual.<name>` loads into
`Inkling2VisionEncoder.<name>` unchanged, except that each layer's
`q_proj`/`k_proj`/`v_proj` load into one fused `qkv_proj`.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.inkling2.ops.vision import (
    apply_rotary_qk_,
    embed_tokens,
    gather_patches,
    scaled_residual_norm_,
)
from vllm.third_party.tml_fa4 import flash_attn_varlen_func

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# The image processor's CLIP std (float32-rounded constants): pixels are first
# CLIP-normalized with it and rounded to bf16, then mapped to [-1, 1].
_PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, dtype=torch.bfloat16)


class VisionMetadata(NamedTuple):
    # int32 [T, 4] (y, x, height, width) in native patches; CLS: y = x = -1.
    meta: torch.Tensor
    # Per token row, its row in the patch embeddings (unused on CLS rows).
    patch_rows: torch.Tensor
    # Per native patch, the offset of its top-left pixel.
    pixel_offsets: torch.Tensor
    # Token rows in 2x2-merge order, for the adapter.
    pool_rows: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv_proj = _linear(width, 3 * width)
        self.out_proj = _linear(width, width)

    def forward(self, hidden: torch.Tensor, metadata: VisionMetadata) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        apply_rotary_qk_(qkv, metadata.meta, self.num_heads)
        q, k, v = qkv.view(-1, 3, self.num_heads, self.head_dim).unbind(1)
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=metadata.cu_seqlens,
            cu_seqlens_k=metadata.cu_seqlens,
            max_seqlen_q=metadata.max_seqlen,
            max_seqlen_k=metadata.max_seqlen,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
        )
        return self.out_proj(out.view(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = _linear(width, intermediate_size)
        self.c_proj = _linear(intermediate_size, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(
        self,
        hidden: torch.Tensor,
        normed: torch.Tensor,
        metadata: VisionMetadata,
        next_norm: nn.LayerNorm | None,
    ) -> torch.Tensor | None:
        """Updates the residual `hidden` in place; `normed = ln_1(hidden)`.

        Returns `next_norm(hidden)` (the next layer's `ln_1`), if any.
        """
        attn = self.self_attn(normed, metadata)
        normed = scaled_residual_norm_(hidden, attn, self.ls_1.gamma, self.ln_2)
        mlp = self.mlp(normed)
        return scaled_residual_norm_(hidden, mlp, self.ls_2.gamma, next_norm)


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = _linear(
            3 * config["native_patch_size"] ** 2, width, bias=False
        )
        self.position_embedding = nn.Parameter(
            torch.empty(grid * grid, width, dtype=torch.bfloat16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16)
        )


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"], dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            Inkling2VisionLayer(config) for _ in range(config["num_hidden_layers"])
        )


class Inkling2VisionEncoder(nn.Module):
    """Maps raw `[N, 40, 40, 3]` bf16 pixel patches (-1 = padding) to decoder
    embeddings: a ViT with a CLS token per image, then a 2x2 merge projection to
    `text_hidden_size`. Runs replicated on every TP rank. Images are stills,
    so the `temporal_patch_size` frames of a patch are equal and their mean is
    the frame itself.
    """

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.text_hidden_size = text_hidden_size
        self.backbone = Inkling2VisionBackbone(config)
        self.proj_in = _linear(
            self.merge_size**2 * config["hidden_size"],
            config["projection_hidden_dim"],
            bias=config["projector_bias"],
        )
        self.proj_out = _linear(
            config["projection_hidden_dim"],
            text_hidden_size,
            bias=config["projector_bias"],
        )

    def metadata(self, grid: list[list[int]], device: torch.device) -> VisionMetadata:
        m, p, side = self.merge_size, self.patch_size, self.merge_size * self.patch_size
        frame = side * side * 3
        meta, patch_rows, offsets, pool_rows, lengths = [], [], [], [], []
        num_tokens = num_patches = num_pixels_rows = 0
        for _, h, w in grid:
            height, width = h * m, w * m
            y, x = np.divmod(np.arange(height * width, dtype=np.int64), width)
            image_meta = np.empty((height * width + 1, 4), np.int64)
            image_meta[0, :2] = -1
            image_meta[1:, 0], image_meta[1:, 1] = y, x
            image_meta[:, 2], image_meta[:, 3] = height, width
            meta.append(image_meta)
            rows = np.arange(height * width + 1, dtype=np.int64) + num_pixels_rows - 1
            rows[0] = 0
            patch_rows.append(rows)
            patch = num_patches + (y // m) * w + x // m
            offsets.append(patch * frame + (y % m) * (p * side * 3) + (x % m) * (p * 3))
            # [h, w, m, m] merge order of the rows after the CLS row.
            merged = (y // m) * (w * m * m) + (x // m) * (m * m) + (y % m) * m + x % m
            pool = np.empty(height * width, np.int64)
            pool[merged] = num_tokens + 1 + np.arange(height * width)
            pool_rows.append(pool)
            lengths.append(height * width + 1)
            num_tokens += height * width + 1
            num_patches += h * w
            num_pixels_rows += height * width
        cu_seqlens = np.cumsum([0, *lengths])
        parts = [
            np.concatenate(meta).ravel(),
            *map(np.concatenate, (patch_rows, offsets, pool_rows)),
            cu_seqlens,
        ]
        packed = np.concatenate(parts)
        assert packed.max() < 2**31
        packed_t = torch.from_numpy(packed.astype(np.int32)).pin_memory()
        packed_t = packed_t.to(device, non_blocking=True)
        sizes = [part.size for part in parts]
        meta_t, patch_rows_t, offsets_t, pool_t, cu_t = packed_t.split(sizes)
        # Same max_seqlen as the reference FA4 launch.
        max_seqlen = max(257, num_tokens - len(lengths) + 1)
        return VisionMetadata(
            meta_t.view(-1, 4), patch_rows_t, offsets_t, pool_t, cu_t, max_seqlen
        )

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        side = self.patch_size * self.merge_size
        assert pixels.dtype == torch.bfloat16
        assert pixels.shape[1:] == (side, side, 3)
        grid = grid_thw.tolist()
        assert all(t == 1 for t, _, _ in grid)
        assert sum(h * w for _, h, w in grid) == pixels.shape[0]

        return self.encode(pixels, self.metadata(grid, pixels.device))

    def encode(self, pixels: torch.Tensor, metadata: VisionMetadata) -> torch.Tensor:
        """Encodes packed patches with fixed-shape (CUDA-graphable) kernels."""
        backbone = self.backbone
        embeddings = backbone.embeddings
        patches = gather_patches(
            pixels.contiguous(),
            metadata.pixel_offsets,
            self.patch_size,
            _CLIP_MEAN,
            _PROCESSOR_CLIP_STD,
            _CLIP_STD,
        )
        g = self.posemb_grid_size
        hidden, normed = embed_tokens(
            embeddings.patch_embedding(patches),
            metadata.patch_rows,
            embeddings.position_embedding.view(g, g, -1),
            metadata.meta,
            embeddings.class_embedding,
            embeddings.cls_position_embedding,
            backbone.ln_pre,
            backbone.layers[0].ln_1,
        )
        layers = backbone.layers
        for i, layer in enumerate(layers):
            next_norm = layers[i + 1].ln_1 if i + 1 < len(layers) else None
            normed = layer(hidden, normed, metadata, next_norm)
        pooled = hidden.index_select(0, metadata.pool_rows)
        pooled = pooled.view(-1, self.merge_size**2 * hidden.shape[-1])
        return self.proj_out(F.gelu(self.proj_in(pooled)))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `model.visual.*` checkpoint tensors with that prefix stripped."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv_shards: Counter[str] = Counter()
        for name, weight in weights:
            for shard, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                if f".self_attn.{proj}." in name:
                    name = name.replace(proj, "qkv_proj")
                    param = params[name]
                    size = param.shape[0] // 3
                    default_weight_loader(
                        param.data[shard * size : (shard + 1) * size], weight
                    )
                    qkv_shards[name] += 1
                    break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded | {name for name, count in qkv_shards.items() if count == 3}
