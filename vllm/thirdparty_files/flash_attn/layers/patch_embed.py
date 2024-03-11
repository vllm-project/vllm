# We use the same API as https://github.com/rwightman/pytorch-image-models/blob/v0.6.11/timm/models/layers/patch_embed.py
# But we use nn.Linear instead of Conv2d and it's about 8x faster.

from functools import partial

import torch.nn as nn
from einops import rearrange
from torch import _assert
from torch.nn.modules.utils import _pair

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        fused_bias_fc=False,
    ):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")

        linear_cls = nn.Linear if not fused_bias_fc or not bias else FusedDense
        self.proj = linear_cls(in_chans * patch_size[0] * patch_size[1], embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(
            rearrange(
                x,
                "b c (h p1) (w p2) -> b h w (c p1 p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
        )
        if self.flatten:
            x = rearrange(x, "b h w c -> b (h w) c")
        x = self.norm(x)
        return x
