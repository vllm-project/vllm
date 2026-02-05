# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.custom_op import PluggableLayer


# --8<-- [start:rel_pos_attention]
@PluggableLayer.register("rel_pos_attention")
class RelPosAttention(PluggableLayer):
    """Multi-head Attention block with relative position embeddings."""

    # --8<-- [end:rel_pos_attention]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """  # noqa: E501
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def _get_rel_pos(
        self,
        q_size: int,
        k_size: int,
        rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            dtype = rel_pos.dtype
            rel_pos = rel_pos.to(torch.float32)
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            ).to(dtype)
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(
            k_size / q_size, 1.0
        )
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(
            q_size / k_size, 1.0
        )
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )

        return rel_pos_resized[relative_coords.long()]

    def _add_decomposed_rel_pos(
        self,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
        Args:
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
            rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """  # noqa: E501
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self._get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self._get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        rel_h = rel_h.unsqueeze(-1)
        rel_w = rel_w.unsqueeze(-2)
        rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
        rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

        return rel_h, rel_w

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = self._add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            assert rel_h is not None and rel_w is not None
            rel_h = rel_h.view(
                B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(
                B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias
            )
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)

        return x
