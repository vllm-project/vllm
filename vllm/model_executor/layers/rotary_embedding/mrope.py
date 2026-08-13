# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .base import RotaryEmbeddingBase
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding, yarn_get_mscale


@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, rotary_dim // 2)
    # instead of (3, bsz, seq_len, head_dim), also supports interleaved rotary
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, rotary_dim // 2)

    # Updated stride calculation for half rotary_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half rotary_dim
    cos_offsets = tl.arange(0, pad_rd // 2)
    if is_interleaved:
        valid_mask = cos_offsets < half_rd
        h_mask = (
            valid_mask & ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        )
        w_mask = (
            valid_mask & ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        )
        t_mask = valid_mask & ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the two values in each rotary pair for the current token.
    # NeoX pairs the first and second halves, while GPT-J pairs
    # adjacent values.
    # ####################################################################
    if is_neox_style:
        rotary_offsets = tl.arange(0, pad_rd // 2)
        first_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + rotary_offsets[None, :]
        first_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + rotary_offsets[None, :]
        first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            rotary_offsets[None, :] < rd // 2
        )
        first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            rotary_offsets[None, :] < rd // 2
        )

        q_tile_1 = tl.load(q_ptr + first_q_offsets, mask=first_q_mask, other=0).to(
            sin_row.dtype
        )
        k_tile_1 = tl.load(k_ptr + first_k_offsets, mask=first_k_mask, other=0).to(
            sin_row.dtype
        )

        second_q_offsets = first_q_offsets + (rd // 2)
        second_k_offsets = first_k_offsets + (rd // 2)
        q_tile_2 = tl.load(q_ptr + second_q_offsets, mask=first_q_mask, other=0).to(
            sin_row.dtype
        )
        k_tile_2 = tl.load(k_ptr + second_k_offsets, mask=first_k_mask, other=0).to(
            sin_row.dtype
        )

        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_q_offsets, new_q_tile_2, mask=first_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_k_offsets, new_k_tile_2, mask=first_k_mask)
    else:
        # Load and store adjacent rotary pairs contiguously. Using stride-two
        # even/odd offsets makes Triton emit scalar 16-bit memory operations on
        # AMD, while split/interleave only rearranges values in registers.
        rotary_offsets = tl.arange(0, pad_rd)
        q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + rotary_offsets[None, :]
        k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + rotary_offsets[None, :]
        q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            rotary_offsets[None, :] < rd
        )
        k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            rotary_offsets[None, :] < rd
        )

        q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0).to(sin_row.dtype)
        k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_row.dtype)
        q_tile_1, q_tile_2 = tl.split(tl.reshape(q_tile, (pad_n_qh, pad_rd // 2, 2)))
        k_tile_1, k_tile_2 = tl.split(tl.reshape(k_tile, (pad_n_kh, pad_rd // 2, 2)))

        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        new_q_tile = tl.interleave(new_q_tile_1, new_q_tile_2)
        tl.store(q_ptr + q_offsets, new_q_tile, mask=q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        new_k_tile = tl.interleave(new_k_tile_1, new_k_tile_2)
        tl.store(k_ptr + k_offsets, new_k_tile, mask=k_mask)


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2VL mrope kernel.

    Args:
        q: [num_tokens, num_heads * head_size]
        k: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, rotary_dim // 2]
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, rotary_dim // 2]
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
        is_neox_style: Whether rotary pairs use split-half (NeoX) or
            adjacent (GPT-J) layout.

    """
    n_row, n_q_head_head_dim = q.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = k.shape[1] // head_size
    pad_rd = triton.next_power_of_2(rotary_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # ensure tensors passed into the kernel are contiguous.
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Small adjacent-pair tiles perform best with one wave per program on
    # ROCm. Keep the existing launch shape for larger rotary dimensions,
    # NeoX, and other backends.
    use_single_wave = current_platform.is_rocm() and not is_neox_style and pad_rd <= 64
    num_warps = 1 if use_single_wave else 4
    _triton_mrope_forward[(n_row,)](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_rd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        is_neox_style,
        num_warps=num_warps,
    )
    return q, k


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


class MRotaryEmbedding(RotaryEmbeddingBase):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        # YaRN parameters.
        *,
        scaling_factor: float | None = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        truncate: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        if self.scaling_factor is not None:
            # Get n-d magnitude scaling corrected for interpolation
            self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        else:
            self.mscale = 1.0

        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_inv_freq(base)
        return YaRNScalingRotaryEmbedding._compute_inv_freq(self, base)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_cos_sin_cache()
        return YaRNScalingRotaryEmbedding._compute_cos_sin_cache(self)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]

        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            if self.mrope_interleaved:
                cos = apply_interleaved_rope(cos, self.mrope_section)
                sin = apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(
            query_rot,
            cos,
            sin,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb.forward_native(
            key_rot,
            cos,
            sin,
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            assert self.mrope_section

            q, k = triton_mrope(
                query,
                key,
                cos,
                sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.mrope_interleaved,
                self.is_neox_style,
            )

            return q.reshape(query_shape), k.reshape(key_shape)

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb(
            query_rot,
            cos,
            sin,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb(
            key_rot,
            cos,
            sin,
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(
                    context_len + mrope_position_delta, seq_len + mrope_position_delta
                )
            )
            for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ):
        values = np.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values
