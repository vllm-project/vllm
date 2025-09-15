# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.model_executor.custom_op import CustomOp

from .common import rotate_gptj, rotate_neox


@CustomOp.register("dual_chunk_rotary_embedding")
class DualChunkRotaryEmbedding(CustomOp):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        chunk_size: int,
        local_size: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.chunk_size = chunk_size
        self.local_size = local_size
        self.dtype = dtype
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        (q_cache, qc_cache, k_cache, qc_no_clamp_cache,
         q_inter_cache) = self._compute_cos_sin_cache()

        self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)
        self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)
        self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)
        self.register_buffer("cos_sin_qc_no_clamp_cache",
                             qc_no_clamp_cache,
                             persistent=False)
        self.register_buffer("cos_sin_q_inter_cache",
                             q_inter_cache,
                             persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        chunk_len = self.chunk_size - self.local_size
        q_t = torch.arange(chunk_len, dtype=torch.float)
        qc_t = (torch.arange(chunk_len, dtype=torch.float) +
                chunk_len).clamp(max=self.chunk_size)
        k_t = torch.arange(self.max_position_embeddings,
                           dtype=torch.float) % chunk_len

        # count from chunk_len, no clamp(self.chunk_size) restriction
        qc_no_clamp_t = torch.arange(chunk_len, dtype=torch.float) + chunk_len
        # count from self.chunk_size for q_inter's rope
        q_inter_t = torch.arange(chunk_len,
                                 dtype=torch.float) + self.chunk_size

        q_freqs = torch.outer(q_t, inv_freq)
        qc_freqs = torch.outer(qc_t, inv_freq)
        k_freqs = torch.outer(k_t, inv_freq)
        qc_no_clamp_freqs = torch.outer(qc_no_clamp_t, inv_freq)
        q_inter_freqs = torch.outer(q_inter_t, inv_freq)

        q_cos = q_freqs.cos()
        q_sin = q_freqs.sin()
        qc_cos = qc_freqs.cos()
        qc_sin = qc_freqs.sin()
        k_cos = k_freqs.cos()
        k_sin = k_freqs.sin()

        qc_no_clamp_cos = qc_no_clamp_freqs.cos()
        qc_no_clamp_sin = qc_no_clamp_freqs.sin()
        q_inter_cos = q_inter_freqs.cos()
        q_inter_sin = q_inter_freqs.sin()

        q_cache = torch.cat((q_cos, q_sin), dim=-1).to(dtype=self.dtype,
                                                       device=self.device)
        qc_cache = torch.cat((qc_cos, qc_sin), dim=-1).to(dtype=self.dtype,
                                                          device=self.device)
        k_cache = torch.cat((k_cos, k_sin), dim=-1).to(dtype=self.dtype,
                                                       device=self.device)
        qc_no_clamp_cache = torch.cat((qc_no_clamp_cos, qc_no_clamp_sin),
                                      dim=-1).to(dtype=self.dtype,
                                                 device=self.device)
        q_inter_cache = torch.cat((q_inter_cos, q_inter_sin),
                                  dim=-1).to(dtype=self.dtype,
                                             device=self.device)
        return q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]
        else:
            query_pass = None
            key_pass = None

        positions_with_offsets = (torch.add(positions, offsets)
                                  if offsets is not None else positions)
        key = self._apply_rotary_embedding(
            self.cos_sin_k_cache[positions_with_offsets], key_rot, key_pass)
        chunk_len = self.chunk_size - self.local_size
        query = self._apply_rotary_embedding(
            self.cos_sin_q_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_succ = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_inter = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[chunk_len - 1].repeat(positions.shape[0], 1),
            query_rot, query_pass)
        query_succ_critical = self._apply_rotary_embedding(
            self.cos_sin_qc_no_clamp_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_inter_critical = self._apply_rotary_embedding(
            self.cos_sin_q_inter_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)

        # merge query into one tensor to simplify the interfaces
        query = torch.cat((
            query,
            query_succ,
            query_inter,
            query_succ_critical,
            query_inter_critical,
        ),
                          dim=-1)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(positions, query, key, offsets)

    def _apply_rotary_embedding(self, cos_sin, hidden_rot, hidden_pass):
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin

        if self.rotary_dim < self.head_size:
            hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)
        else:
            hidden = hidden_rot
        return hidden.flatten(-2).squeeze(0)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", chunk_size={self.chunk_size}, local_size={self.local_size}"
        return s
