# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch fallback backend for MLA prefill.

This backend is intended as a last-resort fallback when no optimized MLA
prefill kernel (FlashAttention 3/4, FlashInfer, etc.) is available for the
current device.  It materializes the full attention score matrix, so memory
and compute scale with ``O(num_heads * q_len * kv_len)``.  Keep prompts
reasonably short on memory-constrained hardware, or disable chunked prefill
if context lengths grow large.
"""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TorchMHAPrefillBackend(MLAPrefillBackend):
    """PyTorch manual-attention fallback for MLA prefill.

    Supports DeepSeek V2/V3/R1-style MLA dimensions, including different
    query/key and value head dimensions.  All math is performed in FP32 for
    numerical stability and then cast back to the input dtype.
    """

    @staticmethod
    def get_name() -> str:
        return "TORCH_MHA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # Pure PyTorch; available everywhere CUDA/Python is.
        return True

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        qo_indptr = prefill_metadata.query_start_loc
        self._query_seq_lens = qo_indptr[1:] - qo_indptr[:-1]

    def _split_by_seq_lens(
        self,
        tensor: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Split a ragged tensor into per-sequence chunks."""
        seq_lens = (cu_seq_lens[1:] - cu_seq_lens[:-1]).cpu().tolist()
        chunks: list[torch.Tensor] = []
        offset = 0
        for length in seq_lens:
            chunks.append(tensor[offset : offset + length])
            offset += length
        return chunks

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Manual multi-head attention for a single sequence.

        Args:
            q: ``(q_len, num_heads, qk_head_dim)``.
            k: ``(kv_len, num_heads, qk_head_dim)``.
            v: ``(kv_len, num_heads, v_head_dim)``.
            causal: Whether to apply a lower-triangular causal mask.

        Returns:
            ``(output, lse)`` where ``output`` has shape
            ``(q_len, num_heads, v_head_dim)`` and ``lse`` has shape
            ``(num_heads, q_len)``.
        """
        q_len = q.shape[0]
        kv_len = k.shape[0]
        num_heads = q.shape[1]

        # Cast to FP32 for the numerically sensitive parts.
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()

        # scores: (num_heads, q_len, kv_len)
        scores = torch.einsum("qhd,khd->hqk", q_f, k_f) * self.scale

        if causal:
            # Build a causal mask: positions where k_index > q_index are masked.
            q_idx = torch.arange(q_len, device=scores.device)
            k_idx = torch.arange(kv_len, device=scores.device)
            mask = (
                (k_idx[None, :] > q_idx[:, None]).unsqueeze(0).expand(num_heads, -1, -1)
            )
            scores = scores.masked_fill(mask, float("-inf"))

        lse = torch.logsumexp(scores, dim=-1)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn_weights, v_f)
        return out.to(q.dtype), lse.to(q.dtype)

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_cu_seq_lens: torch.Tensor,
        kv_cu_seq_lens: torch.Tensor,
        causal: bool,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Ragged prefill or context-chunk attention across a batch."""
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        q_chunks = self._split_by_seq_lens(q, q_cu_seq_lens)
        k_chunks = self._split_by_seq_lens(k, kv_cu_seq_lens)
        v_chunks = self._split_by_seq_lens(v, kv_cu_seq_lens)

        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for q_b, k_b, v_b in zip(q_chunks, k_chunks, v_chunks):
            out_b, lse_b = self._attention(q_b, k_b, v_b, causal=causal)
            out_chunks.append(out_b)
            lse_chunks.append(lse_b)

        output = torch.cat(out_chunks, dim=0)
        # Keep the output 3-D (q_len, num_heads, v_head_dim).  The caller
        # (mla_attention.forward_mha) expects this shape and flattens the last
        # two dimensions itself, or passes the 3-D tensor to merge_attn_states.
        if return_softmax_lse:
            lse = torch.cat(lse_chunks, dim=1)
            return output, lse
        return output

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        qo_indptr = self._prefill_metadata.query_start_loc
        return self._run_prefill(
            q=q,
            k=k,
            v=v,
            q_cu_seq_lens=qo_indptr,
            kv_cu_seq_lens=qo_indptr,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunked = self._prefill_metadata.chunked_context
        assert chunked is not None
        qo_indptr = self._prefill_metadata.query_start_loc
        kv_cu_seq_lens = chunked.cu_seq_lens[chunk_idx]
        return self._run_prefill(
            q=q,
            k=k,
            v=v,
            q_cu_seq_lens=qo_indptr,
            kv_cu_seq_lens=kv_cu_seq_lens,
            causal=False,
            return_softmax_lse=True,
        )
