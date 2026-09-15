# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA prefill backend implemented with dense per-request attention.

This backend is selected on CPU devices, where flash-attn is unavailable. It
computes MLA prefill attention by slicing the ragged (varlen) q/k/v tensors
per request and evaluating dense attention in PyTorch.
"""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


class CPUSDPAMLAPrefillBackend(MLAPrefillBackend):
    """MLA prefill backend for CPU using dense PyTorch attention.

    flash-attn is not installable on CPU, so this backend provides an
    equivalent prefill implementation using per-request dense attention.
    On AMD Zen CPUs with zentorch loaded, the inner attention kernel uses
    ``zentorch_sdpa``; otherwise it falls back to torch SDPA or a reference
    matmul/softmax path for correctness (including LSE for context chunks).
    """

    @staticmethod
    def get_name() -> str:
        return "CPU_SDPA_MLA"

    @classmethod
    def is_available(cls) -> bool:
        # SDPA is always available on CPU.
        return True

    def supports_out(self) -> bool:
        return True

    @staticmethod
    def _use_zentorch_sdpa() -> bool:
        from vllm.platforms import current_platform

        return (
            current_platform.is_zen_cpu()
            and hasattr(torch.ops, "zentorch")
            and hasattr(torch.ops.zentorch, "zentorch_sdpa")
        )

    @staticmethod
    def _sdpa_layout(
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Convert [NH, S, D] tensors to [1, NH, S, D] SDPA layout.

        Returns padded V and the original V head dim so callers can slice
        padding off after attention.
        """
        h_q = q_r.shape[-1]
        h_v = v_r.shape[-1]
        pad = h_q - h_v

        qs = q_r.unsqueeze(0).contiguous()
        ks = k_r.unsqueeze(0).contiguous()
        vs = v_r.unsqueeze(0).contiguous()
        if pad > 0:
            vs = torch.nn.functional.pad(vs, (0, pad))
        return qs, ks, vs, h_v

    def _zentorch_per_request_attn(
        self,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
        *,
        causal: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        qs, ks, vs, h_v = self._sdpa_layout(q_r, k_r, v_r)
        attn, _ = torch.ops.zentorch.zentorch_sdpa(
            qs,
            ks,
            vs,
            0.0,
            causal,
            attn_mask=None,
            scale=self.scale,
        )
        return attn[0].transpose(0, 1)[..., :h_v].to(out_dtype)

    def _torch_per_request_attn(
        self,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
        *,
        causal: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        qs, ks, vs, h_v = self._sdpa_layout(q_r, k_r, v_r)
        attn = torch.nn.functional.scaled_dot_product_attention(
            qs,
            ks,
            vs,
            is_causal=causal,
            scale=self.scale,
        )
        return attn[0].transpose(0, 1)[..., :h_v].to(out_dtype)

    def _reference_per_request_attn(
        self,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
        *,
        causal: bool,
        return_lse: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # fp32 reference path used by upstream tests and LSE fallbacks.
        q_r = q_r.float()
        k_r = k_r.float()
        v_r = v_r.float()
        attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale
        if causal:
            sq = q_r.shape[-2]
            sk = k_r.shape[-2]
            causal_mask = torch.ones(
                (sq, sk),
                dtype=torch.bool,
                device=attn_scores.device,
            ).triu(1)
            attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
        out_r = torch.matmul(attn_probs, v_r).transpose(0, 1).to(out_dtype)
        if not return_lse:
            return out_r
        return out_r, torch.logsumexp(attn_scores, dim=-1)

    def _per_request_attn(
        self,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
        *,
        causal: bool,
        return_lse: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # zentorch_sdpa's second return is not a usable LSE for prefix/suffix
        # merge, so keep the fp32 reference path whenever LSE is required.
        if self._use_zentorch_sdpa() and not return_lse:
            return self._zentorch_per_request_attn(
                q_r,
                k_r,
                v_r,
                causal=causal,
                out_dtype=out_dtype,
            )
        if not return_lse:
            return self._torch_per_request_attn(
                q_r,
                k_r,
                v_r,
                causal=causal,
                out_dtype=out_dtype,
            )
        return self._reference_per_request_attn(
            q_r,
            k_r,
            v_r,
            causal=causal,
            return_lse=True,
            out_dtype=out_dtype,
        )

    def _ragged_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_cu_seq_lens: torch.Tensor,
        kv_cu_seq_lens: torch.Tensor,
        causal: bool,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # q/k/v: [num_tokens, num_heads, head_dim] (ragged, concatenated).
        # q_cu_seq_lens/kv_cu_seq_lens: [num_reqs + 1] row offsets for each
        # request's query and key/value tokens.
        num_reqs = q_cu_seq_lens.numel() - 1
        assert num_reqs == kv_cu_seq_lens.numel() - 1

        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for r in range(num_reqs):
            q_start = int(q_cu_seq_lens[r])
            q_end = int(q_cu_seq_lens[r + 1])
            kv_start = int(kv_cu_seq_lens[r])
            kv_end = int(kv_cu_seq_lens[r + 1])

            q_r = q[q_start:q_end].transpose(0, 1)
            k_r = k[kv_start:kv_end].transpose(0, 1)
            v_r = v[kv_start:kv_end].transpose(0, 1)

            if return_softmax_lse:
                out_r, lse_r = self._per_request_attn(
                    q_r,
                    k_r,
                    v_r,
                    causal=causal,
                    return_lse=True,
                    out_dtype=v.dtype,
                )
                assert isinstance(out_r, torch.Tensor)
                lse_chunks.append(lse_r)
            else:
                out_r = self._per_request_attn(
                    q_r,
                    k_r,
                    v_r,
                    causal=causal,
                    return_lse=False,
                    out_dtype=v.dtype,
                )
                assert isinstance(out_r, torch.Tensor)

            out_chunks.append(out_r)

        out = torch.cat(out_chunks, dim=0)
        if not return_softmax_lse:
            return out
        lse = torch.cat(lse_chunks, dim=1)
        return out, lse

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert output_scale is None, (
            "CPU MLA prefill backend does not support fused FP8 output"
        )
        out_t = self._ragged_sdpa(
            q,
            k,
            v,
            self._prefill_metadata.query_start_loc,
            self._prefill_metadata.query_start_loc,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )
        if out is not None and not return_softmax_lse:
            assert isinstance(out_t, torch.Tensor)
            out.copy_(out_t)
            return out
        return out_t

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out is None, "CPU MLA context prefill does not support in-place output"
        result = self._ragged_sdpa(
            q,
            k,
            v,
            chunk.query_start_loc,
            chunk.cu_seq_lens,
            causal=False,
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        out, lse = result
        return out, lse
