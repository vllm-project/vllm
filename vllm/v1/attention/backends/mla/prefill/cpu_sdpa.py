# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA prefill backend implemented on top of torch SDPA.

This backend is selected on CPU devices, where flash-attn is unavailable. It
computes MLA prefill attention by converting the ragged (varlen) q/k/v tensors
into per-request padded-dense tensors and invoking
``torch.nn.functional.scaled_dot_product_attention``.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


class CPUSDPAMLAPrefillBackend(MLAPrefillBackend):
    """MLA prefill backend for CPU using torch SDPA.

    flash-attn is not installable on CPU, so this backend provides an
    equivalent prefill implementation using dense SDPA with padding. Each
    request is materialized as a padded-dense tensor and attended in
    isolation, then the relevant rows are scattered back into the ragged
    layout.
    """

    @staticmethod
    def get_name() -> str:
        return "CPU_SDPA_MLA"

    @classmethod
    def is_available(cls) -> bool:
        # SDPA is always available on CPU.
        return True

    def _dense_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        # q/k/v: [num_tokens, num_heads, head_dim] (ragged, concatenated).
        # cu_seq_lens: [num_reqs + 1] row offsets shared by q and k.
        num_reqs = cu_seq_lens.numel() - 1

        out_chunks: list[torch.Tensor] = []
        for r in range(num_reqs):
            start, end = int(cu_seq_lens[r]), int(cu_seq_lens[r + 1])
            q_r = q[start:end].transpose(0, 1)  # [NH, Sq, H]
            k_r = k[start:end].transpose(0, 1)  # [NH, Sk, H]
            v_r = v[start:end].transpose(0, 1)  # [NH, Sk, H]

            out_r = F.scaled_dot_product_attention(
                q_r.unsqueeze(0),
                k_r.unsqueeze(0),
                v_r.unsqueeze(0),
                is_causal=causal,
                scale=self.scale,
            )
            out_chunks.append(out_r.squeeze(0).transpose(0, 1))  # [Sq, NH, H]

        return torch.cat(out_chunks, dim=0)

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not return_softmax_lse, (
            "chunked-context path is not supported on CPU MLA prefill"
        )
        out_t = self._dense_sdpa(
            q, k, v, self._prefill_metadata.query_start_loc, causal=True
        )
        if out is not None:
            out.copy_(out_t)
            return out
        return out_t

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "chunked-context prefill is not supported on CPU MLA prefill"
        )
