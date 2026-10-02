# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA prefill backend using zentorch SDPA on AMD Zen CPUs.

Subclasses the generic CPU SDPA backend and replaces the per-request attention
with ``zentorch_sdpa``, which fuses the scale, causal mask, softmax and both
matmuls of the base implementation into a single call.

``zentorch_sdpa`` does not return a usable log-sum-exp, so requests that need
one (context chunks, prefix/suffix merge) are delegated to the base backend.

Availability is reported through :meth:`is_available`, so the prefill selector
gates this backend once at selection time rather than per attention call.
"""

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import CPUSDPAMLAPrefillBackend


class ZenCPUSDPAMLAPrefillBackend(CPUSDPAMLAPrefillBackend):
    """MLA prefill backend for AMD Zen CPUs backed by ``zentorch_sdpa``."""

    @staticmethod
    def get_name() -> str:
        return "ZEN_CPU_SDPA_MLA"

    @classmethod
    def is_available(cls) -> bool:
        return has_zentorch_op(["zentorch_sdpa"])

    @staticmethod
    def _sdpa_layout(
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Convert one request's [S, NH, D] tensors to the [1, NH, S, D] layout.

        MLA has ``v_head_dim < qk_head_dim``, which the fused kernel does not
        accept, so V is padded up to the query head dim. The original V head
        dim is returned so the caller can slice the padding off afterwards.
        """
        h_q = q_r.shape[-1]
        h_v = v_r.shape[-1]
        pad = h_q - h_v

        qs = q_r.transpose(0, 1).unsqueeze(0).contiguous()
        ks = k_r.transpose(0, 1).unsqueeze(0).contiguous()
        vs = v_r.transpose(0, 1).unsqueeze(0).contiguous()
        if pad > 0:
            vs = torch.nn.functional.pad(vs, (0, pad))
        return qs, ks, vs, h_v

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
        if return_softmax_lse:
            # zentorch_sdpa's second return is not a usable LSE, so fall back
            # to the base fp32 implementation whenever one is required.
            return super()._ragged_sdpa(
                q,
                k,
                v,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                causal,
                return_softmax_lse,
            )

        num_reqs = q_cu_seq_lens.numel() - 1
        assert num_reqs == kv_cu_seq_lens.numel() - 1

        out_chunks: list[torch.Tensor] = []
        for r in range(num_reqs):
            q_start = int(q_cu_seq_lens[r])
            q_end = int(q_cu_seq_lens[r + 1])
            kv_start = int(kv_cu_seq_lens[r])
            kv_end = int(kv_cu_seq_lens[r + 1])

            qs, ks, vs, h_v = self._sdpa_layout(
                q[q_start:q_end],
                k[kv_start:kv_end],
                v[kv_start:kv_end],
            )

            attn, _ = torch.ops.zentorch.zentorch_sdpa(
                qs,
                ks,
                vs,
                0.0,
                causal,
                attn_mask=None,
                scale=self.scale,
            )
            out_chunks.append(attn[0].transpose(0, 1)[..., :h_v].to(v.dtype))

        return torch.cat(out_chunks, dim=0)
