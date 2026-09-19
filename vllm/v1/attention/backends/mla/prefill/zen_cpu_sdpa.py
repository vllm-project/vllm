# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA prefill backend using zentorch SDPA on AMD Zen CPUs.

Subclasses the generic CPU SDPA backend and swaps only the inner attention
kernel for ``zentorch_sdpa``, which fuses the scale, causal mask, softmax and
both matmuls of the fp32 reference path into one call.

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
        """Convert [NH, S, D] tensors to the [1, NH, S, D] SDPA layout.

        MLA has ``v_head_dim < qk_head_dim``, which the fused kernel does not
        accept, so V is padded up to the query head dim. The original V head
        dim is returned so the caller can slice the padding off afterwards.
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
        # merge, so defer to the base backend whenever LSE is required.
        if return_lse:
            return super()._per_request_attn(
                q_r,
                k_r,
                v_r,
                causal=causal,
                return_lse=True,
                out_dtype=out_dtype,
            )
        return self._zentorch_per_request_attn(
            q_r,
            k_r,
            v_r,
            causal=causal,
            out_dtype=out_dtype,
        )
