# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed CuTe DSL backend for MLA prefill."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TokenspeedMLAPrefillBackend(MLAPrefillBackend):
    """TokenSpeed CuTe DSL backend for MLA prefill."""

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "TOKENSPEED_MLA"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    _INSTALL_HINT = (
        "tokenspeed_mla package is not installed. "
        "Install it with: `uv pip install tokenspeed-mla`"
    )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from tokenspeed_mla import (
                tokenspeed_mla_prefill,  # noqa: F401
            )

            return True
        except ImportError:
            return False

    @classmethod
    def validate_configuration(
        cls,
        device_capability,
        selector_config,
    ) -> list[str]:
        # Replace the generic "required dependencies not available" message
        # from the base class with a specific install hint so users know
        # exactly which package to install when they explicitly select this
        # backend without having tokenspeed_mla installed.
        reasons = super().validate_configuration(device_capability, selector_config)
        return [
            cls._INSTALL_HINT if r == "required dependencies not available" else r
            for r in reasons
        ]

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

        # Pre-JIT BF16 and FP8 prefill kernels. Idempotent — also called from
        # TokenspeedMLAImpl.__init__; second call is a no-op.
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=qk_nope_head_dim + qk_rope_head_dim,
                d_v=v_head_dim,
                enable_pdl=False,
            )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        # Kernel signature requires `seq_lens` but the implementation never reads
        # it (per-batch lengths are derived from `cum_seq_lens` diffs); compute
        # for parity with trtllm_ragged. cuda-graph padding in
        # `query_start_loc` is saturated to `total_num_tokens`
        # (gpu_model_runner.py:1905), so trailing diffs are 0 and padded batches
        # are kernel no-ops — same reason trtllm passes the padded length as
        # batch_size directly.
        self._query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from tokenspeed_mla import tokenspeed_mla_prefill

        # `v` arrives as the second half of `kv_nope.split(...)` in
        # mla_attention.forward_mha — a non-contiguous view of `kv_nope` along
        # dim=-1. The kernel does `v.reshape(1, total_kv, h_k, 1, d_v)` which
        # would silently copy on a non-contiguous tensor; force contiguity here
        # so the copy (if any) happens once outside the kernel call.
        v = v.contiguous()

        ret = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=self._query_seq_lens,
            cum_seq_lens=self._prefill_metadata.query_start_loc,
            max_seq_len=self._prefill_metadata.max_query_len,
            batch_size=self._query_seq_lens.shape[0],
            softmax_scale=self.scale,
            is_causal=True,
            return_lse=return_softmax_lse,
            enable_pdl=False,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tokenspeed_mla import tokenspeed_mla_prefill

        assert self._prefill_metadata.chunked_context is not None
        chunked = self._prefill_metadata.chunked_context

        # See note in run_prefill_new_tokens — `v` is a split-view of `kv_nope`
        # in `_compute_prefill_context` and arrives non-contiguous.
        v = v.contiguous()

        attn_out, lse = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=chunked.seq_lens[chunk_idx],
            cum_seq_lens=chunked.cu_seq_lens[chunk_idx],
            max_seq_len=chunked.max_seq_lens[chunk_idx],
            batch_size=chunked.seq_lens[chunk_idx].shape[0],
            softmax_scale=self.scale,
            is_causal=False,
            return_lse=True,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            max_seq_len_q=self._prefill_metadata.max_query_len,
            enable_pdl=False,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
