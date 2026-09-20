# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM Ragged backend for MLA prefill."""

from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)
from vllm.v1.attention.backends.utils import log2_lse_to_ln
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability


class TrtllmRaggedPrefillBackend(MLAPrefillBackend):
    """TRT-LLM Ragged backend for MLA prefill."""

    supported_mla_dimensions: ClassVar[list[MLADimensions]] = [
        MLADimensions(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        ),
    ]

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_RAGGED"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.prefill import (
                trtllm_ragged_attention_deepseek,  # noqa: F401
            )

            return True
        except ImportError:
            return False

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
        (self._workspace_buffer,) = current_workspace_manager().get_simultaneous(
            (
                (envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,),
                torch.uint8,
            ),
        )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        super().prepare_metadata(prefill_metadata)
        self._query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )
        query_lens_cpu = prefill_metadata.query_lens_cpu
        if query_lens_cpu is None:
            raise ValueError("TRTLLM ragged prefill requires CPU query lengths")
        if query_lens_cpu.device.type != "cpu":
            raise ValueError("TRTLLM ragged prefill query lengths must be on CPU")
        if query_lens_cpu.ndim != 1 or query_lens_cpu.numel() == 0:
            raise ValueError(
                "TRTLLM ragged prefill requires a non-empty 1D query-length tensor"
            )
        min_query_len = int(torch.min(query_lens_cpu).item())
        if min_query_len < 0:
            raise ValueError("TRTLLM ragged prefill query lengths must be non-negative")
        self._has_active_rows = min_query_len > 0
        if not self._has_active_rows:
            has_mixed_rows = bool(torch.any(query_lens_cpu > 0).item())
            if has_mixed_rows:
                raise ValueError(
                    "TRTLLM ragged prefill contains mixed active and empty query "
                    f"rows: query_lens={query_lens_cpu.tolist()}"
                )

    def supports_out(self) -> bool:
        # Output head dim is v.shape[-1] == v_head_dim, so `out` is unpadded.
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        if out is None:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self._prefill_metadata.output_dtype,
            )

        if not self._has_active_rows:
            out.zero_()
            if return_softmax_lse:
                lse = torch.empty(
                    self.num_heads,
                    q.shape[0],
                    dtype=torch.float32,
                    device=q.device,
                )
                return out, lse.fill_(-float("inf"))
            return out

        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=self._query_seq_lens,
            max_q_len=self._prefill_metadata.max_query_len,
            max_kv_len=self._prefill_metadata.max_query_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=self._query_seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=self._prefill_metadata.query_start_loc,
            cum_seq_lens_kv=self._prefill_metadata.query_start_loc,
            enable_pdl=False,
            is_causal=True,
            return_lse=return_softmax_lse,
            out=out,
            skip_all_rows_active_check=True,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], log2_lse_to_ln(ret[1].transpose(0, 1))
        return ret

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        if not chunk.all_rows_active:
            raise ValueError(
                "TRTLLM ragged context prefill contains an empty query or KV row"
            )

        if out is None:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                v.shape[2],
                device=q.device,
                dtype=self._prefill_metadata.output_dtype,
            )

        attn_out, lse = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=self._workspace_buffer,
            seq_lens=chunk.seq_lens,
            max_q_len=chunk.max_query_len,
            max_kv_len=chunk.max_seq_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=chunk.num_requests,
            window_left=-1,
            cum_seq_lens_q=chunk.query_start_loc,
            cum_seq_lens_kv=chunk.cu_seq_lens,
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=out,
            skip_all_rows_active_check=True,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, log2_lse_to_ln(lse.transpose(0, 1))
