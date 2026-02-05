# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TRT-LLM Ragged backend for MLA prefill."""

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
    MLAPrefillBuilderState,
    MLAPrefillImpl,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec


class TrtllmRaggedPrefillBackend(MLAPrefillBackend):
    """TRT-LLM Ragged backend for MLA prefill.

    This backend is optimized for Blackwell (SM100) architecture and
    uses TRT-LLM's ragged attention kernel for DeepSeek models.
    """

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "TRTLLM_RAGGED_PREFILL"

    @staticmethod
    def get_prefill_impl_cls() -> type["TrtllmRaggedPrefillImpl"]:
        return TrtllmRaggedPrefillImpl

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # TRT-LLM ragged prefill is optimized for Blackwell
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

    @classmethod
    def create_builder_state(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        device: torch.device,
    ) -> MLAPrefillBuilderState:
        """Create TRT-LLM Ragged-specific builder state."""
        workspace_buffer = torch.empty(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )

        return MLAPrefillBuilderState(
            workspace_buffer=workspace_buffer,
        )

    @classmethod
    def post_process_prefill_metadata(
        cls,
        prefill_metadata: "MLACommonPrefillMetadata",
        builder_state: MLAPrefillBuilderState,
        prefill_query_start_loc: torch.Tensor,
    ) -> None:
        """Set TRT-LLM Ragged-specific fields on the prefill metadata."""
        prefill_metadata.query_seq_lens = (
            prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        )
        prefill_metadata.workspace_buffer = builder_state.workspace_buffer


class TrtllmRaggedPrefillImpl(MLAPrefillImpl):
    """TRT-LLM Ragged implementation for MLA prefill."""

    requires_v_padding: bool = False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
            device=device,
        )

    def run_prefill_new_tokens(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """TRT-LLM ragged attention for new tokens (causal)."""
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        assert prefill_metadata.query_seq_lens is not None
        assert prefill_metadata.workspace_buffer is not None

        ret = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=prefill_metadata.workspace_buffer,
            seq_lens=prefill_metadata.query_seq_lens,
            max_q_len=prefill_metadata.max_query_len,
            max_kv_len=prefill_metadata.max_query_len,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=prefill_metadata.query_seq_lens.shape[0],
            window_left=-1,
            cum_seq_lens_q=prefill_metadata.query_start_loc,
            cum_seq_lens_kv=prefill_metadata.query_start_loc,
            enable_pdl=False,
            is_causal=True,
            return_lse=return_softmax_lse,
        )

        if isinstance(ret, tuple):
            # Convert from (q_len, num_heads) to (num_heads, q_len)
            return ret[0], ret[1].transpose(0, 1).contiguous()
        return ret

    def run_prefill_context_chunk(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TRT-LLM ragged attention for context chunks (non-causal)."""
        from flashinfer.prefill import trtllm_ragged_attention_deepseek

        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.seq_lens[chunk_idx] is not None
        assert prefill_metadata.workspace_buffer is not None

        out = torch.zeros(
            q.shape[0],
            q.shape[1],
            v.shape[2],
            device=q.device,
            dtype=q.dtype,
        )
        prefill_metadata.workspace_buffer.fill_(0)

        attn_out, lse = trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=prefill_metadata.workspace_buffer,
            seq_lens=prefill_metadata.chunked_context.seq_lens[chunk_idx],
            max_q_len=prefill_metadata.max_query_len,
            max_kv_len=prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            o_sf_scale=1.0,
            batch_size=prefill_metadata.chunked_context.seq_lens[chunk_idx].shape[0],
            window_left=-1,
            cum_seq_lens_q=prefill_metadata.query_start_loc,
            cum_seq_lens_kv=prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            enable_pdl=False,
            is_causal=False,
            return_lse=True,
            out=out,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
