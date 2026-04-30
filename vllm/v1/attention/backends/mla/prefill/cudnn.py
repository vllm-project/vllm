# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuDNN backend for MLA prefill."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
    MLAPrefillImpl,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

CUDNN_WORKSPACE_SIZE = 12800


# Import base class for metadata - runtime import to avoid circular dependency
def _get_base_metadata_cls():
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )

    return MLACommonPrefillMetadata


@dataclass
class CudnnPrefillMetadata(_get_base_metadata_cls()):  # type: ignore[misc]
    """cuDNN-specific prefill metadata."""

    class ChunkedContextMetadata(
        _get_base_metadata_cls().ChunkedContextMetadata  # type: ignore[misc]
    ):
        seq_lens: torch.Tensor

    cudnn_workspace: torch.Tensor | None = None


class CudnnPrefillBackend(MLAPrefillBackend):
    """cuDNN backend for MLA prefill.

    This backend is optimized for Blackwell (SM100) architecture and
    requires NVIDIA artifactory access.
    """

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "CUDNN_PREFILL"

    @staticmethod
    def get_prefill_impl_cls() -> type["CudnnPrefillImpl"]:
        return CudnnPrefillImpl

    @staticmethod
    def get_chunked_context_metadata_cls() -> type:
        return CudnnPrefillMetadata.ChunkedContextMetadata

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer.prefill import (
                cudnn_batch_prefill_with_kv_cache,  # noqa: F401
            )
        except ImportError:
            return False

        from vllm.utils.flashinfer import has_nvidia_artifactory

        return has_nvidia_artifactory()


class CudnnPrefillImpl(MLAPrefillImpl):
    """cuDNN implementation for MLA prefill."""

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
        layer_names: list[str] | None = None,
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
            layer_names=layer_names,
        )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        assert isinstance(prefill_metadata, CudnnPrefillMetadata)
        prefill_metadata.query_seq_lens = (
            prefill_metadata.query_start_loc[1:] - prefill_metadata.query_start_loc[:-1]
        )
        num_seqs = prefill_metadata.query_seq_lens.shape[0]
        (prefill_metadata.cudnn_workspace,) = (
            current_workspace_manager().get_simultaneous(
                ((CUDNN_WORKSPACE_SIZE * num_seqs,), torch.int8),
            )
        )

    def run_prefill_new_tokens(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

        assert isinstance(prefill_metadata, CudnnPrefillMetadata)
        assert prefill_metadata.query_seq_lens is not None

        output, lse = cudnn_batch_prefill_with_kv_cache(
            q=q,
            k_cache=k,
            v_cache=v,
            scale=self.scale,
            workspace_buffer=prefill_metadata.cudnn_workspace,
            max_token_per_sequence=prefill_metadata.max_query_len,
            max_sequence_kv=prefill_metadata.max_query_len,
            actual_seq_lens_q=prefill_metadata.query_seq_lens.view(-1, 1, 1, 1),
            actual_seq_lens_kv=prefill_metadata.query_seq_lens.view(-1, 1, 1, 1),
            causal=True,
            # Do not support False for now
            return_lse=True,
            # Indicates actual_seq_lens are on GPU or CPU.
            is_cuda_graph_compatible=True,
        )

        if return_softmax_lse:
            return output, lse
        return output

    def run_prefill_context_chunk(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

        assert isinstance(prefill_metadata, CudnnPrefillMetadata)
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.seq_lens[chunk_idx] is not None
        assert prefill_metadata.query_seq_lens is not None

        return cudnn_batch_prefill_with_kv_cache(
            q=q,
            k_cache=k,
            v_cache=v,
            scale=self.scale,
            workspace_buffer=prefill_metadata.cudnn_workspace,
            max_token_per_sequence=prefill_metadata.max_query_len,
            max_sequence_kv=prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            actual_seq_lens_q=prefill_metadata.query_seq_lens.view(-1, 1, 1, 1),
            actual_seq_lens_kv=prefill_metadata.chunked_context.seq_lens[
                chunk_idx
            ].view(-1, 1, 1, 1),
            causal=False,
            return_lse=True,
            # Indicates actual_seq_lens are on GPU or CPU.
            is_cuda_graph_compatible=True,
        )
