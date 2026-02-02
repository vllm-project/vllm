# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer backend for MLA prefill."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object  # type: ignore[misc,assignment]


# Import base class for metadata - runtime import to avoid circular dependency
def _get_base_metadata_cls():
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )

    return MLACommonPrefillMetadata


@dataclass
class FlashInferPrefillMetadata(_get_base_metadata_cls()):  # type: ignore[misc]
    """FlashInfer-specific prefill metadata."""

    prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
    prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = field(
        default_factory=list
    )


class FlashInferPrefillBackend(MLAPrefillBackend):
    """FlashInfer backend for MLA prefill.

    This backend is optimized for Blackwell (SM100) architecture.
    """

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_PREFILL"

    @staticmethod
    def get_prefill_impl_cls() -> type["FlashInferPrefillImpl"]:
        return FlashInferPrefillImpl

    @staticmethod
    def get_prefill_metadata_cls() -> type["FlashInferPrefillMetadata"]:
        return FlashInferPrefillMetadata

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # FlashInfer prefill is optimized for Blackwell
        return device_capability.major == 10

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flashinfer import (
                BatchPrefillWithRaggedKVCacheWrapper,  # noqa: F401
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
        """Create FlashInfer-specific builder state."""
        # Import here to avoid circular dependency
        from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl
        from vllm.v1.attention.backends.utils import (
            get_per_layer_parameters,
            infer_global_hyperparameters,
        )

        workspace_buffer = torch.empty(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )

        global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl)  # type: ignore[type-abstract]
        )

        return MLAPrefillBuilderState(
            workspace_buffer=workspace_buffer,
            backend_state={
                "prefill_main": None,
                "prefill_chunks": [],
                "global_hyperparameters": global_hyperparameters,
            },
        )

    @classmethod
    def finalize_attention_metadata(
        cls,
        attn_metadata: Any,
        builder_state: MLAPrefillBuilderState,
        num_prefills: int,
        num_heads: int,
        kv_cache_spec: "AttentionSpec",
        mla_dims: Any,
        model_config: Any,
    ) -> None:
        """Build FlashInfer prefill wrappers."""
        if num_prefills == 0:
            return

        prefill = attn_metadata.prefill
        if prefill is None:
            return

        assert isinstance(prefill, FlashInferPrefillMetadata)

        qo_indptr = prefill.query_start_loc
        has_context = prefill.chunked_context is not None

        prefill_main = builder_state.backend_state.get("prefill_main")
        prefill_chunks = builder_state.backend_state.get("prefill_chunks", [])
        workspace_buffer = builder_state.workspace_buffer
        global_hyperparameters = builder_state.backend_state["global_hyperparameters"]

        if prefill_main is None:
            prefill_main = BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD", backend="cutlass"
            )
            builder_state.backend_state["prefill_main"] = prefill_main

        if has_context:
            chunked_context = prefill.chunked_context
            num_chunks = chunked_context.cu_seq_lens.shape[0]
            if len(prefill_chunks) < num_chunks:
                for _ in range(len(prefill_chunks), num_chunks):
                    prefill_chunks.append(
                        BatchPrefillWithRaggedKVCacheWrapper(
                            workspace_buffer, "NHD", backend="cutlass"
                        )
                    )
                builder_state.backend_state["prefill_chunks"] = prefill_chunks
            assert num_chunks <= len(prefill_chunks)

        num_qo_heads = num_heads
        num_kv_heads = num_qo_heads
        assert kv_cache_spec.num_kv_heads == 1

        head_dim_qk = mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim
        head_dim_vo = mla_dims.v_head_dim
        kv_indptr = qo_indptr.clone()

        prefill_main.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,
            sm_scale=global_hyperparameters.sm_scale,
            window_left=global_hyperparameters.window_left,
            logits_soft_cap=global_hyperparameters.logits_soft_cap,
            q_data_type=model_config.dtype,
        )

        if has_context:
            for i in range(num_chunks):
                kv_indptr_chunk = chunked_context.cu_seq_lens[i]

                prefill_chunks[i].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr_chunk,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    causal=False,
                    sm_scale=global_hyperparameters.sm_scale,
                    window_left=global_hyperparameters.window_left,
                    logits_soft_cap=global_hyperparameters.logits_soft_cap,
                    q_data_type=model_config.dtype,
                )

        prefill.prefill_main = prefill_main
        prefill.prefill_chunks = prefill_chunks


class FlashInferPrefillImpl(MLAPrefillImpl):
    """FlashInfer implementation for MLA prefill."""

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
        assert isinstance(prefill_metadata, FlashInferPrefillMetadata)
        assert prefill_metadata.prefill_main is not None

        ret = prefill_metadata.prefill_main.run(
            q=q,
            k=k,
            v=v,
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
        assert isinstance(prefill_metadata, FlashInferPrefillMetadata)

        attn_out, lse = prefill_metadata.prefill_chunks[chunk_idx].run(
            q=q,
            k=k,
            v=v,
            return_lse=True,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
