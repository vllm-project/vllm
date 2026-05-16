# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer backend for MLA prefill."""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.utils import (
    get_per_layer_parameters,
    infer_global_hyperparameters,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = object  # type: ignore[misc,assignment]

_DEFAULT_NUM_CHUNKS = 32


class FlashInferPrefillBackend(MLAPrefillBackend):
    """FlashInfer backend for MLA prefill."""

    requires_r1_mla_dimensions = True

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
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

        self._prefill_main: BatchPrefillWithRaggedKVCacheWrapper | None = None
        self._prefill_chunks: list[BatchPrefillWithRaggedKVCacheWrapper] = []
        if layer_names is None:
            raise ValueError(
                "FlashInferPrefillBackend requires layer_names to "
                "initialize global hyperparameters."
            )

        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonImpl,
        )

        self._global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl)  # type: ignore[type-abstract]
        )

    def _ensure_chunks(
        self,
        num_chunks: int,
        workspace_buffer: torch.Tensor,
    ) -> None:
        if len(self._prefill_chunks) < num_chunks:
            for _ in range(len(self._prefill_chunks), num_chunks):
                self._prefill_chunks.append(
                    BatchPrefillWithRaggedKVCacheWrapper(
                        workspace_buffer, "NHD", backend="cutlass"
                    )
                )

    def prepare_metadata(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        qo_indptr = prefill_metadata.query_start_loc
        has_context = prefill_metadata.chunked_context is not None
        (workspace_buffer,) = current_workspace_manager().get_simultaneous(
            ((envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,), torch.uint8),
        )

        if self._prefill_main is None:
            self._prefill_main = BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD", backend="cutlass"
            )
            self._ensure_chunks(_DEFAULT_NUM_CHUNKS, workspace_buffer)

        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            num_chunks = chunked_context.cu_seq_lens.shape[0]
            self._ensure_chunks(num_chunks, workspace_buffer)

        num_qo_heads = self.num_heads
        num_kv_heads = num_qo_heads

        head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        head_dim_vo = self.v_head_dim
        kv_indptr = qo_indptr.clone()

        assert self._prefill_main is not None
        self._prefill_main.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,
            sm_scale=self._global_hyperparameters.sm_scale,
            window_left=self._global_hyperparameters.window_left,
            logits_soft_cap=self._global_hyperparameters.logits_soft_cap,
            q_data_type=prefill_metadata.q_data_type,
            o_data_type=prefill_metadata.output_dtype,
        )

        if has_context:
            chunked_context = prefill_metadata.chunked_context
            assert chunked_context is not None
            for i in range(num_chunks):
                kv_indptr_chunk = chunked_context.cu_seq_lens[i]

                self._prefill_chunks[i].plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr_chunk,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    causal=False,
                    sm_scale=self._global_hyperparameters.sm_scale,
                    window_left=self._global_hyperparameters.window_left,
                    logits_soft_cap=self._global_hyperparameters.logits_soft_cap,
                    q_data_type=prefill_metadata.q_data_type,
                    o_data_type=prefill_metadata.output_dtype,
                )

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_main is not None

        ret = self._prefill_main.run(
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
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, lse = self._prefill_chunks[chunk_idx].run(
            q=q,
            k=k,
            v=v,
            return_lse=True,
        )

        # Convert from (q_len, num_heads) to (num_heads, q_len)
        return attn_out, lse.transpose(0, 1).contiguous()
