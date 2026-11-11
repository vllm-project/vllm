# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in windowed FlashMLA with a standard FP8 cache and Triton fallback."""

import math

import torch

import vllm.envs as envs
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionLayer, max_decode_query_len
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
from vllm.v1.attention.ops.windowed_flashmla import (
    windowed_flash_mla,
    windowed_flash_mla_workspace,
)
from vllm.v1.worker.workspace import is_workspace_manager_initialized


class FlashMLASWABackend(TritonMLABackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SWA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["FlashMLASWAImpl"]:
        return FlashMLASWAImpl


class FlashMLASWAImpl(TritonMLAImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_window = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and is_flashmla_sparse_supported()[0]
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and self.num_heads in (64, 128)
            and self.dcp_world_size == 1
            and self.decode_sliding_window is not None
            and self.decode_sliding_window >= 2048
            and self.kv_cache_dtype in ("fp8", "fp8_e4m3")
        )

    def reserve_workspace(self) -> None:
        super().reserve_workspace()
        if not self._native_window or not is_workspace_manager_initialized():
            return
        query_len = max_decode_query_len(self._vllm_config)
        if query_len <= 1:
            return
        requests = self._vllm_config.scheduler_config.max_num_seqs
        assert self.decode_sliding_window is not None
        windowed_flash_mla_workspace(
            requests,
            requests * query_len,
            self.decode_sliding_window,
            query_len,
            torch.device(current_platform.device_type),
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The native V3.2 kernel converts FP32 scale exponents to UE8M0. A
        # non-power-of-two scale must keep the original Triton dequantization.
        scale = layer._k_scale_float
        native_scale = (
            math.isfinite(scale) and scale > 0 and math.frexp(scale)[0] == 0.5
        )
        assert attn_metadata.decode is not None
        query = q[0] if isinstance(q, tuple) else q
        requests = attn_metadata.decode.seq_lens.shape[0]
        query_len, remainder = divmod(query.shape[0], requests) if requests else (1, 0)
        dtype = q[0].dtype if isinstance(q, tuple) else q.dtype
        if (
            not self._native_window
            or envs.VLLM_BATCH_INVARIANT
            or not native_scale
            or dtype != torch.bfloat16
            or query_len <= 1
            or remainder != 0
        ):
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert self.decode_sliding_window is not None
        return windowed_flash_mla(
            q,
            kv_c_and_k_pe_cache,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_metadata.query_start_loc[: attn_metadata.num_decodes + 1],
            self.decode_sliding_window,
            query_len,
            layer._k_scale,
            self.scale,
            attn_metadata.causal,
        )
