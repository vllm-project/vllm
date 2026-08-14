# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

# V3.2 indexers don't expose `n_head`; GLM-5.1-NVFP4 sets index_n_heads=32.
# Autotune key includes (num_heads, head_dim), so a wrong warmup shape forces
# a re-tune on first real request.
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune(kwargs["indexer"])

    def _warmup_autotune(self, indexer) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        q = torch.empty(1, self.num_heads, _DIM_QK, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, _DIM_QK, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )
        indexer_num_heads = getattr(indexer, "n_head", _INDEXER_NUM_HEADS)
        indexer_head_dim = getattr(indexer, "head_dim", _INDEXER_HEAD_DIM)
        warmup_fp8_mqa_logits_triton(
            num_heads=indexer_num_heads, head_dim=indexer_head_dim, device=device
        )
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            warmup_fp8_paged_mqa_logits_triton(
                num_heads=indexer_num_heads,
                head_dim=indexer_head_dim,
                block_size=cfg.cache_config.block_size,
                device=device,
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output[:, : self.num_heads, :]


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @staticmethod
    def get_metadata_cls() -> type[XPUMLASparseMetadata]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DIM_QK]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True
