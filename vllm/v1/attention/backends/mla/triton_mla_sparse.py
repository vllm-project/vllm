# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for compute capabilities without a native
FlashMLA sparse kernel (e.g. SM120 / RTX PRO 6000).

This reuses the full CUDA sparse-MLA plumbing from ``flashmla_sparse`` — its
metadata, builder, prefill/decode split, NoPE-aware ``forward_mqa`` and
index-globalization — and swaps out only the one sm90/sm100-only piece: the
``flash_mla_sparse_fwd`` bf16 decode/prefill kernel, replaced by a Triton
split-KV sparse attention kernel that runs on any recent CUDA arch.
"""

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

# V3.2 indexers don't expose `n_head`; GLM-5.1-NVFP4 sets index_n_heads=32.
# Autotune key includes (num_heads, head_dim), so a wrong warmup shape forces
# a re-tune on first real request.
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128


class TritonMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl with the bf16 kernel swapped for Triton split-KV.

    Everything else (metadata, index globalization, the NoPE-aware
    ``forward_mqa`` concat path, prefill/decode routing) is inherited
    unchanged from ``FlashMLASparseImpl``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune(kwargs.get("indexer"))

    def _warmup_autotune(self, indexer) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]

        # dim_qk = kv_lora_rank + qk_rope_head_dim; self.head_size carries it
        # (512 for glm5_next NoPE, 576 for DeepSeek-V3.2 / GLM-5 rope MLA).
        dim_qk = self.head_size
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
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

    def _bf16_flash_mla_kernel(
        self,
        q,  # [num_mqa_tokens, num_heads, dim_qk]
        kv_c_and_k_pe_cache,  # [num_blocks, block_size, dim_qk]
        topk_indices,  # [num_mqa_tokens, topk] GLOBAL slots, compacted, -1 tail
        topk_length=None,  # per-token valid count (unused: -1 tail already bounds)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The inherited FlashMLASparseImpl._forward_bf16_kv already globalized
        # the per-request indices with return_valid_counts=True, so valid slots
        # are compacted to the front and the tail is guaranteed -1. The Triton
        # kernel masks -1 / out-of-range (indices >= 0 & < seq_kv), so it needs
        # no separate topk_length bound (unlike flash_mla_sparse_fwd). No head
        # padding needed either — the kernel tiles over any head count.
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
        # LSE is only consumed under decode context parallelism, which the base
        # __init__ already rejects for the bf16 sparse path (NotImplementedError
        # unless the kv-cache is fp8_ds_mla). This Triton path is bf16-only, so
        # DCP never reaches here and returning None for the LSE is safe.
        return output[:, : self.num_heads, :], None


class TritonMLASparseBackend(FlashMLASparseBackend):
    # Triton kernel is bf16-only; fp8_ds_mla still needs the native FlashMLA
    # fp8 decode kernel (sm90/sm100), so drop it from the supported set here.
    supported_kv_cache_dtypes: list[CacheDType] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # The native FlashMLA sparse kernel only handles the 576 rope-MLA
        # layout (512 NoPE + 64 RoPE). The Triton kernel derives its geometry
        # from dim_qk at dispatch, so it also serves the pure-NoPE 512 layout
        # (glm5_next / GLM-5.3-Flash, qk_rope_head_dim=0).
        return [512, 576]

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # The whole point: run sparse MLA where FlashMLA's kernel can't.
        return True
