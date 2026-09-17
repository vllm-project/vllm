# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA backend.

This is a *reference-quality* MLA backend for the CPU platform. It is
intended to make DeepSeek-V2/V3 style models runnable on CPU (for
functional verification, tiny-model smoke tests and CI), not to be
performant. The prefill path is a plain PyTorch SDPA and the decode
path forwards to the existing CPU decode kernel
`torch.ops._C.mla_decode_kvcache` (see csrc/cpu/mla_decode.cpp).

Key design points:

* We inherit the shared MLA scaffolding (`MLACommonBackend`,
  `MLACommonImpl`, `MLACommonMetadata`) so that the same
  ``forward_impl`` in ``MLAAttention`` orchestrates weight-absorbed
  decode (MQA) and non-absorbed prefill (MHA) for us.
* The parent ``MLACommonImpl.__init__`` tries to pick a GPU prefill
  kernel (flash_attn / flashinfer / cudnn / trtllm) and raises when
  none of them are available. On CPU none of them apply, so we
  bypass that logic and set the attributes ourselves.
* Chunked prefill and prefix caching are disabled by the CPU platform
  when ``use_mla`` is set.
* The CPU decode kernel only supports ``head_dim=576``,
  ``v_head_dim=512`` and ``block_size=16`` today; the platform layer
  forces ``block_size=16`` for MLA models.
"""

from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MultipleOf,
)

logger = init_logger(__name__)


class CPUMLABackend(MLACommonBackend):
    """Attention backend descriptor for CPU MLA."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek-V2/V3 latent cache size = kv_lora_rank(512)+rope(64)=576.
        # The current CPU decode kernel only compiles for this size.
        return [576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == 16

    @staticmethod
    def get_name() -> str:
        return "CPU_MLA"

    @staticmethod
    def get_impl_cls() -> type["CPUMLAImpl"]:
        return CPUMLAImpl


class CPUMLAImpl(MLACommonImpl[MLACommonMetadata]):
    """CPU implementation of MLA attention.

    See module docstring for the overall design. This class only
    overrides the parts of the parent that would otherwise pull in
    CUDA-only dependencies.
    """

    # Decode always returns the softmax lse if the parent needs it, but
    # CPU decode currently only produces the output tensor. Report False
    # so DCP / cross-batch reductions are not requested from us.
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA-specific arguments (forwarded from `MLAAttention`).
        **mla_args,
    ) -> None:
        # NOTE: We deliberately skip ``MLACommonImpl.__init__`` because it
        # tries to pick a GPU prefill kernel and raises on CPU. Instead we
        # replicate the attribute setup here and then wire our own hooks.
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("CPU MLA only supports decoder self-attention.")
        unsupported = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported):
            raise NotImplementedError(
                "CPU MLA does not support alibi_slopes / sliding_window / "
                "logits_soft_cap."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = mla_args["q_lora_rank"]
        self.kv_lora_rank = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla_args["qk_rope_head_dim"]
        self.qk_head_dim = mla_args["qk_head_dim"]
        self.v_head_dim = mla_args["v_head_dim"]
        self.kv_b_proj = mla_args["kv_b_proj"]
        self.indexer = mla_args.get("indexer")
        self.q_pad_num_heads = mla_args.get("q_pad_num_heads")
        self.supports_quant_query_input = False

        # Never take the flashinfer specialisation of ``_concat_k_nope_k_pe``.
        self._use_flashinfer_concat_mla_k = False

        # We do not need V padding since our SDPA prefill takes k / v with
        # matching head dims.
        self._pad_v = False

        # Distributed / context-parallel bookkeeping expected by the parent.
        self.dcp_world_size = 1
        self.cp_kv_cache_interleave_size = 1

    # ------------------------------------------------------------------
    # Decode (MQA)
    # ------------------------------------------------------------------
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert attn_metadata.decode is not None
        # The parent hands us either a tuple (q_nope, q_pe) or a single
        # tensor already concatenated. CPU decode kernel wants a single
        # (bs, num_heads, head_dim=kv_lora_rank+rope) contiguous tensor.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        q = q.contiguous()

        bs, num_heads, _ = q.shape
        out = torch.zeros(
            bs, num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        # The CPU kernel is templated on int32 block_tables / seq_lens.
        block_table = attn_metadata.decode.block_table.to(torch.int32)
        seq_lens = attn_metadata.decode.seq_lens.to(torch.int32)

        ops.mla_decode_kvcache_cpu(
            out,
            q,
            kv_c_and_k_pe_cache,
            self.scale,
            block_table,
            seq_lens,
        )
        # The parent only consumes the LSE when DCP is active, which we
        # disable, so returning None here is safe.
        return out, None

    # ------------------------------------------------------------------
    # KV cache write
    # ------------------------------------------------------------------
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        k_pe_sq = k_pe.squeeze(1)
        kv_lora_rank = kv_c_normed.shape[-1]
        pe_dim = k_pe_sq.shape[-1]
        flat = kv_cache.view(-1, kv_lora_rank + pe_dim)
        slots = slot_mapping.flatten().to(torch.long)
        # Only write valid slots. Padding slots are set to -1 by the
        # scheduler and must be skipped.
        valid_mask = slots >= 0
        if not valid_mask.all().item():
            slots = slots[valid_mask]
            kv_c_normed = kv_c_normed[valid_mask]
            k_pe_sq = k_pe_sq[valid_mask]
        target_dtype = flat.dtype
        flat[slots, :kv_lora_rank] = kv_c_normed.to(target_dtype)
        flat[slots, kv_lora_rank:] = k_pe_sq.view(-1, pe_dim).to(target_dtype)
