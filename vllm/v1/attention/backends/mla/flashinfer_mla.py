# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar, Optional, Union

import torch

# Removed TRT-LLM MLA import; we use the wrapper instead:
# from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.mla import BatchMLAPagedAttentionWrapper

from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport

logger = init_logger(__name__)

# Workspace for the MLA wrapper. Wrapper expects a byte buffer; allocate int8.
FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class FlashInferMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # enable spec-as-decode optimization
    supports_uniform_spec_as_decode: ClassVar[bool] = True

    # enable full CUDA Graph support for decode-only capture
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class FlashInferMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder


# Keep a reusable workspace on device
_g_fi_workspace_i8 = torch.empty(
    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
    dtype=torch.int8,
    device="cuda",
)


class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferMLAImpl"
            )

        # Batch-invariant, deterministic path via FlashInfer MLA wrapper
        self._workspace_buffer_i8 = _g_fi_workspace_i8
        # Select FA2 backend unless overridden
        self._mla_backend = "fa2"
        self._mla_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None
        self._planned_bs: Optional[int] = None

        # Cache scales
        self.bmm1_scale: Optional[float] = None
        self.bmm2_scale: Optional[float] = None

    def _ensure_wrapper(self, device: torch.device):
        if self._mla_wrapper is None or self._workspace_buffer_i8.device != device:
            # Recreate wrapper if device changes
            self._workspace_buffer_i8 = torch.empty(
                FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
                dtype=torch.int8,
                device=device,
            )
            self._mla_wrapper = BatchMLAPagedAttentionWrapper(
                self._workspace_buffer_i8, backend=self._mla_backend
            )

    def _plan_for_batch(
        self,
        batch_size: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ):
        assert self._mla_wrapper is not None

        # For decode, each query length is 1 => qo_indptr = [0, 1, ..., B]
        qo_indptr = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device=block_table.device
        )

        # We use page_size=1 for the compressed KV (ckv/kpe are [num_pages, 1, dim])
        page_size = 1

        # Number of KV "pages" per request equals seq_len when page_size=1.
        # kv_len_arr: [B]
        kv_len_arr = seq_lens.to(dtype=torch.int32)

        # We can provide a kv_indices buffer larger than kv_indptr[-1]. To avoid
        # per-request slicing, flatten the full block table and stride via kv_indptr.
        # block_table shape: [B, max_pages]
        B, max_pages = block_table.shape
        kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=block_table.device)
        kv_indptr = kv_indptr * max_pages
        kv_indices = block_table.contiguous().view(-1).to(dtype=torch.int32)

        # sm_scale = 1/sqrt(head_dim_ckv + head_dim_kpe), already provided by self.scale
        sm_scale = float(self.scale)

        # num_heads here is local heads
        num_heads = self.num_heads
        head_dim_ckv = int(self.qk_nope_head_dim)
        head_dim_kpe = int(self.qk_rope_head_dim)

        # causal = True for decoder self-attention
        causal = True

        # Plan the MLA attention computation
        self._mla_wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_dtype,
            kv_dtype,
            False,  # use_profiler
        )

        self._planned_bs = batch_size

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        # q comes in as either (q_nope, q_pe) or concatenated; reshape to [B, H, D]
        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            # q could be [B, q_len, H, D], but for decode we require q_len == 1.
            # If uneven, earlier code unsqueezed an extra dim; here we enforce decode=1.
            # Flatten to [B, H, D] and split by dims.
            q = q.view(-1, q.shape[-2], q.shape[-1])  # [B, H, Dtot]
            head_dim_ckv = int(self.qk_nope_head_dim)
            q_nope, q_pe = q[..., :head_dim_ckv], q[..., head_dim_ckv:]

        # Derive ckv/kpe from combined KV cache:
        # expect last dim = head_dim_ckv + head_dim_kpe
        head_dim_ckv = int(self.qk_nope_head_dim)
        head_dim_kpe = int(self.qk_rope_head_dim)
        total_dim = head_dim_ckv + head_dim_kpe
        kv_flat = kv_c_and_k_pe_cache
        assert kv_flat.shape[-1] == total_dim, (
            f"Unexpected KV dim: got {kv_flat.shape[-1]}, expected {total_dim}"
        )

        # Reshape KV into [num_pages, page_size(=1), dim]
        num_pages = kv_flat.shape[0]
        ckv_cache = (
            kv_flat[..., :head_dim_ckv].contiguous().view(num_pages, 1, head_dim_ckv)
        )
        kpe_cache = (
            kv_flat[..., head_dim_ckv:].contiguous().view(num_pages, 1, head_dim_kpe)
        )

        if self.bmm1_scale is None:
            self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float

        # Prepare wrapper and plan for current batch if needed
        decode_md = attn_metadata.decode
        block_table = decode_md.block_table  # [B, max_pages]
        seq_lens = decode_md.seq_lens  # [B]
        batch_size = block_table.shape[0]

        # Ensure wrapper initialized on the correct device
        self._ensure_wrapper(block_table.device)
        assert self._mla_wrapper is not None

        # Plan per batch size (simple cache).
        # If your shapes vary beyond batch, replan each call.
        if self._planned_bs != batch_size:
            self._plan_for_batch(
                batch_size=batch_size,
                block_table=block_table,
                seq_lens=seq_lens,
                q_dtype=q_nope.dtype,
                kv_dtype=ckv_cache.dtype,
            )

        # Run MLA attention
        # Return LSE is currently unused by vLLM MLA path
        o, _ = self._mla_wrapper.run(
            q_nope.contiguous(),
            q_pe.contiguous(),
            ckv_cache,
            kpe_cache,
            return_lse=False,
        )

        # Ensure consistent shape with vLLM expectations: [tokens, H, head_dim_ckv]
        # Decode returns one token per request => tokens == batch_size
        o = o.view(-1, o.shape[-2], o.shape[-1])
        return o, None
