# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer packed sparse-MLA SM120 decode (gated).

Subclasses the FlashMLA V4 attention to reuse its packed ``fp8_ds_mla`` KV cache,
sparse-index metadata, and packed prefill, and overrides only the decode path to
use FlashInfer's official SM120 packed sparse-MLA decode kernel (FlashInfer
PR3395, merged in flashinfer >= 0.6.13). That kernel scales better at high
concurrency in the MTP speculative-verify (multi-query) decode shape than the
FlashMLA decode kernel, which is the root cause of the C8-C64 ctx0 decode gap.

Implementation note: flashinfer main exposes this kernel through the
``trtllm_batch_decode_sparse_mla_dsv4`` wrapper, but that wrapper re-validates
inputs and -- critically -- carves the split-K ``mid_out``/``mid_lse`` scratch
from a fixed workspace only for ``num_tokens <= 64``, falling back to a fresh
``torch.empty`` of hundreds of MB on every decode step above that. The MTP
multi-query decode shape routinely exceeds 64 tokens (C32/C64), so that per-step
allocation dominates and makes the wrapper materially slower than the FlashMLA
path. We instead drive the same kernel through its low-level
``_SparseMLAPagedAttentionRunner``, constructed once and fed graph-stable scratch
from vLLM's workspace manager -- so the scratch is reserved during warmup and
reused, never reallocated per step.

Gated behind ``VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE``; selected only on SM12x
when the official packed kernel is importable (see ``_select_dsv4_attn_cls``).
Default on; gate-off behavior is identical to the FlashMLA decode path.
"""

from typing import TYPE_CHECKING

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# Split-K decode scratch sizing, mirrored from the FlashInfer sparse-sm120
# kernel (``_BI`` = 64 candidates per partition tile): one tile per SWA top-k
# plus one per compressed (extra) top-k. Cap the warmup reservation at the
# largest single-graph decode batch.
_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(_cdiv(max_model_len, compress_ratio), _C128A_TOPK_ALIGNMENT)
        * _C128A_TOPK_ALIGNMENT
    )


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    topk: int,
    extra_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, head_dim), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


def _as_sparse_sm120_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dtype == torch.float8_e4m3fn:
        kv_cache = kv_cache.view(torch.uint8)
    if kv_cache.dim() == 4:
        return kv_cache
    return kv_cache.unsqueeze(-2)


def _get_prefill_swa_scratch(
    num_tokens: int, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Graph-stable per-token SWA window indices + lengths for the prefill tokens.
    swa_indices, swa_lens = current_workspace_manager().get_simultaneous(
        ((num_tokens, 1, window_size), torch.int32),
        ((num_tokens,), torch.int32),
    )
    return swa_indices, swa_lens


class DeepseekV4FlashInferSM120DecodeAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA V4 attention with the official FlashInfer SM120 packed decode.

    Reuses every FlashMLA V4 behavior (packed ``fp8_ds_mla`` cache, metadata
    pipeline, packed prefill); only :meth:`_forward_decode` differs.
    """

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 FlashInfer sparse-sm120 decode does not support "
            f"{num_heads} heads (SM120 kernel requires h_q in {{16, 32, 64, 128}})."
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_trtllm_sparse_mla_dsv4

        if not has_flashinfer_trtllm_sparse_mla_dsv4():
            raise RuntimeError(
                "VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE requires FlashInfer's "
                "SM120 packed sparse-MLA decode kernel "
                "(trtllm_batch_decode_sparse_mla_dsv4, PR3395, "
                "flashinfer >= 0.6.13)."
            )

        from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        runner_device = torch.device("cuda", torch.accelerator.current_device_index())
        # Construct the low-level runner once: its only per-instance state is a
        # pre-sized LSE buffer. We feed it graph-stable mid_out/mid_lse scratch
        # explicitly on every call, so it never allocates per step.
        self._sm120_runner = _SparseMLAPagedAttentionRunner(
            max_num_tokens=max_tokens,
            max_num_heads=self.padded_heads,
            d_v=self.head_dim,
            kv_scale_format="auto",
            device=runner_device,
        )
        logger.info_once(
            "DeepSeek V4: using official FlashInfer SM120 packed sparse-MLA decode "
            "via the low-level runner (VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE=1)."
        )

    def _reserve_sm120_decode_workspace(self) -> None:
        if self.compress_ratio <= 1:
            extra_topk = 0
        elif self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            extra_topk = self.topk_indices_buffer.shape[-1]
        elif self.compress_ratio == 128:
            extra_topk = _c128a_max_compressed(self.max_model_len, self.compress_ratio)
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        _get_decode_scratch(
            _max_decode_workspace_tokens(self.max_num_batched_tokens),
            self.padded_heads,
            self.head_dim,
            self.window_size,
            extra_topk,
        )

    def _prepare_sm120_query(
        self, q: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # The SM120 packed kernel consumes a bf16 query; the FlashMLA fp8 path
        # keeps q in fp8, so convert here. q already arrives padded to
        # ``padded_heads`` by the outer attention wrapper.
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert q.dtype == torch.float8_e4m3fn
            q = q.to(torch.bfloat16)
        else:
            assert q.dtype == torch.bfloat16
        padded_heads = output.shape[1]
        if q.shape[1] < padded_heads:
            padded_query = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
            padded_query[:, : q.shape[1], :] = q
            q = padded_query
        return q.contiguous()

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Mirror the FlashMLA warmup branch but also reserve the graph-stable
        # split-K decode scratch the sparse-sm120 kernel needs, then defer to the
        # parent for the real prefill/decode split (which calls our overridden
        # _forward_decode).
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            self._reserve_prefill_workspace(self)
            self._reserve_sm120_decode_workspace()
            output.zero_()
            return
        super().forward_mqa(q, kv, positions, output)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: "DeepseekV4FlashMLAMetadata | None",
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Identical decode-side index/length construction to the FlashMLA decode
        # path; only the kernel launch below differs.
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            # The sparse-sm120 kernel asserts the extra (compressed) index
            # tensor is int32 and contiguous; current's metadata builder can hand
            # back a non-contiguous view, so normalize before the launch.
            if topk_indices is not None:
                topk_indices = topk_indices.contiguous()

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None

        extra_topk = topk_indices.shape[-1] if topk_indices is not None else 0
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            output.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )

        # Each decode token is a one-token query: [num_decode_tokens, 1, h, d];
        # the runner squeezes the singleton s_q dim internally.
        q = self._prepare_sm120_query(q, output).unsqueeze(1)
        swa_cache = _as_sparse_sm120_cache(self.swa_cache_layer.kv_cache)
        extra_cache = (
            _as_sparse_sm120_cache(kv_cache)
            if (kv_cache is not None and not swa_only)
            else None
        )
        self._sm120_runner.run(
            q,
            swa_cache,
            swa_indices,
            output,
            self.scale,
            topk_length=swa_lens,
            attn_sink=self.attn_sink,
            extra_kv_cache=extra_cache,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "DeepseekV4FlashMLAMetadata | None",
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        import vllm.envs as envs

        # Packed prefill is an independent opt-in on top of the decode port; when
        # off, defer to the FlashMLA indexed-D512 prefill path byte-for-byte.
        if not envs.VLLM_DEEPSEEK_V4_FLASHINFER_SM120_PREFILL:
            super()._forward_prefill(
                q,
                positions,
                compressed_k_cache,
                swa_k_cache,
                output,
                attn_metadata,
                swa_metadata,
            )
            return

        swa_only = attn_metadata is None
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_prefill_tokens == 0:
            return

        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.seq_lens is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.block_table is not None

        # --- Prefill SWA window indices. The metadata builder hoists this once per
        # step (DeepseekSparseSWAMetadataBuilder.build widens its decode-SWA launch
        # over the prefill tail), so steady-state just reads the precomputed views
        # and skips ~60 redundant per-layer kernel launches. The builder deliberately
        # leaves them None on the warmup/profile dummy (its all-(-1) slot_mapping
        # makes is_valid_token all-False over the prefill tail) and during CUDA-graph
        # capture; we then self-compute exactly as before, keeping those paths
        # byte-identical to the validated v1 (the prefill packed kernel is a no-op
        # over the all-invalid dummy: every token gets swa_len=0).
        swa_indices = swa_metadata.prefill_swa_indices
        swa_lens = swa_metadata.prefill_swa_lens
        if swa_indices is None or swa_lens is None:
            from vllm.v1.attention.backends.mla.sparse_swa import (
                _compute_swa_indices_and_lens_kernel,
            )

            swa_idx_full, swa_len_full = _get_prefill_swa_scratch(
                num_tokens, self.window_size
            )
            _compute_swa_indices_and_lens_kernel[(num_tokens,)](
                swa_idx_full,
                swa_idx_full.stride(0),
                swa_len_full,
                self.window_size,
                swa_metadata.query_start_loc,
                swa_metadata.seq_lens,
                swa_metadata.token_to_req_indices,
                swa_metadata.is_valid_token,
                swa_metadata.block_table,
                swa_metadata.block_table.stride(0),
                swa_metadata.block_size,
                token_offset=0,
                TRITON_BLOCK_SIZE=1024,
            )
            swa_indices = swa_idx_full[num_decode_tokens:num_tokens]
            swa_lens = swa_len_full[num_decode_tokens:num_tokens]

        # --- Compressed (extra) prefill indices, mirroring the FlashMLA prefill
        # construction but converted to global slots for the packed kernel.
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                prefill_local = self.topk_indices_buffer[num_decode_tokens:num_tokens]
                # Rebase the indexer's BATCH-GLOBAL compressed top-k positions
                # (cu_seqlen_ks = exclusive cumsum of seq_len // compress_ratio; see
                # indexer.py) to per-request-local so block_table[req] maps them
                # in-range. Without this, req>0 positions overflow into the wrong
                # request's physical blocks. No-op at num_prefills==1 (cu_base[0]==0).
                comp_lens = (
                    swa_metadata.seq_lens[num_decodes:num_reqs] // self.compress_ratio
                )
                cu_base = (torch.cumsum(comp_lens, dim=0) - comp_lens).to(torch.int32)
                req_local = (
                    swa_metadata.token_to_req_indices[num_decode_tokens:num_tokens]
                    - num_decodes
                ).long()
                base_per_token = cu_base[req_local].unsqueeze(1)
                prefill_local = torch.where(
                    prefill_local >= 0, prefill_local - base_per_token, prefill_local
                )
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    prefill_local,
                    swa_metadata.token_to_req_indices[num_decode_tokens:num_tokens],
                    attn_metadata.block_table[:num_reqs],
                    block_size,
                    swa_metadata.is_valid_token[num_decode_tokens:num_tokens],
                )
                topk_indices = global_indices.view(num_prefill_tokens, 1, -1)
            else:
                assert attn_metadata.c128a_prefill_topk_indices is not None
                # c128a_prefill_topk_indices are per-request-local compressed
                # block positions (0..n-1, -1 padded) -- the same basis the
                # decode path maps through the block table. The paged packed
                # runner has no block table of its own, so convert to global
                # KV-cache slots (+ lens) here, mirroring the C4 branch above.
                # C128A is already per-request-local, so unlike C4 it needs no
                # cu_base rebase.
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    attn_metadata.c128a_prefill_topk_indices,
                    swa_metadata.token_to_req_indices[num_decode_tokens:num_tokens],
                    attn_metadata.block_table[:num_reqs],
                    block_size,
                    swa_metadata.is_valid_token[num_decode_tokens:num_tokens],
                )
                topk_indices = global_indices.view(num_prefill_tokens, 1, -1)
            topk_indices = topk_indices.contiguous()

        # --- Launch the packed prefill kernel via the runner. num_tokens > 64
        # auto-dispatches the prefill kernel; mid_out/mid_lse are decode-only and
        # only needed for the (rare) <=64-token prefill chunk.
        query = self._prepare_sm120_query(q, output)
        # Bug-C guard: under CUDA-graph padding or MTP-draft, q can carry more rows
        # than the real prefill-token count; the runner sizes its writes by the
        # query row count, so slice to num_prefill_tokens to match output/indices/
        # scratch (no-op in the common, unpadded case).
        if query.shape[0] > num_prefill_tokens:
            query = query[:num_prefill_tokens]
        # The packed kernel hard-asserts output.size(0) == num_tokens (derived from
        # the sliced query row count). The output buffer can still carry padded
        # prefill rows (output[num_decode_tokens:] of a CUDA-graph / MTP-draft padded
        # batch), so slice it the same way as the query/indices/scratch above. It is
        # a view into the same storage and the padded tail rows are never read or
        # written downstream, so this only narrows what the kernel writes; no-op in
        # the common unpadded case. Without it the kernel aborts (84 vs 83).
        out = (
            output[:num_prefill_tokens]
            if output.shape[0] > num_prefill_tokens
            else output
        )
        swa_cache = _as_sparse_sm120_cache(swa_k_cache)
        extra_cache = (
            _as_sparse_sm120_cache(compressed_k_cache)
            if (compressed_k_cache is not None and not swa_only)
            else None
        )
        mid_out = None
        mid_lse = None
        if num_prefill_tokens <= _DECODE_MAX_TOKENS:
            extra_topk = topk_indices.shape[-1] if topk_indices is not None else 0
            mid_out, mid_lse = _get_decode_scratch(
                num_prefill_tokens,
                output.shape[1],
                output.shape[-1],
                swa_indices.shape[-1],
                extra_topk,
            )
        self._sm120_runner.run(
            query,
            swa_cache,
            swa_indices,
            out,
            self.scale,
            topk_length=swa_lens,
            attn_sink=self.attn_sink,
            extra_kv_cache=extra_cache,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
