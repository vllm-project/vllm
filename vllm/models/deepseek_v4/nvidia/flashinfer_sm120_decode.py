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
Default off; gate-off behavior is identical to the FlashMLA decode path.
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


class DeepseekV4FlashInferSM120Attention(DeepseekV4FlashMLAAttention):
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
