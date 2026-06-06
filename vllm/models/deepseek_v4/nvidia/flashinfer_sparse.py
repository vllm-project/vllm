# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 FlashInfer sparse MLA backend.

Uses FlashInfer's public TRTLLM-gen sparse MLA launcher on SM10-family GPUs and
FlashInfer's ``sparse-sm120`` MLA wrapper on SM12-family GPUs. The TRTLLM-gen
variant uses a plain bf16 / per-tensor FP8 KV row, while the SM120 wrapper keeps
the pre-merge ``fp8_ds_mla`` / ``uint8`` cache layout. Both variants share the
V4 sparse-index pipeline (SWA cache + compressor + indexer, 256-token blocks,
head_size 512) with the FlashMLA V4 backend; only the attention forward differs.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    build_flashinfer_mixed_sparse_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLASparseBackend
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}

_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(
            _cdiv(max_model_len, compress_ratio),
            _C128A_TOPK_ALIGNMENT,
        )
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


class DeepseekV4FlashInferMLASparseBackend(DeepseekV4FlashMLASparseBackend):
    """Shares the FlashMLA V4 metadata/cache pipeline; swaps the attention impl.

    Inheriting from the FlashMLA V4 backend reuses its ``FlashMLASparseMetadata``
    builder (which the V4 sparse-index pipeline needs — the V3.2 FlashInfer
    builder lacks the ``c128a_*`` fields), 256-token blocks, head_size 512, and
    the (num_blocks, block_size, 512) cache shape for non-``fp8_ds_mla`` dtypes.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV4"


class _DeepseekV4FlashInferMLAAttentionBase(DeepseekV4Attention):
    """Shared DeepSeek V4 FlashInfer sparse MLA attention helpers."""

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
        # Per-tensor FP8 scale buffers + precomputed scalar BMM scales. Only the
        # per-tensor FP8 cache path consumes these; bf16 reads ``self.scale``.
        if self.kv_cache_torch_dtype != torch.float8_e4m3fn:
            return
        # TODO: load real per-tensor Q/KV scales from the checkpoint; unit
        # scales until the scale tensor names are wired.
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        # TRTLLM-gen takes scalar scale args on a distinct (correct) C++ path
        # vs 1-elem tensors, so these are Python floats. bmm1 folds the softmax
        # scale and the Q/KV per-tensor scales; bmm2 is the KV scale.
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

    def _reserve_empty_forward_workspace(self) -> None:
        pass

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        raise NotImplementedError

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The FlashInfer sparse kernels dispatch on arch-specific padded head
        # counts, so the output buffer is allocated at the padded head count while
        # q arrives at the local head count; the per-path helpers pad q before
        # invoking FlashInfer.
        assert output.shape[0] == q.shape[0] and output.shape[-1] == q.shape[-1], (
            f"output buffer shape {output.shape} incompatible with q shape {q.shape}"
        )
        assert output.shape[1] >= q.shape[1], (
            f"output heads {output.shape[1]} must be >= q heads {q.shape[1]}"
        )
        # Per-tensor FP8 q produces a bf16 attention output.
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output dtype {output.dtype} must match expected {expected_output_dtype} "
            f"for q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own compressed KV cache.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _build_trtllm_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, self.window_size
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                if num_prefill_tokens > 0:
                    prefill_topk_indices = self.topk_indices_buffer[
                        num_decode_tokens:num_tokens
                    ]
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                    top_k = 0

                decode_compressed_indices_are_local = True
                assert swa_metadata.is_valid_token is not None
                decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
                if num_decode_tokens > 0:
                    decode_compressed_indices = self.topk_indices_buffer[
                        :num_decode_tokens
                    ]
                else:
                    # Keep the logical width aligned with the mixed-batch case so
                    # pure-prefill steps reuse the same Triton specialization.
                    decode_compressed_indices = prefill_topk_indices[:0]
            else:
                if num_prefill_tokens > 0:
                    assert attn_metadata.c128a_prefill_topk_indices is not None
                    prefill_topk_indices = attn_metadata.c128a_prefill_topk_indices
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = decode_swa_indices[:0, :0]
                    top_k = 0

                if num_decode_tokens > 0:
                    assert attn_metadata.c128a_global_decode_topk_indices is not None
                    assert attn_metadata.c128a_decode_topk_lens is not None
                    decode_compressed_indices = (
                        attn_metadata.c128a_global_decode_topk_indices.view(
                            num_decode_tokens, -1
                        )
                    )
                    decode_compressed_topk_lens = attn_metadata.c128a_decode_topk_lens
                    if num_prefill_tokens == 0:
                        prefill_topk_indices = decode_compressed_indices[:0, :0]
                else:
                    decode_compressed_indices = prefill_topk_indices[:0]
                    decode_compressed_topk_lens = swa_metadata.seq_lens[:0]

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
            decode_swa_indices,
            decode_compressed_indices,
            decode_compressed_topk_lens,
            prefill_topk_indices[:num_prefill_tokens],
            query_start_loc,
            seq_lens,
            swa_metadata.token_to_req_indices[:num_tokens],
            swa_metadata.block_table[:num_reqs],
            swa_metadata.block_size,
            compressed_block_table,
            compressed_block_size,
            self.window_size,
            self.compress_ratio,
            top_k,
            decode_compressed_indices_are_local=decode_compressed_indices_are_local,
            decode_is_valid_token=decode_is_valid_token,
        )
        return compressed_kv_cache, seq_lens, sparse_indices, sparse_topk_lens

    def _forward_trtllm(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        (
            compressed_kv_cache,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = self._build_trtllm_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )

        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        padded_heads = output.shape[1]
        if query.shape[1] < padded_heads:
            padded_query = query.new_zeros(
                (query.shape[0], padded_heads, query.shape[2])
            )
            padded_query[:, : query.shape[1], :] = query
            query = padded_query

        workspace = _get_flashinfer_dsv4_workspace(q.device)
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc is not None and query_start_loc_cpu is not None

        if num_decode_tokens > 0:
            decode_cu = query_start_loc[: num_decodes + 1]
            decode_cu_cpu = query_start_loc_cpu[: num_decodes + 1]
            decode_lens_cpu = decode_cu_cpu[1:] - decode_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[:num_decode_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                seq_lens=seq_lens[:num_decodes],
                out=output[:num_decode_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=int(decode_lens_cpu.max().item()),
            )

        if num_prefill_tokens > 0:
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[num_decode_tokens:num_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=output[num_decode_tokens:num_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_cu,
                max_q_len=int(prefill_lens_cpu.max().item()),
            )

    def _reserve_sm120_decode_workspace(self) -> None:
        if self.compress_ratio <= 1:
            extra_topk = 0
        elif self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            extra_topk = self.topk_indices_buffer.shape[-1]
        elif self.compress_ratio == 128:
            extra_topk = _c128a_max_compressed(
                self.max_model_len,
                self.compress_ratio,
            )
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

    def _forward_sm120_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        extra_topk = (
            extra_sparse_indices.shape[-1] if extra_sparse_indices is not None else 0
        )
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            output.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )

        # Decode sparse indices are [num_decode_tokens, 1, topk]. Keep the
        # matching singleton query axis from the pre-merge SM120 adapter so each
        # decode token is treated as a one-token query.
        q = self._prepare_sm120_query(q, output).unsqueeze(1)
        swa_cache = _as_sparse_sm120_cache(self.swa_cache_layer.kv_cache)
        extra_cache = _as_sparse_sm120_cache(kv_cache) if kv_cache is not None else None
        assert self._sm120_wrapper is not None
        self._sm120_wrapper.run_sparse_mla(
            q=q,
            kv_cache=swa_cache,
            sparse_indices=swa_indices,
            sparse_lengths=swa_lens,
            out=output,
            sm_scale=self.scale,
            sinks=self.attn_sink,
            extra_kv_cache=extra_cache if not swa_only else None,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_lengths=extra_sparse_lengths,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    def _forward_sm120_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None

        q = self._prepare_sm120_query(q, output)
        swa_kv_paged = _as_sparse_sm120_cache(swa_k_cache)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = _as_sparse_sm120_cache(compressed_k_cache)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )
            chunk_tokens = query_end - query_start

            mid_out = None
            mid_lse = None
            if chunk_tokens <= _DECODE_MAX_TOKENS:
                extra_topk = (
                    extra_sparse_indices_chunk.shape[-1]
                    if extra_sparse_indices_chunk is not None
                    else 0
                )
                mid_out, mid_lse = _get_decode_scratch(
                    chunk_tokens,
                    output.shape[1],
                    output.shape[-1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    extra_topk,
                )

            assert self._sm120_wrapper is not None
            self._sm120_wrapper.run_sparse_mla(
                q=q[query_start:query_end],
                kv_cache=swa_kv_paged,
                sparse_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                sparse_lengths=swa_metadata.prefill_swa_lens[query_start:query_end],
                out=output[query_start:query_end],
                sm_scale=self.scale,
                sinks=self.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_lengths=extra_sparse_lengths_chunk,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )


class DeepseekV4FlashInferTRTLLMAttention(_DeepseekV4FlashInferMLAAttentionBase):
    """DeepSeek V4 FlashInfer sparse MLA attention via TRTLLM-gen."""

    use_fp8_ds_mla_layout: ClassVar[bool] = False

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
            "(TRTLLM-gen kernel requires h_q in {64, 128})."
        )

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        self._forward_trtllm(
            q=q,
            kv_cache=self_kv_cache,
            swa_k_cache=swa_kv_cache,
            swa_metadata=swa_metadata,
            attn_metadata=flashmla_metadata,
            swa_only=swa_only,
            output=output,
        )


class DeepseekV4FlashInferSM120Attention(_DeepseekV4FlashInferMLAAttentionBase):
    """DeepSeek V4 FlashInfer sparse MLA attention via sparse-sm120."""

    use_fp8_ds_mla_layout: ClassVar[bool] = True

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
            f"DeepseekV4 FlashInfer MLA Sparse does not support {num_heads} heads "
            "(SM120 kernel requires h_q in {16, 32, 64, 128})."
        )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires FlashInfer's "
                "sparse-sm120 MLA wrapper."
            )

        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        wrapper_device = torch.device("cuda", torch.accelerator.current_device_index())
        self._sm120_wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(1, dtype=torch.int8, device=wrapper_device),
            backend="sparse-sm120",
            max_num_tokens=max_tokens,
            max_num_heads=self.padded_heads,
            d_v=self.head_dim,
            kv_scale_format="auto",
        )

    def _reserve_empty_forward_workspace(self) -> None:
        self._reserve_sm120_decode_workspace()

    def _forward_sparse_impl(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        flashmla_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        self_kv_cache: torch.Tensor | None,
        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_sm120_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_sm120_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )
