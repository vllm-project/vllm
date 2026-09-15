# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 attention on FlashMLA's mega-attention kernel (SM100).

One kernel does Q RoPE, sparse attention, the inverse RoPE of the output and
its FP8 cast, and writes straight into the buffer ``wo_a`` consumes -- so the
layer declares both halves of the interface contract:
``accepts_unnormed_unroped_query`` (the kernel RoPEs Q itself) and
``produces_inv_roped_quantized_output`` (its output is a QuantizedActivation).

``wq_b`` rows and ``wo_a`` columns are permuted once at load so the surrounding
GEMMs speak the kernel's chunk-interleaved layouts directly. A step's prefill
and decode segments write disjoint token ranges of one output buffer pair, so
a single ``wo_a`` einsum covers the whole step.
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    permute_wo_a_,
    permute_wq_b_,
)
from vllm.models.deepseek_v41.common.ops.q_layout import pad_fused_q_heads
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    FlashMLAMegaAttnBackend,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.ops.flashmla import (
    alloc_mega_attn_output,
    flash_mla_mega_attn_decode,
    flash_mla_mega_attn_prefill,
    mega_attn_token_range,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# Decode always goes through the mega kernel: there is no batch size at which
# falling back to the split-KV decode pays, so there is no minimum-token
# threshold and no fallback path to maintain. What decides whether this layer
# is worth selecting at all is the live/padded head ratio, not the batch size.
#
# Measured on one GB300 with benchmarks/kernels/benchmark_dsv41_mega_attn.py,
# CUDA graphs on, counting every per-step op each path runs (topk_swa=128,
# topk_extra=512, NVFP4 compressed cache), as mega / split-KV microseconds:
#
#     s_q                    1      32     128     256     512
#     64 live heads       29/29   29/31   31/40   53/72   95/140
#     16 live heads       31/29   31/29   33/33   59/57  107/108
#
# The kernel absorbs the inverse RoPE and the FP8 cast, whose cost scales with
# *live* heads, while its own cost scales with *padded* heads (64 here) and it
# needs a separate Q-pad kernel the split-KV path gets for free inside its
# fused KV insert. So at 64 live heads it wins by up to 1.47x, and at 16 (TP4
# on a 64-head model) the two cancel and it is a wash. Measure before assuming
# a win at high TP -- and measure under CUDA graphs: in eager mode launch
# overhead swamps both paths and overstates the fused one several-fold.


class DeepseekV4MegaAttnAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA mega-attention layer for DeepSeek V4.1 (SM100)."""

    backend_cls = FlashMLAMegaAttnBackend
    accepts_unnormed_unroped_query: ClassVar[bool] = True
    produces_inv_roped_quantized_output: ClassVar[bool] = True
    # This kernel is the only one that reads an NVFP4 compressed cache, so it
    # is what an unspecific --kv-cache-dtype resolves to here.
    packed_kv_cache_dtype: ClassVar[CacheDType] = "nvfp4_ds_mla"

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        if self.n_local_heads % WV_GROUP_SIZE:
            raise ValueError(
                f"{self.prefix}: mega attention needs the local head count "
                f"({self.n_local_heads}) to be a multiple of {WV_GROUP_SIZE}."
            )
        if self.n_local_heads // self.n_local_groups != WV_GROUP_SIZE:
            raise ValueError(
                f"{self.prefix}: mega attention needs {WV_GROUP_SIZE} heads per "
                "wo_a group."
            )
        self.n_wv_group = self.padded_heads // WV_GROUP_SIZE
        self._fused_layouts_ready = False

    # ---- weights -----------------------------------------------------------

    def finalize_loaded_weights(self) -> None:
        """Permute wq_b rows / wo_a columns into the kernel's layouts.

        Idempotent: a second post-load pass must not permute twice.
        """
        if self._fused_layouts_ready:
            return
        permute_wq_b_(
            self.wq_b.weight.data, self.wq_b.weight_scale.data, self.n_local_heads
        )
        permute_wo_a_(
            self.wo_a.weight.data,
            self.wo_a.weight_scale.data,
            self.n_local_heads // self.n_local_groups,
        )
        self._fused_layouts_ready = True

    # ---- interface contract ------------------------------------------------

    def _o_proj(
        self, attn_out: QuantizedActivation, positions: torch.Tensor
    ) -> torch.Tensor:
        """wo_a + wo_b over the kernel's already-quantized output.

        The inverse RoPE and the FP8 cast happened inside the attention
        kernel, and ``wo_a`` is permuted for its output layout, so there is
        nothing to rotate or quantize here and ``positions`` is unused.
        """
        del positions
        return self.wo_b(self._wo_a_einsum(attn_out).flatten(1))

    def _alloc_attn_out(
        self, num_tokens: int, hidden_states: torch.Tensor
    ) -> QuantizedActivation:
        return alloc_mega_attn_output(num_tokens, self.n_wv_group, hidden_states.device)

    def _prepare_q_and_insert_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        self._insert_swa_kv(kv, positions, attn_metadata)
        # MRV2 captures this preparation before the eager attention region.
        return pad_fused_q_heads(q, self.padded_heads)

    # ---- forward -----------------------------------------------------------

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: QuantizedActivation,
    ) -> None:
        if not self._fused_layouts_ready:
            raise RuntimeError(
                f"{self.prefix}: wq_b / wo_a were never permuted for the mega "
                "attention kernel; refusing to run with mismatched layouts."
            )
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: reserve the prefill workspace and produce zeros.
            self._reserve_prefill_workspace(q)
            output.data.zero_()
            output.scale.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", attn_metadata[self.swa_cache_layer.prefix]
        )
        # The kernel takes int32 RoPE positions.
        positions_int32 = positions.to(torch.int32)
        num_decode_tokens = swa_metadata.num_decode_tokens
        assert q.shape[1] == self.padded_heads

        if swa_metadata.num_prefills > 0:
            self._forward_prefill_mega(
                q[num_decode_tokens:],
                positions_int32[num_decode_tokens:],
                flashmla_metadata,
                swa_metadata,
                output,
                num_decode_tokens,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode_mega(
                q[:num_decode_tokens],
                positions_int32[:num_decode_tokens],
                flashmla_metadata,
                swa_metadata,
                mega_attn_token_range(output, 0, num_decode_tokens),
            )

    def _reserve_prefill_workspace(self, q: torch.Tensor) -> None:
        swa_only = self.compress_ratio == 0
        n = 0 if swa_only else -(-self.max_model_len // self.compress_ratio)
        m = n + self.window_size + self.max_num_batched_tokens
        if swa_only:
            top_k = 0
        else:
            assert self.topk_indices_buffer is not None
            top_k = self.topk_indices_buffer.shape[-1]
        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
        current_workspace_manager().get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, m, q.shape[-1]), torch.bfloat16),
            ((self.max_num_batched_tokens, combined_topk), torch.int32),
            ((self.max_num_batched_tokens,), torch.int32),
        )

    def _decode_extra(
        self,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """The compressed cache and its ``[n, topk]`` slot indices / lengths."""
        if self.compress_ratio == 0:
            return None, None, None
        assert flashmla_metadata is not None
        assert swa_metadata.is_valid_token is not None
        assert self.topk_indices_buffer is not None
        num_decode_tokens = swa_metadata.num_decode_tokens
        indices, lens = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            flashmla_metadata.block_table[: swa_metadata.num_decodes],
            flashmla_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:num_decode_tokens],
        )
        return self._compressed_kv_cache().unsqueeze(-2), indices, lens

    def _forward_decode_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        out: QuantizedActivation,
    ) -> None:
        extra_cache, extra_idx, extra_len = self._decode_extra(
            flashmla_metadata, swa_metadata
        )
        assert swa_metadata.decode_swa_indices is not None
        flash_mla_mega_attn_decode(
            q,
            self.swa_cache_layer.kv_cache.unsqueeze(-2),
            swa_metadata.decode_swa_indices.view(q.shape[0], -1),
            self.scale,
            positions_int32,
            self.rotary_emb.cos_sin_cache,
            self.n_wv_group,
            out,
            attn_sink=self.attn_sink,
            topk_length=swa_metadata.decode_swa_lens,
            extra_k_cache=extra_cache,
            extra_indices=extra_idx,
            extra_topk_length=extra_len,
        )

    def _forward_prefill_mega(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        out: QuantizedActivation,
        token_base: int,
    ) -> None:
        swa_only = self.compress_ratio == 0
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:][
            : swa_metadata.num_prefill_tokens
        ]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty when num_prefills > 0"
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv_ws, idx_ws, lens_ws = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            if not swa_only:
                assert flashmla_metadata is not None
                dequantize_and_gather_k_cache(
                    kv_ws[:chunk_size],
                    self._compressed_kv_cache(),
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=flashmla_metadata.block_table[num_decodes:][
                        chunk_start:chunk_end
                    ],
                    block_size=flashmla_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache(
                kv_ws[:chunk_size],
                self.swa_cache_layer.kv_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_metadata.block_table[num_decodes:][
                    chunk_start:chunk_end
                ],
                block_size=swa_metadata.block_size,
                offset=chunk_n,
            )
            first, last = num_decodes + chunk_start, num_decodes + chunk_end
            qs = int(query_start_loc_cpu[first] - prefill_token_base)
            qe = int(query_start_loc_cpu[last] - prefill_token_base)
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[qs:qe],
                query_start_loc[first : last + 1],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
                out=(idx_ws[: qe - qs], lens_ws[: qe - qs]),
                left_visible=(
                    swa_metadata.prefill_left_visible[
                        num_decode_tokens + qs : num_decode_tokens + qe
                    ]
                    if swa_metadata.prefill_left_visible is not None
                    else None
                ),
                right_visible=(
                    swa_metadata.prefill_right_visible[
                        num_decode_tokens + qs : num_decode_tokens + qe
                    ]
                    if swa_metadata.prefill_right_visible is not None
                    else None
                ),
                max_image_tokens=self.max_image_tokens,
            )
            flash_mla_mega_attn_prefill(
                q[qs:qe],
                kv_ws.view(-1, 1, q.shape[-1]),
                combined_indices.unsqueeze(1),
                self.scale,
                positions_int32[qs:qe],
                self.rotary_emb.cos_sin_cache,
                self.n_wv_group,
                mega_attn_token_range(out, token_base + qs, token_base + qe),
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
