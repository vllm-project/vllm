# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 attention on FlashMLA's fused sparse kernel (SM100).

Q RoPE, sparse attention, inverse RoPE and the FP8 cast run in one kernel whose
output feeds the ``wo_a`` einsum; ``wq_b`` rows and ``wo_a`` columns are
permuted at load (``finalize_loaded_weights``) so the GEMMs speak the kernel's
chunk-interleaved layouts. Small decode batches fall back to the split-KV
kernel (``dsv4_fused_decode_min_tokens``).
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4_1.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    permute_wo_a_,
    permute_wq_b_,
)
from vllm.models.deepseek_v4_1.common.ops.q_layout import dsv41_q_layout
from vllm.models.deepseek_v4_1.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.ops.flashmla import (
    flash_mla_fused_sparse_decode,
    flash_mla_fused_sparse_prefill,
    flash_mla_with_kvcache,
    is_flashmla_fused_sparse_supported,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)

_SUPPORTED_KV_CACHE_DTYPES = ("auto", "fp8", "fp8_ds_mla", "nvfp4_ds_mla")
_FUSED_KV_LAYOUTS = ("fp8_ds_mla", "nvfp4_ds_mla")


def dsv4_fused_attention_enabled(vllm_config: VllmConfig) -> bool:
    """Resolve ``attention_config.dsv4_fused_attention`` for this model."""
    flag = vllm_config.attention_config.dsv4_fused_attention
    if flag is False:
        return False
    ok, reason = is_flashmla_fused_sparse_supported()
    hf_config = vllm_config.model_config.hf_config
    if ok and hf_config.num_attention_heads // hf_config.o_groups != WV_GROUP_SIZE:
        ok, reason = False, f"needs {WV_GROUP_SIZE} heads per o_group"
    if ok and vllm_config.cache_config.cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
        ok, reason = False, "needs the fp8_ds_mla or nvfp4_ds_mla KV cache layout"
    if not ok:
        if flag is True:
            raise ValueError(f"dsv4_fused_attention=True is unsupported: {reason}")
        logger.info_once("FlashMLA fused sparse attention disabled: %s", reason)
    return ok


def _weight_was_loaded(prefix: str, loaded_params: set[str] | None, leaf: str) -> bool:
    """Whether the layer at ``prefix`` had its ``leaf`` parameter loaded.

    ``loaded_params`` may be relative to an enclosing module (AutoWeightsLoader
    hands each child its own namespace), so match on the ``layers.<id>`` tail of
    the prefix rather than the full name.
    """
    if loaded_params is None:
        return True
    idx = prefix.rfind("layers.")
    anchor = prefix[idx:] if idx >= 0 else prefix
    target = f"{anchor}.{leaf}"
    return any(n == target or n.endswith("." + target) for n in loaded_params)


class DeepseekV4FlashMLAFusedAttention(DeepseekV4FlashMLAAttention):
    uses_fused_kernel_layouts: ClassVar[bool] = True

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        if self.n_local_heads // self.n_local_groups != WV_GROUP_SIZE:
            raise ValueError("fused attention needs 8 heads per wo_a group")
        if self.kv_cache_dtype not in _FUSED_KV_LAYOUTS:
            raise NotImplementedError(
                f"fused attention does not support kv-cache dtype {self.kv_cache_dtype}"
            )
        self.n_wv_group = self.padded_heads // WV_GROUP_SIZE
        self.fused_decode_min_tokens = (
            vllm_config.attention_config.dsv4_fused_decode_min_tokens
        )
        self._permuted_wq_b = False
        self._permuted_wo_a = False

    # ---- weights -------------------------------------------------------

    def finalize_loaded_weights(self, loaded_params: set[str] | None) -> None:
        if _weight_was_loaded(self.prefix, loaded_params, "wq_b.weight"):
            permute_wq_b_(
                self.wq_b.weight.data, self.wq_b.weight_scale.data, self.n_local_heads
            )
            self._permuted_wq_b = True
        if _weight_was_loaded(self.prefix, loaded_params, "wo_a.weight"):
            permute_wo_a_(
                self.wo_a.weight.data,
                self.wo_a.weight_scale.data,
                self.n_local_heads // self.n_local_groups,
            )
            self._permuted_wo_a = True

    # ---- forward plumbing ----------------------------------------------

    def _alloc_attn_out(
        self, num_tokens: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Post-``wo_a`` activation ``z``; ``wo_b`` consumes it in the graph."""
        return torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    def _finish_o_proj(
        self, attn_out: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.wo_b(attn_out.flatten(1))

    def _prepare_q_and_insert_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        # q is [N, n_local_heads, 512] in the fused layout (permuted wq_b); the
        # padding to the kernel head count happens per path in forward_mqa.
        if isinstance(attn_metadata, dict):
            swa_metadata = cast(
                "DeepseekSparseSWAMetadata",
                attn_metadata[self.swa_cache_layer.prefix],
            )
            rope_quant_insert(
                kv,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.swa_cache_layer.kv_cache,
                swa_metadata.slot_mapping,
                compress_ratio=1,
            )
        return q

    def _wo_a_einsum(
        self, out_fp8: torch.Tensor, out_sf: torch.Tensor, z: torch.Tensor
    ) -> None:
        groups = self.n_local_groups
        fp8_einsum(
            "bhr,hdr->bhd",
            (out_fp8[:, :groups], out_sf[:, :groups]),
            (self.wo_a.weight, self.wo_a.weight_scale),
            z,
            recipe=self._einsum_recipe,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if not (self._permuted_wq_b and self._permuted_wo_a):
            raise RuntimeError(
                f"{self.prefix}: wq_b / wo_a were not permuted for the FlashMLA "
                "fused kernel (finalize_loaded_weights did not see this layer's "
                "weights); refusing to run with mismatched layouts."
            )
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: reserve the prefill workspace, compile the Q
            # padding kernel, produce zeros.
            self._reserve_dummy_run_workspace(q)
            dsv41_q_layout(q, self.padded_heads, "fused")
            output.zero_()
            return
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata",
            attn_metadata[self.swa_cache_layer.prefix],
        )
        assert swa_metadata.positions_int32 is not None
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill_fused(
                q[num_decode_tokens:],
                swa_metadata.positions_int32[num_decode_tokens:],
                flashmla_metadata,
                swa_metadata,
                output[num_decode_tokens:],
            )
        if swa_metadata.num_decodes > 0:
            if num_decode_tokens >= self.fused_decode_min_tokens:
                self._forward_decode_fused(
                    q[:num_decode_tokens],
                    swa_metadata.positions_int32[:num_decode_tokens],
                    flashmla_metadata,
                    swa_metadata,
                    output[:num_decode_tokens],
                )
            else:
                self._forward_decode_split_kv(
                    q[:num_decode_tokens],
                    positions[:num_decode_tokens],
                    flashmla_metadata,
                    swa_metadata,
                    output[:num_decode_tokens],
                )

    def _reserve_dummy_run_workspace(self, q: torch.Tensor) -> None:
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
        """Compressed cache and its ``[n, topk]`` slot indices / lengths."""
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

    def _forward_decode_fused(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        z: torch.Tensor,
    ) -> None:
        extra_cache, extra_idx, extra_len = self._decode_extra(
            flashmla_metadata, swa_metadata
        )
        assert swa_metadata.decode_swa_indices is not None
        out_fp8, out_sf, _ = flash_mla_fused_sparse_decode(
            dsv41_q_layout(q, self.padded_heads, "fused"),
            self.swa_cache_layer.kv_cache.unsqueeze(-2),
            swa_metadata.decode_swa_indices.view(q.shape[0], -1),
            self.scale,
            positions_int32,
            self.rotary_emb.cos_sin_cache,
            self.n_wv_group,
            attn_sink=self.attn_sink,
            topk_length=swa_metadata.decode_swa_lens,
            extra_k_cache=extra_cache,
            extra_indices=extra_idx,
            extra_topk_length=extra_len,
        )
        self._wo_a_einsum(out_fp8, out_sf, z)

    def _forward_decode_split_kv(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        z: torch.Tensor,
    ) -> None:
        extra_cache, extra_idx, extra_len = self._decode_extra(
            flashmla_metadata, swa_metadata
        )
        num_tokens = q.shape[0]
        q_std = dsv41_q_layout(
            q,
            self.padded_heads,
            "standard_rope",
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
        )
        tile_metadata = {
            0: swa_metadata.tile_sched_swaonly,
            1: swa_metadata.tile_sched_c1a,
            2: swa_metadata.tile_sched_c2a,
        }[self.compress_ratio]
        assert tile_metadata is not None
        out, _ = flash_mla_with_kvcache(
            q=q_std.unsqueeze(1),
            k_cache=self.swa_cache_layer.kv_cache.unsqueeze(-2),
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_metadata.decode_swa_indices,
            topk_length=swa_metadata.decode_swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None
            if extra_idx is None
            else extra_idx.view(num_tokens, 1, -1),
            extra_topk_length=extra_len,
        )
        o_fp8, o_sf = fused_inv_rope_fp8_quant(
            out.squeeze(1)[:, : self.n_local_heads],
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=WV_GROUP_SIZE,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            quant_group_size=32,
            tma_aligned_scales=True,
            permuted_output=True,
        )
        self._wo_a_einsum(o_fp8, o_sf, z)

    def _forward_prefill_fused(
        self,
        q: torch.Tensor,
        positions_int32: torch.Tensor,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        z: torch.Tensor,
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
        q_pad = dsv41_q_layout(q, self.padded_heads, "fused")
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        assert chunk_plan, "prefill chunk plan must be non-empty"
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
            qs = query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            qe = query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[qs:qe],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
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
            out_fp8, out_sf, _, _ = flash_mla_fused_sparse_prefill(
                q_pad[qs:qe],
                kv_ws.view(-1, 1, q.shape[-1]),
                combined_indices.unsqueeze(1),
                self.scale,
                positions_int32[qs:qe],
                self.rotary_emb.cos_sin_cache,
                self.n_wv_group,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            self._wo_a_einsum(out_fp8, out_sf, z[qs:qe])
