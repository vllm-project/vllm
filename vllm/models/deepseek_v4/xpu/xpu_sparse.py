# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU DeepSeek-V4 attention subclass.

Subclasses the shared ``DeepseekV4Attention`` ABC and provides XPU-native
Triton kernels for decode (FP8 dequant + BF16 attention) and prefill
(BF16 gathered KV + sparse attention).
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (
    xpu_sparse_decode_fp8,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4XPUSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_V4_MLA_SPARSE"


class DeepseekV4XPUAttention(DeepseekV4Attention):
    """XPU sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4XPUSparseBackend
    use_flashmla_fp8_layout = True

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event() raises RuntimeError on XPU ("dummy base class").
        # The Base and DeepseekV4Indexer both create cuda Events in __init__, so
        # we temporarily redirect torch.cuda.Event → torch.xpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = _orig_event  # type: ignore[misc]

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        from typing import cast

        if not isinstance(attn_metadata, dict):
            # Profile run: no-op, just return q (no padding needed on XPU).
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
            xpu_qnorm_rope_kv_fp8_insert,
        )

        xpu_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        return q

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )

        wo_a_raw_weight = cast(torch.Tensor, self.wo_a.weight)

        # BF16 fallback path: wo_a has been dequantized from MXFP4/8 to BF16.
        # Do inverse RoPE + group concat in BF16, then matmul with BF16 weight.
        if wo_a_raw_weight.dtype == torch.bfloat16:
            num_tokens, num_heads, head_dim = o.shape
            nope_dim = self.nope_head_dim
            rope_dim = self.rope_head_dim
            heads_per_group = num_heads // self.n_local_groups

            # Inverse RoPE on the rope portion
            cos_sin = self.rotary_emb.cos_sin_cache[positions]  # [T, rope_dim]
            cos = cos_sin[:, : rope_dim // 2]
            sin = cos_sin[:, rope_dim // 2 :]
            o_nope = o[..., :nope_dim]  # [T, H, nope_dim]
            o_rope = o[..., nope_dim:]  # [T, H, rope_dim]
            # Inverse rotate: rotate by -theta => cos same, sin negated
            o_rope_r = o_rope.reshape(num_tokens, num_heads, rope_dim // 2, 2)
            x0 = o_rope_r[..., 0]  # [T, H, rope_dim//2]
            x1 = o_rope_r[..., 1]
            cos_u = cos.unsqueeze(1)  # [T, 1, rope_dim//2]
            sin_u = sin.unsqueeze(1)
            # Inverse of (x0*cos - x1*sin, x0*sin + x1*cos) is
            # (x0*cos + x1*sin, -x0*sin + x1*cos)
            y0 = x0 * cos_u + x1 * sin_u
            y1 = -x0 * sin_u + x1 * cos_u
            o_rope_inv = torch.stack([y0, y1], dim=-1).reshape(
                num_tokens, num_heads, rope_dim
            )
            o_full = torch.cat([o_nope, o_rope_inv], dim=-1)  # [T, H, D]

            # Group concat: [T, H, D] → [T, G, heads_per_group*D]
            D = heads_per_group * head_dim
            o_grouped = o_full.reshape(num_tokens, self.n_local_groups, D)  # [T, G, D]

            wo_a_weight = torch.reshape(
                wo_a_raw_weight, (self.n_local_groups, self.o_lora_rank, D)
            ).transpose(1, 2)  # [G, D, R]

            # Batched matmul: [G, T, D] x [G, D, R] -> [G, T, R]
            z = torch.bmm(o_grouped.transpose(0, 1), wo_a_weight)  # [G, T, R]
            return self.wo_b(z.transpose(0, 1).flatten(1))

        # FP8 path (original block-FP8 or e8m0 scales)
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=False,
        )

        # Precomputed contiguous [G, K, N] weight and [G, K/bs, N/bs] scale.
        wo_a_weight = self.wo_a.bmm_weight
        wo_a_scale = self.wo_a.bmm_scale

        # TODO: optimize fused_inv_rope_fp8_quant for xpu bmm to
        # eliminate o_scale transpose + contiguous
        z = torch.ops.vllm.xpu_fp8_bmm(
            o_fp8.transpose(0, 1),
            wo_a_weight,
            torch.bfloat16,
            o_scale.transpose(0, 1).contiguous(),
            wo_a_scale,
            None,
        )

        return self.wo_b(z.transpose(0, 1).flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve workspace, skip actual kernels.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
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
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert swa_indices is not None and swa_lens is not None
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
            )
            output[query_start:query_end] = out


# The following is a one-time materialization pass to dequantize the MXFP4 / MXFP8
# weights of ``wo_a`` consumed via raw ``torch.mm`` / 3D einsum paths to bf16
# in-place.
# TODO: remove after the relevant kernels support MXFP4 / MXFP8 directly.
def _materialize_mxfp_wo_a_bf16(model: torch.nn.Module) -> None:
    """Dequantize MX-quantized weights of wo_a consumed via raw ``torch.mm`` /
    3D einsum paths to bf16 in-place.

    DeepSeek V4's attention path bypasses the CT linear method for two
    weights:

    * ``mla_attn.wo_a`` -- a per-group projection that the FP8 path
      evaluates as a 3D einsum (``tgd,grd->tgr``); the reference
      fallback views ``wo_a.weight`` directly.

    Neither path understands MX-quantized layouts (MXFP4 packed
    ``weight_packed`` + ``weight_scale`` uint8 E8M0; MXFP8 group=32
    ``weight`` fp8_e4m3fn + ``weight_scale`` uint8 E8M0). For
    correctness we materialize a bf16 ``weight`` once after loading
    and swap the quant method to ``UnquantizedLinearMethod`` so the
    model loader's ``process_weights_after_loading`` doesn't try to
    repack the already-removed packed buffers.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        dequant_mxfp4_to_bf16,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    def _dequant_to_bf16(layer: torch.nn.Module) -> torch.Tensor | None:
        # MXFP4 packed: weight_packed (M, K/2) uint8 + weight_scale
        #   (M, K/group) uint8 E8M0 -> bf16 (M, K).
        if hasattr(layer, "weight_packed"):
            return dequant_mxfp4_to_bf16(
                layer.weight_packed.data, layer.weight_scale.data
            )

        # MXFP8 group=32: weight (M, K) fp8_e4m3fn + weight_scale
        #   (M, K/32) uint8 E8M0 -> bf16 (M, K). Detect via fp8 dtype on
        #   ``weight`` plus a uint8 ``weight_scale`` companion.
        weight = getattr(layer, "weight", None)
        weight_scale = getattr(layer, "weight_scale", None)
        if (
            weight is not None
            and weight_scale is not None
            and weight.dtype == torch.float8_e4m3fn
            and weight_scale.dtype == torch.uint8
        ):
            return dequant_mxfp8_to_bf16(weight.data, weight_scale.data).contiguous()

        return None

    # Only materialize `wo_a` modules -- keep other paths untouched.
    for module in model.modules():
        wo_a = getattr(module, "wo_a", None)
        if wo_a is None:
            continue

        layer = wo_a
        # Preserve block-FP8 / special einsum paths and skip already-processed.
        if hasattr(layer, "weight_scale_inv") or getattr(
            layer, "_mxfp4_dequantized", False
        ):
            continue

        bf16 = _dequant_to_bf16(layer)
        if bf16 is None:
            continue

        new_weight = torch.nn.Parameter(bf16, requires_grad=False)
        layer.register_parameter("weight", new_weight)
        for attr in ("weight_packed", "weight_scale"):
            if hasattr(layer, attr):
                delattr(layer, attr)
        layer.quant_method = UnquantizedLinearMethod()
        import contextlib

        with contextlib.suppress(AttributeError):
            delattr(layer, "scheme")
        layer._mxfp4_dequantized = True
