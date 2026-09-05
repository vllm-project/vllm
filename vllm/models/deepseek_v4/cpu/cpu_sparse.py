# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DeepSeek-V4 attention subclass.

``forward_mqa`` resolves SWA/compressed top-k indices to paged-cache slot
ids and calls the fused ``flash_mla_with_kvcache_cpu`` kernel. ``_o_proj``
stays eager -- not on the attention hot path.
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    DeepseekV4Indexer,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v4.cpu.cpu_compressor import DeepseekV4CPUCompressor
from vllm.models.deepseek_v4.cpu.cpu_mla import (
    DeepseekV4CPUIndexerBackend,
    DeepseekV4CPUSparseBackend,
)
from vllm.models.deepseek_v4.cpu.cpu_utils import map_local_to_global_slots_cpu
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_ROPE_DIM = 64


def _deepseek_v4_cpu_prepare_and_attn(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    qr_scale: torch.Tensor | None,
    kv_score: torch.Tensor | None,
    indexer_kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    o_padded: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Opaque custom-op boundary around attention-prep + sparse-indexer +
    MLA attention + output projection -- CPU-only.

    Wraps cache writes with no explicit tensor args (SWA/compressor/indexer
    state), which torch.compile can't otherwise track as mutations -- one
    opaque call sidesteps that. ``_o_proj`` is folded in too because CPU's
    ``DYNAMO_TRACE_ONCE`` mode traces exactly once, during warmup when
    ``attn_metadata`` is ``None``; tracing ``_o_proj`` separately would bake
    that warmup branch into the compiled graph permanently, dead-code-
    eliminating the whole attention computation upstream.
    """
    layer_name_str = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name_str]
    self._prepare_and_attn_fn(
        hidden_states,
        qr,
        kv,
        qr_scale,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        o_padded,
    )
    o = o_padded[:, : self.n_local_heads, :]
    return self._o_proj(o, positions)


def _deepseek_v4_cpu_prepare_and_attn_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    qr_scale: torch.Tensor | None,
    kv_score: torch.Tensor | None,
    indexer_kv_score: torch.Tensor | None,
    indexer_weights: torch.Tensor | None,
    positions: torch.Tensor,
    o_padded: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    # hidden_states' last dim already equals _o_proj's output width, so we
    # don't need to resolve layer_name (an opaque object) here.
    return hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[-1]))


direct_register_custom_op(
    op_name="deepseek_v4_cpu_prepare_and_attn",
    op_func=_deepseek_v4_cpu_prepare_and_attn,
    mutates_args=["o_padded"],
    fake_impl=_deepseek_v4_cpu_prepare_and_attn_fake,
)


def _dequant_linear_weight(layer: torch.nn.Module) -> torch.Tensor:
    """Dequantize a linear layer's weight, including FP8 block scales.

    ``_o_proj`` reads ``wo_a.weight`` directly (not via
    ``quant_method.apply()``), so the per-block FP8 scale must be applied
    here explicitly.
    """
    weight = layer.weight
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(layer, "weight_scale", None)
    if scale is None:
        return weight.float()

    if scale.dtype == torch.float8_e8m0fnu:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        scale = _upcast_e8m0_to_fp32(scale)
    scale = scale.to(torch.float32)

    if scale.numel() == 1:
        return weight.float() * scale

    block_size = getattr(layer, "weight_block_size", None)
    if block_size is None:
        # Per-channel scale: one row-scale per output row.
        return weight.float() * scale.view(-1, 1)

    block_m, block_k = block_size
    m, k = weight.shape
    scale = scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    return weight.float() * scale[:m, :k]


def _fused_indexer_q_rope_quant_cpu(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-only equivalent of ``fused_indexer_q_rope_quant``'s FP8 path
    (CPU never reaches its MXFP4 arm -- Blackwell-only). Returns
    ``(q_fp8, weights_out)``, folding the per-token q scale into
    ``index_weights``. ``positions``/``index_q_cos_sin_cache`` are already
    contiguous int64/fp32, so no conversion is needed here.
    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    index_q_fp8 = torch.empty_like(index_q, dtype=current_platform.fp8_dtype())
    ops.fused_indexer_q_rope_quant_cpu(
        positions,
        index_q.to(torch.float32),
        index_q_cos_sin_cache,
        index_q_fp8,
        index_weights.to(torch.float32),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
    )
    return index_q_fp8, index_weights_out


class _NoOpEvent:
    """Stand-in for ``torch.cuda.Event`` on CPU: since CPU never sets an aux
    stream list, these events are constructed but never actually used."""


class DeepseekV4CPUIndexerCache(DeepseekV4IndexerCache):
    """CPU indexer K-cache descriptor: same fields as the shared base, just
    pointing ``get_attn_backend()`` at ``DeepseekV4CPUIndexerBackend``
    instead of the shared, CUDA/XPU-oriented ``DeepseekV4IndexerBackend``."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4CPUIndexerBackend


class DeepseekV4CPUIndexer(DeepseekV4Indexer):
    """CPU indexer: the C4A short-context fallback runs as eager PyTorch
    instead of Triton. Never constructed directly -- ``__class__`` is
    swapped onto an existing ``DeepseekV4Indexer`` instance post-construction.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        qr_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """CPU override: no aux streams, so wq_b_and_q_quant and the
        compressor run straight-line instead of through
        ``maybe_execute_in_parallel``; the short-context check drops the
        CUDA-only ``is_current_stream_capturing()`` guard (always False
        here)."""
        compressor = self.compressor

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                # Fewer candidates than topk: all are selected, but the K
                # cache still needs to be built.
                compressor(compressed_kv_score, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    self._fill_short_context_topk(positions, num_tokens)
                return None, None, None

        def wq_b_and_q_quant():
            q = self._wq_b_proj(qr, qr_scale)
            q = q.view(-1, self.n_head, self.head_dim)
            return _fused_indexer_q_rope_quant_cpu(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
            )

        # Order matches the base class's parallel-join contract: compressor
        # must complete before indexer_op runs (skip_k_cache_insert=True).
        q, weights = wq_b_and_q_quant()
        compressor(compressed_kv_score, positions, rotary_emb)
        # No q_scale: FP8-only path (MXFP4 needs Blackwell).
        return q, None, weights

    def _fill_short_context_topk(
        self, positions: torch.Tensor, num_tokens: int
    ) -> None:
        assert self.topk_indices_buffer is not None
        device = positions.device
        arange = torch.arange(self.topk_tokens, device=device)
        num_compressed = torch.div(
            positions[:num_tokens] + 1, self.compress_ratio, rounding_mode="floor"
        )
        valid = arange.unsqueeze(0) < num_compressed.unsqueeze(-1)
        filled = torch.where(
            valid, arange.unsqueeze(0), torch.full_like(arange.unsqueeze(0), -1)
        )
        self.topk_indices_buffer[:num_tokens, : self.topk_tokens] = filled.to(
            self.topk_indices_buffer.dtype
        )


class DeepseekV4CPUAttention(DeepseekV4Attention):
    """CPU sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4CPUSparseBackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Swap classes post-construction (no class-swap hook on the shared
        # base) so forward()/get_attn_backend() dispatch to CPU kernels.
        # SWA-only layers (compress_ratio <= 1) never build a
        # compressor/indexer.
        if self.compressor is not None:
            self.compressor.__class__ = DeepseekV4CPUCompressor
        if self.indexer is not None:
            self.indexer.__class__ = DeepseekV4CPUIndexer
            self.indexer.k_cache.__class__ = DeepseekV4CPUIndexerCache
            self.indexer.compressor.__class__ = DeepseekV4CPUCompressor

        self._wo_a_packed: torch.Tensor | None = None
        self._wrap_wo_a_process_weights_after_loading()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Cache fp32-contiguous copies of tensors CPU kernels want that way
        but that arrive bf16 (rotary cos/sin table, compressor RMSNorm
        weights) -- cast once here instead of per forward call.

        Runs after every quantized layer's own
        ``process_weights_after_loading`` (see ``is_deferred_attention_layer``);
        ``wo_a``'s packing must happen *before* that phase instead, so it's a
        separate monkeypatch in ``__init__``.
        """
        self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(
            torch.float32
        ).contiguous()
        if self.compressor is not None:
            self.compressor.cache_norm_weight_fp32()
        if self.indexer is not None:
            self.indexer.compressor.cache_norm_weight_fp32()

    def _wrap_wo_a_process_weights_after_loading(self) -> None:
        """Snapshot and pack ``wo_a``'s weight into a bf16 copy for
        ``bmm_cpu`` before the FP8 kernel's own
        ``process_weights_after_loading`` VNNI-repacks it in place for
        row-major reads ``_o_proj`` never does."""
        wo_a = self.wo_a
        quant_method = wo_a.quant_method
        orig_pwal = quant_method.process_weights_after_loading
        heads_per_group = self.n_local_heads // self.n_local_groups
        k_dim = heads_per_group * self.head_dim

        def _capture_and_pack(layer: torch.nn.Module) -> None:
            dequant = _dequant_linear_weight(layer).to(torch.bfloat16)
            dequant = dequant.view(self.n_local_groups, self.o_lora_rank, k_dim)
            self._wo_a_packed = torch.ops._C.convert_weight_packed(dequant.contiguous())
            orig_pwal(layer)
            # orig_pwal VNNI-repacks layer.weight/scale for ordinary-linear
            # use that _o_proj never needs -- free them, same pattern as
            # dispatch_cpu_unquantized_gemm's remove_weight=True.
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            scale_attr = (
                "weight_scale_inv"
                if hasattr(layer, "weight_scale_inv")
                else "weight_scale"
            )
            if hasattr(layer, scale_attr):
                setattr(
                    layer,
                    scale_attr,
                    torch.nn.Parameter(torch.empty(0), requires_grad=False),
                )

        quant_method.process_weights_after_loading = (  # type: ignore[method-assign]
            _capture_and_pack
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CPU override: wraps attention-prep + MLA + output projection in the
        CPU-only opaque custom op (see ``_deepseek_v4_cpu_prepare_and_attn``).
        """
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)

        kv_score: torch.Tensor | None = None
        if self.compressor is not None:
            # Must go through the module's own forward (not a raw matmul on
            # its weight): the packed-GEMM kernel frees the raw weight after
            # packing, same as wo_a's.
            kv_score = self.compressor.fused_wkv_wgate(hidden_states).to(torch.float32)

        indexer_weights: torch.Tensor | None = None
        indexer_kv_score: torch.Tensor | None = None
        if self.indexer is not None:
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)
            indexer_kv_score = self.indexer.compressor.fused_wkv_wgate(
                hidden_states
            ).to(torch.float32)

        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)

        return torch.ops.vllm.deepseek_v4_cpu_prepare_and_attn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
            _encode_layer_name(self.prefix),
        )

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        """CPU override: only the fp8_ds_mla (uint8) SWA cache layout is
        supported here, so the base method's bf16/per-tensor-fp8 branches
        are dropped."""
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # positions is already int64 (the runner's buffer), no cast needed.
        assert positions.dtype == torch.int64
        assert swa_kv_cache.dtype == torch.uint8, (
            "DeepseekV4CPUAttention only supports the fp8_ds_mla cache layout"
        )
        # Fused: Q side does per-head weight-free RMSNorm + GPT-J RoPE
        # (zero-filling padded head slots); KV side does GPT-J RoPE + UE8M0
        # FP8 quant + paged insert.
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        return ops.fused_qnorm_rope_kv_insert_cpu(
            q,
            kv,
            positions,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            self.rotary_emb.cos_sin_cache,
            self.padded_heads,
            self.eps,
            swa_metadata.block_size,
        )

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """CPU override: two ``RMSNorm`` calls instead of the shared
        ``fused_q_kv_rmsnorm``, whose raw ``@triton.jit`` kernel fails to
        link under triton-cpu (``undefined symbol: __truncdfbf2``)."""
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return self.q_norm(qr), None, self.kv_norm(kv)

    def _prepare_and_attn(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        o_padded: torch.Tensor,
    ) -> None:
        """CPU override: no aux streams, so query-projection+KV-insert, the
        indexer, and the compressor run straight-line instead of through
        execute_in_parallel/maybe_execute_in_parallel."""
        attn_metadata = get_forward_context().attn_metadata
        indexer = self.indexer
        compressor = self.compressor

        q = self._wq_b_proj(qr, qr_scale).view(-1, self.n_local_heads, self.head_dim)
        q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        index_q: torch.Tensor | None = None
        index_q_scale: torch.Tensor | None = None
        index_weights_out: torch.Tensor | None = None
        if indexer is not None:
            assert compressor is not None
            index_q, index_q_scale, index_weights_out = indexer(
                hidden_states,
                qr,
                indexer_kv_score,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
                qr_scale,
            )
            compressor(kv_score, positions, self.rotary_emb)
        elif compressor is not None:
            compressor(kv_score, positions, self.rotary_emb)

        self._sparse_indexer_and_attn(
            hidden_states,
            index_q,
            index_q_scale,
            index_weights_out,
            q,
            kv,
            positions,
            o_padded,
        )

    def _sparse_indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """CPU override: identical body, minus ``@eager_break_during_capture``
        (this platform never captures a CUDA graph, so it was a no-op)."""
        if self.indexer is not None and index_q is not None:
            assert index_weights is not None
            # index_q_scale is always None on CPU (FP8-only path).
            assert index_q_scale is None
            self.indexer.indexer_op(hidden_states, index_q, None, index_weights)

        self.forward_mqa(q, kv, positions, out)

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # No head-count padding constraint for the (not yet ported) CPU kernel.
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if get_forward_context().attn_metadata is None:
            # Warmup/profile run: no real metadata, o is a zeroed placeholder.
            return torch.zeros(
                (o.shape[0], self.hidden_size), dtype=o.dtype, device=o.device
            )

        num_tokens, num_heads, _ = o.shape
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        # Undo the query's own GPT-J RoPE rotation: the attention output is
        # a weighted sum of KV rows at different positions, so R(-pos_q)
        # recovers the translation-invariant quantity wo_a/wo_b expect.
        o_derot = ops.inverse_gptj_rope_o_proj_cpu(
            o, positions, cos_sin_cache, _ROPE_DIM
        )

        heads_per_group = num_heads // self.n_local_groups
        o_grouped = o_derot.reshape(
            num_tokens, self.n_local_groups, heads_per_group * self.head_dim
        ).to(torch.bfloat16)
        assert self._wo_a_packed is not None
        z = o_grouped.new_empty((self.n_local_groups, num_tokens, self.o_lora_rank))
        ops.bmm_cpu(
            z,
            o_grouped.transpose(0, 1).contiguous(),
            self._wo_a_packed,
            True,
            None,
        )
        z = z.transpose(0, 1).to(o.dtype)
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup/profile run: reserve nothing extra, skip real kernels.
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        swa_metadata = cast(
            DeepseekSparseSWAMetadata,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        assert swa_metadata.token_to_req_indices is not None
        req_idx = swa_metadata.token_to_req_indices

        swa_only = self.compress_ratio <= 1
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )

        num_tokens = q.shape[0]
        device = q.device

        # ---- SWA window: causal sliding window over the raw per-token cache
        window = self.window_size
        win_offsets = torch.arange(window, device=device)
        start_pos = (positions - window + 1).clamp(min=0)
        local_pos = start_pos.unsqueeze(-1) + win_offsets
        valid_win = local_pos <= positions.unsqueeze(-1)
        win_local = torch.where(valid_win, local_pos, torch.full_like(local_pos, -1))
        window_slots = map_local_to_global_slots_cpu(
            win_local, req_idx, swa_metadata.block_table, swa_metadata.block_size
        )
        window_cache_2d = self.swa_cache_layer.kv_cache.view(
            self.swa_cache_layer.kv_cache.shape[0], -1
        )

        # ---- Compressed KV: C4A's learned top-k, or C128A's dense top-k ----
        if swa_only:
            compressed_slots = window_slots.new_full((num_tokens, 0), -1)
            # Unused when num_compressed == 0 (only dtype/2D-ness matter);
            # reuse the window cache to avoid an allocation.
            compressed_cache_2d = window_cache_2d
            compressed_block_size = 1
        else:
            assert flashmla_metadata is not None
            main_block_size = flashmla_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                local_topk = self.topk_indices_buffer[:num_tokens]
            else:
                width = -(-self.max_model_len // self.compress_ratio)
                num_compressed = torch.div(
                    positions + 1, self.compress_ratio, rounding_mode="floor"
                ).clamp(max=width)
                arange_w = torch.arange(width, device=device)
                local_topk = torch.where(
                    arange_w.unsqueeze(0) < num_compressed.unsqueeze(-1),
                    arange_w.unsqueeze(0).expand(num_tokens, width),
                    torch.full(
                        (num_tokens, width), -1, device=device, dtype=torch.int64
                    ),
                )
            compressed_slots = map_local_to_global_slots_cpu(
                local_topk, req_idx, flashmla_metadata.block_table, main_block_size
            )
            compressed_cache_2d = self.kv_cache.view(self.kv_cache.shape[0], -1)
            compressed_block_size = main_block_size

        torch.ops._C.flash_mla_with_kvcache_cpu(
            output,
            q,
            window_cache_2d,
            window_slots,
            swa_metadata.block_size,
            compressed_cache_2d,
            compressed_slots,
            compressed_block_size,
            self.attn_sink,
            self.scale,
        )
