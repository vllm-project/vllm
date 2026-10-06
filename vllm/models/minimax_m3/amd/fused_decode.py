# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 decode with each sparse MoE layer fused into one AITER kernel.

The kernel all-reduces inside the launch, so whether a step takes this path must
be decided identically on every TP rank. It depends only on the step's token
count and attention metadata, which every rank builds from the same scheduler
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from aiter.ops.flydsl.minimax_m3_mono import (
        MiniMaxM3MonoDecode,
        MonoLayerCaches,
        MonoLayerWeights,
    )

    from vllm.models.minimax_m3.amd.model import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3Model,
    )

logger = init_logger(__name__)

_TP = 4
_MAX_CONTEXT = 1 << 20


def fused_decode_unsupported_reason(vllm_config: VllmConfig) -> str | None:
    """Why this deployment cannot run the fused decode path, or None."""
    if not current_platform.is_rocm():
        return "requires ROCm"
    from vllm.platforms.rocm import on_gfx950

    pc = vllm_config.parallel_config
    cc = vllm_config.cache_config
    text_config = vllm_config.model_config.hf_text_config
    indexer_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("bf16")
    checks = (
        (on_gfx950(), "requires gfx950"),
        (pc.tensor_parallel_size == _TP, f"requires tensor parallel size {_TP}"),
        (
            pc.pipeline_parallel_size == 1 and pc.data_parallel_size == 1,
            "does not support pipeline or data parallelism",
        ),
        (not pc.enable_expert_parallel, "does not support expert parallelism"),
        (cc.cache_dtype in ("fp8", "fp8_e4m3"), "requires --kv-cache-dtype fp8"),
        (cc.block_size == 128, "requires --block-size 128"),
        (indexer_dtype == "fp8", 'requires indexer_kv_dtype "fp8"'),
        (
            vllm_config.model_config.max_model_len <= _MAX_CONTEXT,
            f"requires --max-model-len <= {_MAX_CONTEXT}",
        ),
        (
            not getattr(text_config, "use_index_cache", False),
            "does not support use_index_cache",
        ),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    return None


class MiniMaxM3FusedDecode:
    """Routes eligible decode steps of a ``MiniMaxM3Model`` to the fused kernel.

    A step is eligible when it is a pure decode (speculative verify included) of
    at most ``MAX_TOKENS`` tokens; prefill, mixed and larger steps take the
    regular layers. The kernel is bound on the first eligible step, which runs
    eagerly before CUDA graph capture, and rebound whenever the KV cache moves.
    """

    def __init__(self, model: MiniMaxM3Model, vllm_config: VllmConfig) -> None:
        reason = fused_decode_unsupported_reason(vllm_config)
        if reason is not None:
            raise ValueError(f"minimax_m3_fused_decode {reason}.")
        from aiter.ops.flydsl.minimax_m3_mono import MAX_TOKENS

        self._model = model
        self._max_tokens = MAX_TOKENS
        self._op: MiniMaxM3MonoDecode | None = None
        self._bound_cache_ptr = 0
        layers = list(model.layers)
        self._num_dense = next(
            i for i, layer in enumerate(layers) if layer.is_moe_layer
        )
        if any(
            not layer.is_moe_layer or not hasattr(layer.self_attn, "indexer")
            for layer in layers[self._num_dense :]
        ):
            raise ValueError(
                "minimax_m3_fused_decode requires every layer after the leading "
                "dense ones to be a sparse-attention MoE layer."
            )
        self._layer_name = layers[self._num_dense].self_attn.layer_name

    def eligible(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ) -> bool:
        if input_ids is None or inputs_embeds is not None:
            return False
        if torch.compiler.is_compiling():
            return False
        n = input_ids.numel()
        if not 1 <= n <= self._max_tokens:
            return False
        all_md = get_forward_context().attn_metadata
        if not isinstance(all_md, dict):
            return False
        md = all_md.get(self._layer_name)
        if not isinstance(md, MiniMaxM3SparseMetadata):
            return False
        if md.num_prefills or md.decode is None or md.num_decode_tokens != n:
            return False
        if self._bound_cache_ptr != self._cache_ptr():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "minimax_m3_fused_decode was first reached inside CUDA graph "
                    "capture, before an eager step on the current KV cache."
                )
            self._bind()
        return True

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert self._op is not None
        model = self._model
        n = input_ids.numel()
        hidden = model.embed_input_ids(input_ids)
        residual = None
        aux = model._maybe_add_hidden_state([], 0, hidden, residual)
        for idx, layer in enumerate(model.layers[: self._num_dense]):
            hidden, residual = layer(positions, hidden, residual)
            model._maybe_add_hidden_state(aux, idx + 1, hidden, residual)
        # EAGLE3 tap k is the residual stream after layer k - 1.
        sparse_taps = tuple(
            k - 1 for k in model.aux_hidden_state_layers if k > self._num_dense
        )

        ctx = get_forward_context()
        assert isinstance(ctx.attn_metadata, dict)
        md = ctx.attn_metadata[self._layer_name]
        assert isinstance(md, MiniMaxM3SparseMetadata) and md.decode is not None
        decode = md.decode
        q = decode.decode_query_len
        block_table, seq_lens = decode.block_table, decode.seq_lens
        if q > 1:
            # One row per token: token j of a request sees seq_len - (q - 1 - j).
            reqs = n // q
            back = torch.arange(
                q - 1, -1, -1, dtype=seq_lens.dtype, device=seq_lens.device
            )
            block_table = block_table[:reqs].repeat_interleave(q, dim=0)
            seq_lens = (seq_lens[:reqs, None] - back).reshape(-1)
        hidden, residual, sparse_aux = self._op.run(
            hidden,
            residual,
            positions,
            ctx.slot_mapping[self._layer_name][:n],
            block_table,
            seq_lens,
            [_layer_caches(layer) for layer in model.layers[self._num_dense :]],
            query_len=q,
            aux_layers=sparse_taps,
        )
        hidden, _ = model.norm(hidden, residual)
        aux += sparse_aux
        return (hidden, aux) if aux else hidden

    def _cache_ptr(self) -> int:
        return self._model.layers[self._num_dense].self_attn.kv_cache.data_ptr()

    def _bind(self) -> None:
        sparse = list(self._model.layers[self._num_dense :])
        if self._op is None:
            self._op = self._create_op(sparse[0])
        cos_sin = sparse[0].self_attn.rotary_emb.cos_sin_cache.to(torch.bfloat16)
        self._op.register_layers(
            [(layer.layer_id, _layer_weights(layer, cos_sin)) for layer in sparse],
            block_pages=sparse[0].self_attn.get_aiter_sparse_pa_block_page_stride(),
        )
        self._bound_cache_ptr = self._cache_ptr()
        logger.info_once(
            "MiniMax M3 fused decode enabled for %d sparse layers, steps of up to "
            "%d tokens.",
            len(sparse),
            self._max_tokens,
        )

    def _create_op(self, layer: MiniMaxM3DecoderLayer) -> MiniMaxM3MonoDecode:
        from aiter.ops.flydsl.minimax_m3_mono import MiniMaxM3MonoDecode

        config = self._model.config
        attn = layer.self_attn
        moe = layer.block_sparse_moe
        sparse_cfg = config.sparse_attention_config

        meta = rocm_aiter_moe.aiter_topK_meta_data
        if meta is None or not moe.use_aiter_moe_fse:
            raise ValueError(
                "minimax_m3_fused_decode requires the AITER fused shared expert: "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 and --moe-backend aiter."
            )
        topk_weights, topk_ids = meta
        top_k = config.num_experts_per_tok
        if int(topk_ids[0, top_k]) != config.num_local_experts:
            raise ValueError(
                "minimax_m3_fused_decode expects the fused shared expert in the "
                "last expert slot."
            )

        return MiniMaxM3MonoDecode(
            get_tp_group().cpu_group,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
            attn.kv_cache.device,
            sm_scale=attn.scaling,
            eps=config.rms_norm_eps,
            route_scale=float(moe.routed_scaling_factor),
            shared_weight=float(topk_weights[0, top_k]),
            swiglu_limit=float(config.swiglu_limit),
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            scalar_kv_scale=True,
            gate_fp32=moe.gate.weight.dtype == torch.float32,
        )


def _layer_caches(layer: MiniMaxM3DecoderLayer) -> MonoLayerCaches:
    from aiter.ops.flydsl.minimax_m3_mono import MonoLayerCaches

    attn = layer.self_attn
    k_cache, v_cache = attn.get_aiter_sparse_pa_kv_cache()
    k_scale, v_scale = attn.get_kv_scales()
    return MonoLayerCaches(
        k_cache, v_cache, k_scale, v_scale, attn.indexer.index_cache.kv_cache
    )


def _layer_weights(
    layer: MiniMaxM3DecoderLayer, cos_sin: torch.Tensor
) -> MonoLayerWeights:
    from aiter.ops.flydsl.minimax_m3_mono import MonoLayerWeights

    attn = layer.self_attn
    moe = layer.block_sparse_moe
    experts = moe.experts.routed_experts
    for linear in (attn.qkv_proj, attn.o_proj):
        if not isinstance(linear.quant_method, Fp8PtpcOnlineLinearMethod):
            raise ValueError(
                "minimax_m3_fused_decode requires FP8 per-channel attention "
                f"projections, but {linear.prefix} uses "
                f"{type(linear.quant_method).__name__}. Drop the custom "
                "--quantization-config or quantize these projections with "
                "fp8_per_channel."
            )
    return MonoLayerWeights(
        input_norm=layer.input_layernorm.weight,
        w_qkv=attn.qkv_proj.weight,
        s_qkv=attn.qkv_proj.weight_scale,
        q_norm=attn.q_norm.weight,
        k_norm=attn.k_norm.weight,
        index_q_norm=attn.index_q_norm.weight,
        index_k_norm=attn.index_k_norm.weight,
        cos_sin=cos_sin,
        w_o=attn.o_proj.weight,
        s_o=attn.o_proj.weight_scale,
        post_norm=layer.post_attention_layernorm.weight,
        gate=moe.gate.weight,
        gate_bias=moe.e_score_correction_bias,
        w13=experts.w13_weight,
        s13=experts.w13_weight_scale,
        w2=experts.w2_weight,
        s2=experts.w2_weight_scale,
        **_layer_caches(layer)._asdict(),
    )
