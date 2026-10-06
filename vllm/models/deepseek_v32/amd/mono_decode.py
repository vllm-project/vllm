# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 decode with each sparse MoE layer fused into one AITER kernel.

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
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8BlockParams,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    fit_kpool_indices_to_aiter,
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from aiter.ops.flydsl.glm5_mono import Glm5MonoKernel, LayerWeights

    from vllm.models.deepseek_v32.amd.model import (
        DeepseekV32DecoderLayer,
        DeepseekV32Model,
    )

logger = init_logger(__name__)

_TP = 4
_SPARSE_TOPK = 2048
# Decode steps the fused path takes; each needs a kernel chunk of two or more
# rows, and a single row is launched as two.
_STEP_TOKENS = (1, 2, 4, 8)


def mono_decode_unsupported_reason(vllm_config: VllmConfig) -> str | None:
    """Why this deployment cannot run the fused decode path, or None."""
    if not current_platform.is_rocm():
        return "requires ROCm"
    from vllm.platforms.rocm import on_gfx950

    pc = vllm_config.parallel_config
    checks = (
        (on_gfx950(), "requires gfx950"),
        (pc.tensor_parallel_size == _TP, f"requires tensor parallel size {_TP}"),
        (
            pc.pipeline_parallel_size == 1 and pc.data_parallel_size == 1,
            "does not support pipeline or data parallelism",
        ),
        (not pc.enable_expert_parallel, "does not support expert parallelism"),
        (
            vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3"),
            "requires --kv-cache-dtype fp8",
        ),
        (
            vllm_config.speculative_config is None,
            "does not support speculative decoding",
        ),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    return None


class GlmMonoDecode:
    """Routes eligible decode steps of a ``DeepseekV32Model`` to the fused kernel.

    A step is eligible when it is a pure decode of 1, 2, 4 or 8 tokens; prefill,
    mixed and other steps take the regular layers. The leading dense layers
    always run unfused. Weights are bound on the first eligible step, which runs
    eagerly before CUDA graph capture.
    """

    def __init__(self, model: DeepseekV32Model, vllm_config: VllmConfig) -> None:
        reason = mono_decode_unsupported_reason(vllm_config)
        if reason is not None:
            raise ValueError(f"VLLM_ROCM_GLM_MONO_DECODE {reason}.")
        if not model.is_fused_shared_expert_enabled:
            raise ValueError(
                "VLLM_ROCM_GLM_MONO_DECODE requires "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1, which stores the "
                "shared expert in the last expert slot."
            )
        self._model = model
        layers = list(model.layers)
        self._num_dense = next(
            i for i, layer in enumerate(layers) if isinstance(layer.mlp, DeepseekV2MoE)
        )
        if not all(
            isinstance(layer.mlp, DeepseekV2MoE) for layer in layers[self._num_dense :]
        ):
            raise ValueError(
                "VLLM_ROCM_GLM_MONO_DECODE requires every layer after the leading "
                "dense ones to be a MoE layer."
            )
        self._layer_name = layers[self._num_dense].self_attn.layer_name
        self._ops: dict[int, list[Glm5MonoKernel]] = {}
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cur_pos: torch.Tensor | None = None

    def eligible(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ) -> bool:
        if input_ids is None or inputs_embeds is not None:
            return False
        if torch.compiler.is_compiling():
            return False
        if input_ids.numel() not in _STEP_TOKENS:
            return False
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict) or self._layer_name not in md:
            return False
        if md[self._layer_name].max_query_len != 1:
            return False
        if not self._ops:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "VLLM_ROCM_GLM_MONO_DECODE was first reached inside CUDA graph "
                    "capture, before an eager decode step."
                )
            self._bind()
        return True

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        model = self._model
        n = input_ids.numel()
        hidden = model.embed_input_ids(input_ids)
        residual = None
        for layer in model.layers[: self._num_dense]:
            hidden, residual = layer(positions, hidden, residual)
        state = tensor_model_parallel_all_reduce(hidden) + residual

        ctx = get_forward_context()
        md = ctx.attn_metadata[self._layer_name]
        slots = ctx.slot_mapping[self._layer_name][:n]
        indptr = md.paged_kv_indptr[: n + 1]
        kernel_positions = positions
        if n == 1:
            state = torch.cat([state, torch.zeros_like(state)])
            kernel_positions = torch.cat([positions, positions])
            slots = torch.cat([slots, torch.full_like(slots, -1)])
            indptr = torch.cat([indptr, indptr[1:]])

        fp8 = current_platform.fp8_dtype()
        ops = self._ops[max(n, 2)]
        for layer, op in zip(model.layers[self._num_dense :], ops):
            attn = layer.self_attn
            if attn.indexer is not None and not attn.skip_topk:
                self._refresh_indices(layer, state[:n], positions, md)
            cache = attn.kv_cache.view(fp8).view(-1, attn.kv_cache.shape[-1])
            state = op.forward(
                state,
                self._cur_pos,
                cache,
                cache,
                md.paged_kv_indices,
                self._cos,
                self._sin,
                positions=kernel_positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
            )
        return model.norm(state[:n])

    @staticmethod
    def _refresh_indices(
        layer: DeepseekV32DecoderLayer,
        state: torch.Tensor,
        positions: torch.Tensor,
        md,
    ) -> None:
        attn = layer.self_attn
        attn.refresh_sparse_indices(positions, layer.input_layernorm(state))
        tokens = md.num_actual_tokens
        triton_convert_req_index_to_global_index(
            md.req_id_per_token,
            md.block_table,
            fit_kpool_indices_to_aiter(
                attn.topk_indices_buffer[:tokens], md.topk_tokens
            ),
            md.paged_kv_indptr,
            md.paged_kv_indices,
            BLOCK_SIZE=md.block_size,
            NUM_TOPK_TOKENS=md.topk_tokens,
        )

    def _bind(self) -> None:
        from aiter.ops.flydsl.glm5_mono import (
            AttentionWeight,
            Glm5MonoKernel,
            KvCacheLayout,
            glm5_tp_config,
            prepare_glm5_weights,
        )

        cfg = glm5_tp_config(_TP)
        rank = get_tensor_model_parallel_rank()
        group = get_tp_group().cpu_group
        sparse = list(self._model.layers[self._num_dense :])
        weights = [_layer_weights(layer, cfg, rank) for layer in sparse]
        prepared = [
            prepare_glm5_weights(w, AttentionWeight.FP8_BLOCK128) for w in weights
        ]
        for chunk in sorted({max(n, 2) for n in _STEP_TOKENS}):
            runtime = None
            ops = []
            for w, p in zip(weights, prepared):
                op = Glm5MonoKernel(
                    w,
                    chunk,
                    rank=rank,
                    npes=_TP,
                    group=group,
                    topk=_SPARSE_TOPK,
                    launches_per_step=1,
                    with_indexer=False,
                    attention_weight=AttentionWeight.FP8_BLOCK128,
                    kv_cache_layout=KvCacheLayout.ATOM,
                    kv_cache_dtype="fp8",
                    prepared_weights=p,
                    runtime=runtime,
                    native_fp4_mfma=True,
                )
                runtime = runtime or op
                ops.append(op)
            self._ops[chunk] = ops
        # Drop the unpacked attention copies; every op reads the packed ones.
        for w, p in zip(weights, prepared):
            for name in ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o"):
                w.t[name] = p[name]

        attn = sparse[0].self_attn
        cos_sin = attn.rotary_emb.cos_sin_cache
        half = cos_sin.shape[-1] // 2
        self._cos = cos_sin[:, :half].to(torch.bfloat16).contiguous()
        self._sin = cos_sin[:, half:].to(torch.bfloat16).contiguous()
        self._cur_pos = torch.zeros(1, dtype=torch.int32, device=cos_sin.device)
        logger.info_once(
            "GLM fused decode enabled for %d MoE layers, decode steps of %s tokens.",
            len(sparse),
            "/".join(map(str, _STEP_TOKENS)),
        )


def split_kv_b(
    weight: torch.Tensor,
    scale: torch.Tensor,
    heads: int,
    nope_dim: int,
    v_dim: int,
    block: int = 128,
    uk_block_k: int = 64,
    uv_block_rows: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut a block-FP8 ``kv_b_proj`` into the kernel's W_UK and W_UV.

    W_UK is ``[heads * kv_lora, nope_dim]`` (each head's nope rows transposed)
    with ``block x uk_block_k`` scales; W_UV is ``[heads * v_dim, kv_lora]`` with
    ``uv_block_rows x block`` scales. Each new scale block lies inside one
    checkpoint block, so the values are exact.
    """
    rows_per_head = nope_dim + v_dim
    kv_lora = weight.shape[1]
    w = weight.view(heads, rows_per_head, kv_lora)
    w_uk = w[:, :nope_dim].transpose(1, 2).reshape(heads * kv_lora, nope_dim)
    w_uv = w[:, nope_dim:].reshape(heads * v_dim, kv_lora)
    head = torch.arange(heads, device=scale.device)[:, None] * rows_per_head
    uk_rows = (
        head + torch.arange(0, nope_dim, uk_block_k, device=scale.device)
    ) // block
    uv_rows = (
        head + nope_dim + torch.arange(0, v_dim, uv_block_rows, device=scale.device)
    ) // block
    s_uk = scale[uk_rows].permute(0, 2, 1).reshape(heads * kv_lora // block, -1)
    s_uv = scale[uv_rows].reshape(heads * v_dim // uv_block_rows, -1)
    return (
        w_uk.contiguous(),
        s_uk.contiguous(),
        w_uv.contiguous(),
        s_uv.contiguous(),
    )


def _weight_preshuffled(linear: torch.nn.Module) -> bool:
    return any(
        getattr(getattr(method, "fp8_linear", None), "preshuffles_weight", False)
        for method in (
            getattr(linear, "quant_method", None),
            getattr(linear, "scheme", None),
        )
    )


def _block_fp8(linear: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.glm5_mono import unshuffle_linear_weight

    params = FP8BlockParams.from_layer(linear)
    scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    weight = params.weight
    if (
        weight.dtype != current_platform.fp8_dtype()
        or scale is None
        or scale.dtype != torch.float32
        or scale.dim() != 2
    ):
        raise ValueError(
            "VLLM_ROCM_GLM_MONO_DECODE requires block-FP8 attention projections, "
            f"but {getattr(linear, 'prefix', linear)} is {weight.dtype}."
        )
    if _weight_preshuffled(linear):
        weight = unshuffle_linear_weight(weight)
    return weight.contiguous(), scale.contiguous()


def _fp4_storage(tensor: torch.Tensor) -> torch.Tensor:
    view = tensor.view(torch.uint8)
    view.is_shuffled = True
    return view


def _layer_weights(layer: DeepseekV32DecoderLayer, cfg, rank: int) -> LayerWeights:
    from aiter.ops.flydsl.glm5_mono import (
        LayerWeights,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )

    attn = layer.self_attn
    moe = layer.mlp
    experts = moe.experts.routed_experts
    physical = cfg.n_experts + cfg.num_shared_experts
    if experts.w13_weight.shape[0] != physical:
        raise ValueError(
            f"VLLM_ROCM_GLM_MONO_DECODE expects {physical} physical experts, "
            f"got {experts.w13_weight.shape[0]}."
        )
    w_qkv_a, s_qkv_a = _block_fp8(attn.fused_qkv_a_proj)
    w_q_b, s_q_b = _block_fp8(attn.q_b_proj)
    w_o, s_o = _block_fp8(attn.o_proj)
    w_uk, s_uk, w_uv, s_uv = split_kv_b(
        *_block_fp8(attn.kv_b_proj), cfg.local_heads, cfg.nope_dim, cfg.v_dim
    )
    gate = moe.gate
    tensors = {
        "g_in": layer.input_layernorm.weight,
        "g_q": attn.q_a_layernorm.weight,
        "g_kv": attn.kv_a_layernorm.weight,
        "g_post": layer.post_attention_layernorm.weight,
        "w_qkv_a": w_qkv_a,
        "s_qkv_a": s_qkv_a,
        "w_q_b": w_q_b,
        "s_q_b": s_q_b,
        "w_uk": w_uk,
        "s_uk": s_uk,
        "w_uv": w_uv,
        "s_uv": s_uv,
        "w_o": w_o,
        "s_o": s_o,
        "w_r": gate.weight.to(torch.bfloat16).contiguous(),
        "bias": gate.e_score_correction_bias.float().contiguous(),
        "w_ug": _fp4_storage(experts.w13_weight),
        "s_ug": experts.w13_weight_scale.view(torch.uint8),
        "w_dn": _fp4_storage(experts.w2_weight),
        "s_dn": experts.w2_weight_scale.view(torch.uint8),
    }
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        _TP,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical,
    )
