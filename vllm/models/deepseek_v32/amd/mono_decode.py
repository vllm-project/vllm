# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 decode with each decoder layer fused into one AITER kernel.

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
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8BlockParams,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from aiter.ops.flydsl.glm5_mono import Glm5MonoKernel, LayerWeights

    from vllm.models.deepseek_v32.amd.model import (
        DeepseekV32DecoderLayer,
        DeepseekV32Model,
    )

logger = init_logger(__name__)

_TP_SIZES = (4, 8)
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
        (
            pc.tensor_parallel_size in _TP_SIZES,
            "requires tensor parallel size 4 or 8",
        ),
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
        (
            getattr(
                vllm_config.model_config.hf_config, "indexer_rope_interleave", False
            ),
            "requires indexer_rope_interleave",
        ),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    return None


class GlmMonoDecode:
    """Routes eligible decode steps of a ``DeepseekV32Model`` to the fused kernel.

    A step is eligible when it is a pure decode of 1, 2, 4 or 8 tokens; prefill,
    mixed and other steps take the regular layers. The leading dense layers run in
    the kernel too, their MLP stored as expert-shaped slices. Weights are bound on
    the first eligible step, which runs eagerly before CUDA graph capture.
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
        self._max_model_len = vllm_config.model_config.max_model_len
        self._ops: dict[int, list[Glm5MonoKernel]] = {}
        self._runtimes: dict[int, list[Glm5MonoKernel]] = {}
        self._launch_index: list[int] = []
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
        state = model.embed_input_ids(input_ids)

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
        req_ids = md.req_id_per_token[:n]
        if n == 1:
            req_ids = torch.cat([req_ids, req_ids])
        for layer, op, index in zip(model.layers, ops, self._launch_index):
            attn = layer.self_attn
            cache = attn.kv_cache.view(fp8).view(-1, attn.kv_cache.shape[-1])
            if op.with_indexer:
                state = op.forward(
                    state,
                    self._cur_pos,
                    cache,
                    cache,
                    None,
                    self._cos,
                    self._sin,
                    layer=index,
                    advance=False,
                    positions=kernel_positions,
                    slot_mapping=slots,
                    sparse_kv_indptr=indptr,
                    index_cache=attn.indexer.k_cache.kv_cache,
                    block_table=md.block_table,
                    req_ids=req_ids,
                    out_indices=md.paged_kv_indices,
                )
                continue
            state = op.forward(
                state,
                self._cur_pos,
                cache,
                cache,
                md.paged_kv_indices,
                self._cos,
                self._sin,
                layer=index,
                advance=False,
                positions=kernel_positions,
                slot_mapping=slots,
                sparse_kv_indptr=indptr,
            )
        for runtime in self._runtimes[max(n, 2)]:
            runtime.advance_step()
        return model.norm(state[:n])

    def _bind(self) -> None:
        from aiter.ops.flydsl.glm5_mono import (
            AttentionWeight,
            Glm5MonoKernel,
            KvCacheLayout,
            glm5_tp_config,
            prepare_glm5_weights,
        )

        tp = get_tensor_model_parallel_world_size()
        cfg = glm5_tp_config(tp)
        rank = get_tensor_model_parallel_rank()
        group = get_tp_group().cpu_group
        layers = list(self._model.layers)
        fused = [_computes_indices(layer) for layer in layers]
        dense = [i < self._num_dense for i in range(len(layers))]
        weights = [
            _layer_weights(layer, cfg, rank, tp, with_indexer=f, dense=d)
            for layer, f, d in zip(layers, fused, dense)
        ]
        prepared = [
            prepare_glm5_weights(w, AttentionWeight.FP8_BLOCK128) for w in weights
        ]
        index_options = self._index_options(layers, fused)
        launches = {f: fused.count(f) for f in (False, True)}
        self._launch_index = [fused[:i].count(f) for i, f in enumerate(fused)]
        for chunk in sorted({max(n, 2) for n in _STEP_TOKENS}):
            runtimes: dict[bool, Glm5MonoKernel | None] = {False: None, True: None}
            ops = []
            for w, p, f, d in zip(weights, prepared, fused, dense):
                op = Glm5MonoKernel(
                    w,
                    chunk,
                    rank=rank,
                    npes=tp,
                    group=group,
                    topk=_SPARSE_TOPK,
                    launches_per_step=launches[f],
                    with_indexer=f,
                    attention_weight=AttentionWeight.FP8_BLOCK128,
                    kv_cache_layout=KvCacheLayout.ATOM,
                    kv_cache_dtype="fp8",
                    prepared_weights=p,
                    runtime=runtimes[f],
                    native_fp4_mfma=True,
                    dense_experts=w.physical_experts if d else 0,
                    **(index_options if f else {}),
                )
                runtimes[f] = runtimes[f] or op
                ops.append(op)
            self._ops[chunk] = ops
            self._runtimes[chunk] = [r for r in runtimes.values() if r is not None]
        # Drop the unpacked attention copies; every op reads the packed ones.
        for w, p in zip(weights, prepared):
            for name in ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o"):
                w.t[name] = p[name]

        attn = layers[0].self_attn
        cos_sin = attn.rotary_emb.cos_sin_cache
        half = cos_sin.shape[-1] // 2
        self._cos = cos_sin[:, :half].to(torch.bfloat16).contiguous()
        self._sin = cos_sin[:, half:].to(torch.bfloat16).contiguous()
        self._cur_pos = torch.zeros(1, dtype=torch.int32, device=cos_sin.device)
        logger.info_once(
            "GLM fused decode enabled for %d layers (%d dense, %d with the "
            "fused indexer), decode steps of %s tokens.",
            len(layers),
            self._num_dense,
            sum(fused),
            "/".join(map(str, _STEP_TOKENS)),
        )

    def _index_options(
        self, layers: list[DeepseekV32DecoderLayer], fused: list[bool]
    ) -> dict:
        if not any(fused):
            return {}
        attn = next(layer.self_attn for layer, f in zip(layers, fused) if f)
        if not torch.equal(
            attn.rotary_emb.cos_sin_cache, attn.indexer_rope_emb.cos_sin_cache
        ):
            raise ValueError(
                "VLLM_ROCM_GLM_MONO_DECODE requires the indexer and MLA "
                "RoPE tables to match."
            )
        cache = attn.indexer.k_cache.kv_cache
        md = get_forward_context().attn_metadata[self._layer_name]
        block = md.block_size
        return dict(
            index_max_seq=(self._max_model_len + block - 1) // block * block,
            index_paged=True,
            index_block_size=cache.shape[1],
            index_block_bytes=cache.stride(0) * cache.element_size(),
            index_shuffled=attn.indexer.k_cache.uses_shuffled_layout,
            block_table_stride=md.block_table.stride(0),
            index_k_bf16=True,
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
    # The preshuffling kernel leaves these weights in the plain layout.
    if getattr(linear, "skip_weight_relayout", False) or getattr(
        linear, "is_bmm", False
    ):
        return False
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


def _computes_indices(layer: DeepseekV32DecoderLayer) -> bool:
    attn = layer.self_attn
    return attn.indexer is not None and not attn.skip_topk


def _indexer_weights(layer: DeepseekV32DecoderLayer) -> dict[str, torch.Tensor]:
    indexer = layer.self_attn.indexer
    wk_w = indexer.wk_weights_proj.weight
    if (
        wk_w.dtype != torch.bfloat16
        or wk_w.shape[0] != indexer.head_dim + indexer.n_head
    ):
        raise ValueError(
            "VLLM_ROCM_GLM_MONO_DECODE requires a bf16 wk_weights_proj, "
            f"got {wk_w.dtype} {tuple(wk_w.shape)}."
        )
    w_index_q, s_index_q = _block_fp8(indexer.wq_b)
    return {
        "w_index_k": wk_w[: indexer.head_dim].contiguous(),
        "w_index_w": wk_w[indexer.head_dim :].contiguous(),
        "w_index_q": w_index_q,
        "s_index_q": s_index_q,
        "g_index_k": indexer.k_norm.weight.float().contiguous(),
        "b_index_k": indexer.k_norm.bias.float().contiguous(),
    }


def _fp4_storage(tensor: torch.Tensor) -> torch.Tensor:
    view = tensor.view(torch.uint8)
    view.is_shuffled = True
    return view


def _moe_weights(moe: DeepseekV2MoE, cfg) -> tuple[dict[str, torch.Tensor], int]:
    experts = moe.experts.routed_experts
    physical = cfg.n_experts + cfg.num_shared_experts
    if experts.w13_weight.shape[0] != physical:
        raise ValueError(
            f"VLLM_ROCM_GLM_MONO_DECODE expects {physical} physical experts, "
            f"got {experts.w13_weight.shape[0]}."
        )
    tensors = {
        "w_r": moe.gate.weight.to(torch.bfloat16).contiguous(),
        "bias": moe.gate.e_score_correction_bias.float().contiguous(),
        "w_ug": _fp4_storage(experts.w13_weight),
        "s_ug": experts.w13_weight_scale.view(torch.uint8),
        "w_dn": _fp4_storage(experts.w2_weight),
        "s_dn": experts.w2_weight_scale.view(torch.uint8),
    }
    return tensors, physical


def _mxfp4_rows(linear: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    weight = linear.weight.view(torch.uint8)
    rows, half_k = weight.shape
    scale = linear.weight_scale.view(torch.uint8)
    if scale.shape != (half_k * 2 // 32, rows):
        raise ValueError(
            "VLLM_ROCM_GLM_MONO_DECODE requires the dense MLP in the row-major "
            "MXFP4 layout of the AITER Triton GEMM, but "
            f"{getattr(linear, 'prefix', linear)} has weight_scale "
            f"{tuple(scale.shape)}."
        )
    return weight, scale.T.contiguous()


def _dense_mlp_weights(
    mlp: torch.nn.Module, cfg
) -> tuple[dict[str, torch.Tensor], int]:
    from aiter.ops.flydsl.glm5_mono import pack_dense_mlp

    gate_up = _mxfp4_rows(mlp.gate_up_proj)
    slices = gate_up[0].shape[0] // (2 * cfg.inter)
    if gate_up[0].shape[0] != 2 * cfg.inter * slices or not 0 < slices <= cfg.moe_slots:
        raise ValueError(
            "VLLM_ROCM_GLM_MONO_DECODE requires the dense MLP width to be a "
            f"multiple of the expert width {cfg.inter}, at most {cfg.moe_slots} "
            f"times, got {gate_up[0].shape[0] // 2}."
        )
    tensors = pack_dense_mlp(*gate_up, *_mxfp4_rows(mlp.down_proj), slices)
    device = gate_up[0].device
    tensors["w_r"] = torch.zeros(16, cfg.hidden, dtype=torch.bfloat16, device=device)
    tensors["bias"] = torch.zeros(cfg.n_experts, dtype=torch.float32, device=device)
    return tensors, slices


def _layer_weights(
    layer: DeepseekV32DecoderLayer,
    cfg,
    rank: int,
    tp: int,
    with_indexer: bool = False,
    dense: bool = False,
) -> LayerWeights:
    from aiter.ops.flydsl.glm5_mono import (
        LayerWeights,
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )

    attn = layer.self_attn
    if dense:
        mlp, physical = _dense_mlp_weights(layer.mlp, cfg)
    else:
        mlp, physical = _moe_weights(layer.mlp, cfg)
    w_qkv_a, s_qkv_a = _block_fp8(attn.fused_qkv_a_proj)
    w_q_b, s_q_b = _block_fp8(attn.q_b_proj)
    w_o, s_o = _block_fp8(attn.o_proj)
    w_uk, s_uk, w_uv, s_uv = split_kv_b(
        *_block_fp8(attn.kv_b_proj), cfg.local_heads, cfg.nope_dim, cfg.v_dim
    )
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
        **mlp,
    }
    if with_indexer:
        tensors.update(_indexer_weights(layer))
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        tp,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=physical,
    )
