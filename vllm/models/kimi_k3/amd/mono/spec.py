# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's ``MonoSpec`` on MI355X (``MONO_SPEC``, RFC #60904).

- Whole layers (KDA): K1 (AttnRes, in_proj, f_b, spec conv, the KDA recurrence,
  the gated norm), then K2.
- FFN layers (MLA, DCP1 or DCP8 over the TP group): vLLM's AttnRes and MLA
  split as ``mono/mla.py`` does: M1a (the query's prologue, at DCP8 its
  gather), AITER's decode, M1b (at DCP8 the LSE combine, W_UV, the gate), then
  K2 (o_proj, its all-reduce, the MLP AttnRes and the latent MoE).

A launch serves whole spec-verify requests, at most 16 rows (the kernels tile up
to 64, but from 32 rows on they are slower than vLLM's path). Weights are views
of the loaded parameters (``weight_source="views"``), pinned by the MoE
backend; every refusal stage reports every reason.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.common.mono.spec import (
    CacheReq,
    Feature,
    MonoContext,
    MonoSpec,
    MonoStep,
    MonoUnsupported,
)
from vllm.models.kimi_k3.amd.ops.mono.common.plan import BLOCKS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TP = 8
DCP = TP  # the MLA layers' KV shards: one a TP rank
DCP_SIZES = (1, DCP)  # M1's builds: the whole KV a rank, or DCP over the TP group
MLA_HEADS, MLA_Q, MLA_KV, MLA_NOPE, MLA_ROPE, MLA_V = 12, 1536, 512, 128, 64, 128
MAX_TOKENS = 64  # the kernels' rows: the MoE route table's token bitmaps
MAX_ROWS = 16  # a served step's rows: at 32 and 64 rows mono loses to vLLM
WHOLE, FFN = "whole", "ffn"


def widths(vllm_config: "VllmConfig") -> tuple[int, ...]:
    """Whole spec-verify requests of ``num_speculative_tokens + 1`` rows, at most
    MAX_ROWS rows."""
    spec = vllm_config.speculative_config
    q = 1 if spec is None else spec.num_speculative_tokens + 1
    return tuple(range(q, MAX_ROWS + 1, q))


def layer_kinds(config) -> dict[int, str]:
    """Decoder layer -> kind, for every layer a kernel serves: the MoE layers
    with AttnRes and at least one block before them, minus the MLA block-write
    layers (none in Kimi-K3)."""
    block = config.attn_res_block_size
    if block is None or config.num_experts is None:
        return {}
    out = {}
    for i in range(1, config.num_hidden_layers):
        if i < config.first_k_dense_replace or i % config.moe_layer_freq:
            continue
        if config.is_kda_layer(i):
            out[i] = WHOLE
        elif i % block:
            out[i] = FFN
    return out


def refuse_platform() -> list[str]:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["needs ROCm"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["needs gfx950 (MI355X)"]
    if torch.cuda.get_device_properties().multi_processor_count != BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"needs {BLOCKS} CUs (a partitioned GPU has fewer)"]
    return []


def refuse_config(vllm_config: "VllmConfig") -> list[str]:
    reasons = refuse_platform()
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp != TP:
        reasons.append(f"needs TP{TP}, got TP{tp}")
    pc = vllm_config.parallel_config
    dcp = pc.decode_context_parallel_size
    if dcp not in DCP_SIZES:
        reasons.append(f"needs DCP1 or DCP{DCP} (over the TP group), got DCP{dcp}")
    elif dcp > 1 and pc.cp_kv_cache_interleave_size != 1:
        reasons.append(
            f"needs KV interleave 1 (token t on DCP rank t % {DCP}), got "
            f"{pc.cp_kv_cache_interleave_size}"
        )
    if not vllm_config.cache_config.cache_dtype.startswith("fp8"):
        reasons.append(f"needs an fp8 KV cache, got {vllm_config.cache_config.cache_dtype}")
    spec = vllm_config.speculative_config
    if spec is None:
        reasons.append("needs speculative decoding (spec-verify steps)")
    elif spec.num_speculative_tokens + 1 > MAX_ROWS:
        reasons.append(
            f"{spec.num_speculative_tokens} speculative tokens: a request's "
            f"{spec.num_speculative_tokens + 1} rows exceed {MAX_ROWS}"
        )
    return reasons


def refuse_model(model) -> list[str]:
    """``model``: the loaded ``KimiLinearModel``; every kernel layer's weights
    as the kernels read them, and the MoE backend that lays them out."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.kimi_k3.amd.ops.mono.attention import back, kda
    from vllm.models.kimi_k3.amd.ops.mono.stages import moe

    reasons: list[str] = []
    bf, f32 = torch.bfloat16, torch.float32

    def need(name, t, shape, dtype):
        if t is None:
            reasons.append(f"{name}: missing")
        elif tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            reasons.append(
                f"{name}: {tuple(t.shape)} {t.dtype}"
                f"{'' if t.is_contiguous() else ' strided'},"
                f" kernel reads contiguous {shape} {dtype}"
            )

    kinds = layer_kinds(model.config)
    if not kinds:
        return ["no layer has an AttnRes latent MoE"]
    for i, kind in kinds.items():
        layer = model.layers[i]
        p = f"layer {i}"
        m = layer.mlp
        if not getattr(m, "use_latent_moe", False) or m.shared_experts is None:
            reasons.append(f"{p}: needs a latent MoE with shared experts")
            continue
        if m.routed_scaling_factor != 1.0:
            reasons.append(f"{p}: routed_scaling_factor {m.routed_scaling_factor}")
        ex = getattr(m.experts, "routed_experts", m.experts)
        qm = getattr(ex, "quant_method", None)
        if getattr(
            qm, "mxfp4_backend", None
        ) != Mxfp4MoeBackend.AITER_MXFP4_BF16 or not getattr(
            qm, "is_k3_situ_aiter", False
        ):
            reasons.append(f"{p}: MoE backend must be AITER_MXFP4_BF16 (K3 SiTU)")
            continue
        H, LAT, E = moe.HIDDEN, moe.LAT, moe.E
        need(f"{p} gate", m.gate.weight, (E, H), bf)
        need(f"{p} gate bias", m.gate.e_score_correction_bias, (E,), f32)
        need(f"{p} latent down", m.routed_expert_down_proj.weight, (LAT, H), bf)
        need(f"{p} latent norm", m.routed_expert_norm.weight, (LAT,), bf)
        need(f"{p} shared gate_up", m.shared_experts.gate_up_proj.weight,
             (2 * moe.SI, H), bf)  # fmt: skip
        need(f"{p} shared down", m.shared_experts.down_proj.weight, (H, moe.SI), bf)
        sharded = getattr(m.routed_output_transform, "row_sharded", False)
        need(f"{p} up_proj", m.routed_expert_up_proj.weight,
             (moe.UP_N if sharded else H, LAT), bf)  # fmt: skip
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            if getattr(ex, name, None) is None:
                reasons.append(f"{p}: experts have no {name}")
        need(f"{p} mlp_res_norm", layer.mlp_res_norm.weight, (H,), bf)
        need(f"{p} mlp_res_proj", layer.mlp_res_proj.weight, (1, H), bf)
        need(f"{p} post_attention_layernorm", layer.post_attention_layernorm.weight,
             (H,), bf)  # fmt: skip
        need(f"{p} o_proj", layer.self_attn.o_proj.weight, (H, back.OK), bf)
        need(f"{p} input_layernorm", layer.input_layernorm.weight, (H,), bf)
        need(f"{p} self_attention_res_norm", layer.self_attention_res_norm.weight,
             (H,), bf)  # fmt: skip
        need(f"{p} self_attention_res_proj", layer.self_attention_res_proj.weight,
             (1, H), bf)  # fmt: skip
        if kind == FFN:
            reasons += _refuse_mla(p, layer.self_attn.mla_attn, need)
            continue
        a = layer.self_attn
        if a.local_num_heads != kda.NH or a.head_dim != kda.HD or a.conv_size != 4:
            reasons.append(f"{p}: KDA heads/head_dim/conv_size")
            continue
        need(f"{p} in_proj", a.in_proj_qkvgfab.weight, (kda.NPROJ, H), bf)
        need(f"{p} f_b_proj", a.f_b_proj.weight, (kda.PROJ, kda.HD), bf)
        need(f"{p} o_norm", a.o_norm.weight, (kda.HD,), bf)
        need(f"{p} conv1d", a.conv1d.weight.view(-1, 4), (kda.QKV, 4), f32)
        need(f"{p} A_log", a.A_log.view(-1), (kda.NH,), f32)
        need(f"{p} dt_bias", a.dt_bias.view(-1), (kda.PROJ,), f32)
    return reasons


def _refuse_mla(p: str, w, need) -> list[str]:
    """An MLA layer as ``mono/mla.py`` (and M1) runs it: NoPE, gated, DCP1 or
    DCP8, an fp8 cache, W_UK / W_UV as MXFP4 bmms."""
    from vllm.models.kimi_k3.amd.ops.mono.attention.kda import HIDDEN as H

    reasons = []
    a = w.mla_attn
    dims = (w.num_heads, w.q_lora_rank, a.kv_lora_rank, w.qk_nope_head_dim,
            w.qk_rope_head_dim, w.v_head_dim)  # fmt: skip
    if dims != (MLA_HEADS, MLA_Q, MLA_KV, MLA_NOPE, MLA_ROPE, MLA_V):
        return [f"{p}: MLA heads/q_lora/kv_lora/nope/rope/v {dims}"]
    if w.rotary_emb is not None or w.indexer is not None or w.dcp_q_replicate:
        reasons.append(f"{p}: needs NoPE, dense, unreplicated-query MLA")
    if w.g_proj is None:
        reasons.append(f"{p}: needs the output gate (g_proj)")
    dcp = a.impl.dcp_world_size
    if dcp not in DCP_SIZES or a.use_pcp or a.hisparse_cache is not None:
        reasons.append(f"{p}: needs DCP1 or DCP{DCP}, no PCP / HiSparse")
    elif dcp > 1 and (a.dcp_manager is None or a.dcp_manager.query_gather is None):
        reasons.append(f"{p}: no DCP query gather")
    if not a.kv_cache_dtype.startswith("fp8"):
        reasons.append(f"{p}: KV cache {a.kv_cache_dtype}, needs fp8")
    if not a.is_aiter_triton_fp4_bmm_enabled:
        reasons.append(f"{p}: W_UK / W_UV must be AITER MXFP4 bmms")
    if float(a._q_scale_float) != 1.0 or float(a._k_scale_float) != 1.0:
        reasons.append(f"{p}: q / k scales must be 1.0")
    bf, u8 = torch.bfloat16, torch.uint8
    need(f"{p} fused_qkv_a_proj", w.fused_qkv_a_proj.weight,
         (MLA_Q + MLA_KV + MLA_ROPE, H), bf)  # fmt: skip
    need(f"{p} q_a_layernorm", w.q_a_layernorm.weight, (MLA_Q,), bf)
    need(f"{p} kv_a_layernorm", w.kv_a_layernorm.weight, (MLA_KV,), bf)
    need(f"{p} q_b_proj", w.q_b_proj.weight,
         (MLA_HEADS * (MLA_NOPE + MLA_ROPE), MLA_Q), bf)  # fmt: skip
    if w.g_proj is not None:
        need(f"{p} g_proj", w.g_proj.weight, (MLA_HEADS * MLA_V, H), bf)
    if a.is_aiter_triton_fp4_bmm_enabled:
        # the bmm weights are transposed views (rows strided, K dense); the
        # e8m0 scales are aiter's column-major blocks, any strides
        for name, t, shape, dense in (
            ("W_K", a.W_K, (MLA_HEADS, MLA_KV, MLA_NOPE // 2), True),
            ("W_K_scale", a.W_K_scale, (MLA_HEADS, MLA_KV, MLA_NOPE // 32), False),
            ("W_V", a.W_V, (MLA_HEADS, MLA_V, MLA_KV // 2), True),
            ("W_V_scale", a.W_V_scale, (MLA_HEADS, MLA_V, MLA_KV // 32), False),
        ):
            if t is None or tuple(t.shape) != shape:
                reasons.append(f"{p} {name}: needs {shape}")
            elif dense and t.stride(-1) != 1:
                reasons.append(f"{p} {name}: needs the last dim dense")
            elif t.element_size() != 1:
                reasons.append(f"{p} {name}: {t.dtype}, needs bytes ({u8})")
    return reasons


def pack(model, ctx: MonoContext) -> Any:
    """Views need no weight copy: the per-rank launch state (scratch, epoch,
    health words, peer memory). Collective."""
    from vllm.models.kimi_k3.amd.mono.runner import MonoRunner

    runner = MonoRunner(ctx.device)
    spec = ctx.vllm_config.speculative_config
    runner.num_spec = 0 if spec is None else spec.num_speculative_tokens
    kinds = layer_kinds(model.config)
    runner.first_layer = min(i for i in kinds if i >= model.start_layer)
    runner.kda_prefix = next(
        model.layers[i].self_attn.prefix for i, k in kinds.items() if k == WHOLE
    )
    runner.mla_prefix = next(
        (
            model.layers[i].self_attn.mla_attn.mla_attn.layer_name
            for i, k in kinds.items()
            if k == FFN
        ),
        "",
    )
    from vllm.distributed import get_dcp_group, get_tp_group

    dcp = get_dcp_group()
    if dcp.world_size > 1 and (
        dcp.world_size != DCP or dcp.rank_in_group != get_tp_group().rank_in_group
    ):
        raise MonoUnsupported(
            f"MonoKernel {MONO_SPEC.name}: DCP rank must be the TP rank "
            f"(DCP{dcp.world_size} rank {dcp.rank_in_group})"
        )
    ctx.state = runner
    return runner


def bind(ctx: MonoContext, caches: Mapping[str, Any]) -> list[str]:
    """Every KDA layer's conv / recurrent state as K1 reads them; every MLA
    layer's fp8 cache, rows of 576 bytes (the blocks may be strided)."""
    from vllm.models.kimi_k3.amd.ops.mono.attention import kda

    reasons = []
    for name, c in caches.items():
        if isinstance(c, torch.Tensor):
            if c.ndim != 3 or c.size(-1) != MLA_KV + MLA_ROPE or c.element_size() != 1:
                reasons.append(
                    f"{name}: MLA cache {tuple(c.shape)} {c.dtype}, needs fp8 "
                    f"[blocks, block_size, {MLA_KV + MLA_ROPE}]"
                )
            elif c.stride(-1) != 1 or c.stride(1) != MLA_KV + MLA_ROPE:
                reasons.append(f"{name}: MLA cache rows must be dense")
            continue
        conv_state, rstate = c
        if conv_state.dtype != torch.bfloat16:
            reasons.append(f"{name}: conv state {conv_state.dtype}, needs bf16")
        if rstate.dtype != torch.float32:
            reasons.append(f"{name}: recurrent state {rstate.dtype}, needs fp32")
        elif (
            rstate[0].numel() != kda.NH * kda.HD * kda.HD
            or not rstate[0].is_contiguous()
        ):
            reasons.append(f"{name}: a slot's recurrent state must be dense")
    return reasons


def eligible(ctx: MonoContext, step: MonoStep) -> str | None:
    """A pure spec-verify step of whole uniform requests, by the KDA metadata
    (the same on every rank)."""
    md = step.attn_metadata
    if not isinstance(md, dict):
        return "no metadata"
    m: Any = md.get(ctx.state.kda_prefix)
    if m is None or m.spec_sequence_masks is None:
        return "not spec-verify"
    if m.num_prefills or m.num_decodes or m.num_spec_decodes < 1:
        return "not pure decode"
    q = m.spec_state_indices_tensor.size(-1)
    if m.uniform_spec_sequence_length != q or m.num_spec_decodes * q != step.width:
        return "not uniform"
    if ctx.state.mla_prefix:
        a: Any = md.get(ctx.state.mla_prefix)
        if a is None or a.decode is None:
            return "no MLA decode metadata"
        if a.num_prefills or a.num_decode_tokens != step.width:
            return "MLA: not pure decode"
        if a.decode.max_qo_len != q:
            return "MLA: not uniform"
        # Gluon serves bf16 caches only; M1 hands AITER an fp8 query
        if getattr(a.decode, "use_gluon_decode", False) or getattr(
            a.decode, "use_gluon_verify", False
        ):
            return "MLA: Gluon decode"
    return None


def build(ctx: MonoContext, packed, layer, width: int):
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.amd.mono.kernel import FfnLayer, WholeLayer

    whole = isinstance(layer.self_attn, KimiK3DeltaAttention)
    return (WholeLayer if whole else FfnLayer)(layer, packed, width)


MONO_SPEC = MonoSpec(
    name="kimi_k3_mi355x",
    layers=lambda config: list(layer_kinds(config)),
    widths=widths,
    supports=frozenset({Feature.SPECULATIVE_DECODING, Feature.DECODE_CONTEXT_PARALLEL}),
    caches={
        "kda": CacheReq(
            group="mamba",
            dtype="conv bf16, recurrent fp32",
            layout="conv [slots, dim, state_len] (either order), "
            "recurrent [slots, 12, 128, 128] dense a slot",
        ),
        "mla": CacheReq(
            group="attention",
            dtype="fp8 e4m3 (uint8 storage)",
            layout="[blocks, block_size, 576]: kv_c 512 | k_pe 64, rows dense, "
            "blocks may be strided; a slot of -1 is not stored (DCP8: another "
            "rank's token)",
        ),
    },
    weight_source="views",
    options={},
    refuse_config=refuse_config,
    refuse_model=refuse_model,
    pack=pack,
    bind=bind,
    eligible=eligible,
    build=build,
)
