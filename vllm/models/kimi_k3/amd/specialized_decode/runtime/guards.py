# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The refusals that need the loaded model (``weight_reasons``) or the bound KV
caches (``cache_reasons``), and the fail-stop watch on the kernels' error words.
Each refusal reports every reason, not only the first."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    DCP,
    DCP_SIZES,
    FFN,
    layer_kinds,
)

MLA_HEADS, MLA_Q, MLA_KV, MLA_NOPE, MLA_ROPE, MLA_V = 12, 1536, 512, 128, 64, 128


def _need(reasons: list[str]):
    """``need(name, t, shape, dtype)``: appends to ``reasons`` unless ``t`` is
    a contiguous ``shape`` ``dtype`` tensor."""

    def need(name, t, shape, dtype):
        if t is None:
            reasons.append(f"{name}: missing")
        elif tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            reasons.append(
                f"{name}: {tuple(t.shape)} {t.dtype}"
                f"{'' if t.is_contiguous() else ' strided'},"
                f" kernel reads contiguous {shape} {dtype}"
            )

    return need


def weight_reasons(model) -> list[str]:
    """``model``: the loaded ``KimiLinearModel``; every kernel layer's weights as
    K1 and K2 read them, and the MoE backend that lays them out. The MLA layers'
    attention is ``mla_weight_reasons``'."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import back, kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    reasons: list[str] = []
    bf, f32 = torch.bfloat16, torch.float32
    need = _need(reasons)
    kinds = layer_kinds(model.config)
    if not kinds:
        return ["no layer has an AttnRes latent MoE"]
    for i, kind in kinds.items():
        if not model.start_layer <= i < model.end_layer:
            continue
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


def mla_weight_reasons(model) -> list[str]:
    """Why the loaded MLA layers cannot run M1 (empty: they can). Not a refusal:
    those layers run vLLM's attention ops instead, then K2."""
    reasons: list[str] = []
    for i, kind in layer_kinds(model.config).items():
        if kind == FFN and model.start_layer <= i < model.end_layer:
            w = model.layers[i].self_attn.mla_attn
            reasons += _mla_reasons(f"layer {i}", w, _need(reasons))
    return reasons


def _mla_reasons(p: str, w, need) -> list[str]:
    """An MLA layer as M1a / M1b (and ``runtime/mla.py``) run it: NoPE, gated,
    DCP1 or DCP8, an fp8 cache, W_UK / W_UV as MXFP4 bmms."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        HIDDEN as H,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.mla import (
        qkv_gate_weights,
    )

    reasons = []
    a = w.mla_attn
    dims = (w.num_heads, w.q_lora_rank, a.kv_lora_rank, w.qk_nope_head_dim,
            w.qk_rope_head_dim, w.v_head_dim)  # fmt: skip
    if dims != (MLA_HEADS, MLA_Q, MLA_KV, MLA_NOPE, MLA_ROPE, MLA_V):
        return [f"{p}: MLA heads/q_lora/kv_lora/nope/rope/v {dims}"]
    if w.rotary_emb is not None or w.indexer is not None or w.dcp_q_replicate:
        reasons.append(f"{p}: needs NoPE, dense, unreplicated-query MLA")
    w_qkv, w_g = qkv_gate_weights(w)
    if w_g is None:
        reasons.append(f"{p}: needs the output gate (g_proj)")
    dcp = a.impl.dcp_world_size
    if dcp not in DCP_SIZES or a.use_pcp or a.hisparse_cache is not None:
        reasons.append(f"{p}: needs DCP1 or DCP{DCP}, no PCP / HiSparse")
    elif dcp > 1 and (a.dcp_manager is None or a.dcp_manager.query_gather is None):
        reasons.append(f"{p}: no DCP query gather")
    if not a.kv_cache_dtype.startswith("fp8"):
        reasons.append(f"{p}: KV cache {a.kv_cache_dtype}, needs fp8")
    if not a.is_aiter_triton_fp4_bmm_enabled:
        reasons.append(
            f"{p}: W_UK / W_UV must be AITER MXFP4 bmms (VLLM_ROCM_USE_AITER_FP4BMM=1)"
        )
    if float(a._q_scale_float) != 1.0 or float(a._k_scale_float) != 1.0:
        reasons.append(f"{p}: q / k scales must be 1.0")
    bf = torch.bfloat16
    need(f"{p} fused_qkv_a_proj", w_qkv,
         (MLA_Q + MLA_KV + MLA_ROPE, H), bf)  # fmt: skip
    need(f"{p} q_a_layernorm", w.q_a_layernorm.weight, (MLA_Q,), bf)
    need(f"{p} kv_a_layernorm", w.kv_a_layernorm.weight, (MLA_KV,), bf)
    need(f"{p} q_b_proj", w.q_b_proj.weight,
         (MLA_HEADS * (MLA_NOPE + MLA_ROPE), MLA_Q), bf)  # fmt: skip
    if w_g is not None:
        need(f"{p} g_proj", w_g, (MLA_HEADS * MLA_V, H), bf)
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
                reasons.append(f"{p} {name}: {t.dtype}, needs bytes (torch.uint8)")
    return reasons


def cache_reasons(caches: Mapping[str, Any]) -> list[str]:
    """Every KDA layer's (conv, recurrent) state as K1 reads them; every MLA
    layer's fp8 cache, rows of 576 bytes (the blocks may be strided)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

    reasons = []
    for name, c in caches.items():
        if isinstance(c, torch.Tensor):
            row = MLA_KV + MLA_ROPE
            if c.ndim != 3 or c.size(-1) != row or c.element_size() != 1:
                reasons.append(
                    f"{name}: MLA cache {tuple(c.shape)} {c.dtype}, needs fp8 "
                    f"[blocks, block_size, {row}]"
                )
            elif c.stride(-1) != 1 or c.stride(1) != row:
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


class FailStop:
    """Stops every TP rank at the same step once any of them gave up an in-kernel
    wait (its health word FAILED, ``mono.abi``). Each step queues a D2H copy of
    this rank's word and keeps the newest copy that has landed, without blocking;
    every CHECK_EVERY steps the ranks agree on those over the TP CPU group and all
    raise together. A rank that raised alone would leave the others in the next
    collective (the drafter's) for ever. A failure is seen within about
    CHECK_EVERY steps."""

    CHECK_EVERY = 16

    def __init__(self, failed: torch.Tensor, rank: int, world: int, group):
        self.failed, self.rank, self.world, self.group = failed, rank, world, group
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied: torch.cuda.Event | None = None
        self.seen = False
        self.steps = 0

    def after_step(self) -> None:
        """Once per step on every TP rank (compute_logits, itself collective)."""
        if torch.cuda.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.cuda.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        torch.distributed.all_reduce(votes, group=self.group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"Kimi-K3 specialized decode: TP ranks {ranks} gave up an in-kernel "
                "wait; stopping the engine (decode_backend=auto runs vLLM's decode)"
            )
