# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The refusals that need the loaded model (``weight_reasons``) or the bound KV
caches (``cache_reasons``), and the fail-stop watch on the kernels' error words.
Each refusal reports every reason, not only the first."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm.models.kimi_k3.amd.specialized_decode.runtime import WHOLE, layer_kinds


def weight_reasons(model) -> list[str]:
    """``model``: the loaded ``KimiLinearModel``; every kernel layer's weights as
    the kernels read them, and the MoE backend that lays them out."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import back, kda
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

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
        if kind != WHOLE:
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
        need(f"{p} input_layernorm", layer.input_layernorm.weight, (H,), bf)
        need(f"{p} self_attention_res_norm", layer.self_attention_res_norm.weight,
             (H,), bf)  # fmt: skip
        need(f"{p} self_attention_res_proj", layer.self_attention_res_proj.weight,
             (1, H), bf)  # fmt: skip
    return reasons


def cache_reasons(caches: Mapping[str, Any]) -> list[str]:
    """Every KDA layer's (conv, recurrent) state as K1 reads them."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

    reasons = []
    for name, (conv_state, rstate) in caches.items():
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
