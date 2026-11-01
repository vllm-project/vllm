# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer: vLLM's modules and tensors in, the
kernels' arguments out (``Launcher``), and ``qsa_core``, vLLM's QSA attention
up to o_proj's input.

``Launcher`` holds the per-rank state every launch reads: the scratch every
width shares (launches of a step run in order), the block-input buffer, the
two barrier slots the launches of a step alternate (``wide.common.grid``), the
fold counters, the step epoch (bumped once a step; a CUDA graph captures the
bump, and the TP exchange tags carry it), the health words and the peer
mailboxes of the TP exchange."""

import os

import torch

from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    experts_rip,
    gdn_states,
    mxfp4_shuffled,
    router_gate,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layer as K
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import (
    PEER_HEAD,
    health_words,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    FOLD_BASE,
    SLOT_WORDS,
    sync_words,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange
from vllm.platforms import current_platform

# Debugging: with this set, every launch is built with ``LayerBuild.trace`` and
# CTA 0 stamps the clock at each of its phase ends into ``Launcher.trace``
# ([layers, layer.TRACE_WORDS] int64, s_memrealtime ticks at 100 MHz; the
# phase names are ``layer.PHASES[key]``). Off: no stamps are compiled.
TRACE_ENV = "VLLM_QWEN4_EXP_DECODE_TRACE"


def _hc_args(hc, prefix: str) -> dict:
    return {
        f"{prefix}_norm": hc.hc_norm.weight,
        f"{prefix}_di": hc.input_mix_weight_down_block_inject.weight,
        f"{prefix}_up": hc.input_mix_weight_up.weight,
    }


def _unused() -> dict:
    """Every ABI argument as 0, the base of a layer's argument set."""
    return dict.fromkeys(K.ABI.names, 0)


class Launcher:
    """Per-rank launch state and the launches of every layer of a step."""

    def __init__(self, model, widths, tp: int, rank: int, group, dev):
        from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import PeerBuffer

        self.model, self.widths, self.tp, self.rank = model, widths, tp, rank
        self.scratch = torch.empty(
            K.scratch_bytes(max(widths)), dtype=torch.uint8, device=dev
        )
        self.x = torch.empty(max(widths), L.HIDDEN, dtype=torch.bfloat16, device=dev)
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
        self.health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=dev)
        # collective: every TP rank builds its executor at the same point
        self.peers = PeerBuffer(
            PEER_HEAD + exchange.mailbox_bytes(tp), group, rank, tp, dev
        )
        self.peers.bytes.zero_()
        self.peer_table = torch.zeros(4, dtype=torch.int64, device=dev)
        self.peer_table[:tp] = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * tp].view(torch.int32)
        base = self.sync.data_ptr()
        self.slots = (base, base + 4 * SLOT_WORDS)
        self.folds = base + 4 * FOLD_BASE
        self.rip = experts_rip(next(ly.mlp for ly in model.layers))
        self.traced = os.environ.get(TRACE_ENV, "0") not in ("", "0")
        self.trace = torch.zeros(
            len(model.layers) * K.TRACE_WORDS, dtype=torch.int64, device=dev
        )
        self.args = [self._static_args(ly) for ly in model.layers]
        self.launches = sum(
            2 if ly.layer_type != "linear_attention" else 1 for ly in model.layers
        )
        if self.launches % 2:
            raise ValueError(
                f"{self.launches} launches a step: the barrier slots need an even count"
            )

    # the arguments every step of a layer shares (weights; caches at bind)
    def _static_args(self, layer) -> dict:
        a = _unused()
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        a |= _hc_args(layer.attn_hyper_connection, "a")
        a |= _hc_args(layer.mlp_hyper_connection, "m")
        a |= {
            "scratch": self.scratch,
            "w_gate": router_gate(mlp),
            "w13": ex.w13_weight,
            "w13s": ex.w13_weight_scale,
            "w2": ex.w2_weight,
            "w2s": ex.w2_weight_scale,
            "peers": self.peer_table,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": layer.layer_idx,
            "folds": self.folds,
            "health": self.health,
            "alarm": self.alarm,
            "trace": self.trace if self.traced else 0,
        }
        if layer.layer_type == "linear_attention":
            g = layer.linear_attn
            a |= {
                "w_qkvz": g.in_proj_qkvz.weight,
                "w_qkvz_s": g.in_proj_qkvz.weight_scale,
                "w_ba": g.in_proj_ba.weight,
                "w_conv": g.conv1d.weight,
                "a_log": g.A_log,
                "dt_bias": g.dt_bias,
                "gdn_norm": g.norm.weight,
                "w_out": g.out_proj.weight,
                "w_out_s": g.out_proj.weight_scale,
            }
        else:
            a["w_out"] = layer.self_attn.o_proj.weight
        return a

    def bind_caches(self) -> None:
        """The GDN states vLLM just bound, into every GDN layer's arguments."""
        for layer, a in zip(self.model.layers, self.args):
            if layer.layer_type == "linear_attention":
                conv_state, rstate = gdn_states(layer.linear_attn)
                a |= {
                    "conv_state": conv_state,
                    "conv_stride": conv_state.stride(0),
                    "ssm": rstate,
                    "ssm_stride": rstate.stride(0),
                }

    def keys(self, layer, rows: int, stats: bool) -> list:
        """The builds a layer's launches use at ``rows`` (``stats``: its input
        state came from vLLM)."""
        common = dict(
            tokens=rows,
            tp=self.tp,
            rip=self.rip,
            hc_eps=layer.attn_hyper_connection.config.rms_norm_eps,
            trace=self.traced,
        )
        if layer.layer_type == "linear_attention":
            g = layer.linear_attn
            return [
                K.LayerBuild(
                    kind="gdn",
                    stats=stats,
                    qkvz_shuffled=bool(mxfp4_shuffled(g.in_proj_qkvz)),
                    out_shuffled=bool(mxfp4_shuffled(g.out_proj)),
                    gdn_eps=g.norm.eps,
                    **common,
                )
            ]
        return [
            K.LayerBuild(kind="qsa_front", stats=stats, **common),
            K.LayerBuild(kind="qsa_back", **common),
        ]

    def launch(self, key, i: int, seq: int, dynamic: dict) -> int:
        """Launch ``key`` for layer ``i`` as the step's launch ``seq`` (which
        picks its barrier slot); returns the next ``seq``."""
        a = self.args[i] | dynamic
        a["sync"] = self.slots[seq % 2]
        a["sync_next"] = self.slots[(seq + 1) % 2]
        K.run(key, a)
        return seq + 1


def qsa_core(attn, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen4ExpQSAAttention.forward`` up to the output gate (the qsa_back
    launch applies it to o_proj's input) -> (attention output [S, heads HD],
    gate [S, heads HD])."""
    from vllm.models.qwen4_exp.amd import qsa

    qkv, _ = attn.qkv_proj(x)
    n = x.shape[0]
    name = qsa._encode_layer_name(attn.layer_name)
    out = qkv.new_empty(n, attn.num_heads, attn.head_dim)
    gate = torch.empty_like(out)
    if current_platform.opaque_attention_op():
        torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
            x, positions, qkv, out, gate, name
        )
    else:
        qsa.qwen4_exp_qsa_prepare_with_output(x, positions, qkv, out, gate, name)
    return out.view(n, -1), gate.view(n, -1)
