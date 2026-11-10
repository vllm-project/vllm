# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer: vLLM's modules and tensors in, the
kernels' arguments out (``Launcher``), and ``attention_core``, the part of a
full-attention layer that stays vLLM's (q / k norms, RoPE and the attention on
the qkv rows the previous launch projected).

``Launcher`` holds the per-rank state every launch reads: the scratch every
width shares (launches of a step run in order), the two barrier slots the
launches of a step alternate (``wide.common.grid``), the fold counters, the
step epoch (bumped once a step; a CUDA graph captures the bump, and the TP
exchange and tagged hand-off tags carry it), the health words and the peer
mailboxes of the TP exchange."""

import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.utils import warmup_rocm_skinny_gemm_workspaces
from vllm.models.qwen3_5.amd.specialized_decode.runtime.guards import (
    experts_rip,
    gdn_states,
    linear_format,
    router_gate,
    routed_experts,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide import layer as K
from vllm.models.qwen3_5.amd.specialized_decode.wide import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.wide.abi import (
    PEER_HEAD,
    health_words,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.common.grid import (
    FOLD_BASE,
    SLOT_WORDS,
    sync_words,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange

# Profiling: set True in the worker before the model is built (for example on
# import of a ``--worker-extension-cls`` module) to build every launch with
# ``LayerBuild.trace``: CTA 0 stamps the clock at each of its phase ends into
# ``Launcher.trace`` ([layers, layer.TRACE_WORDS] int64, s_memrealtime ticks at
# 100 MHz; the phase names are ``layer.phase_names(key)``). Off: no stamps are
# compiled.
TRACE = False


def _unused() -> dict:
    """Every ABI argument as 0, the base of a layer's argument set."""
    return dict.fromkeys(K.ABI.names, 0)


def _is_gdn(layer) -> bool:
    return layer.layer_type == "linear_attention"


def _scales(linear):
    return 0 if linear_format(linear).fmt == "bf16" else linear.weight_scale


class Launcher:
    """Per-rank launch state and the launches of every layer of a step."""

    def __init__(self, model, widths, tp: int, rank: int, group, dev):
        from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer

        self.model, self.widths, self.tp, self.rank = model, widths, tp, rank
        self.rip = experts_rip(model.layers[0].mlp)
        # zeroed: the tagged hand-offs are compared against step tags
        self.scratch = torch.zeros(
            max(K.scratch_bytes(s, self.rip) for s in widths),
            dtype=torch.uint8,
            device=dev,
        )
        self.out = torch.empty(max(widths), L.HIDDEN, dtype=torch.bfloat16, device=dev)
        # vLLM's skinny GEMMs create their workspace pool on first use, which
        # must not be inside a graph capture
        warmup_rocm_skinny_gemm_workspaces(dev)
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=dev)
        self.health = torch.zeros(health_words(L.BLOCKS), dtype=torch.int32, device=dev)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=dev)
        # collective: every TP rank builds its executor at the same point
        self.peers = PeerBuffer(
            PEER_HEAD + exchange.mailbox_bytes(tp), group, rank, tp, dev
        )
        self.peers.bytes.zero_()
        self.peer_table = torch.zeros(L.TP, dtype=torch.int64, device=dev)
        self.peer_table[:tp] = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * tp].view(torch.int32)
        base = self.sync.data_ptr()
        self.slots = (base, base + 4 * SLOT_WORDS)
        self.folds = base + 4 * FOLD_BASE
        self.traced = TRACE
        self.trace = torch.zeros(
            len(model.layers) * K.TRACE_WORDS, dtype=torch.int64, device=dev
        )
        self.args = [self._static_args(i) for i in range(len(model.layers))]
        self.launches = len(model.layers)
        if self.launches % 2:
            raise ValueError(
                f"{self.launches} launches a step: the barrier slots need an even count"
            )

    def view(self, name: str, rows: int, cols: int) -> torch.Tensor:
        """Scratch buffer ``name`` of a ``rows`` step as [rows, cols] bf16."""
        off, n = K.scratch_layout(rows, self.rip)[name]
        assert n >= rows * cols * 2, (name, n, rows, cols)
        return self.scratch[off : off + rows * cols * 2].view(torch.bfloat16).view(rows, cols)

    def _next_norm(self, i: int):
        layers = self.model.layers
        if i + 1 < len(layers):
            return layers[i + 1].input_layernorm.weight
        return self.model.norm.weight

    # the arguments every step of a layer shares (weights; caches at bind)
    def _static_args(self, i: int) -> dict:
        layer = self.model.layers[i]
        a = _unused()
        mlp = layer.mlp
        ex = routed_experts(mlp)
        a |= {
            "scratch": self.scratch,
            "n_in": layer.input_layernorm.weight,
            "n_post": layer.post_attention_layernorm.weight,
            "n_next": self._next_norm(i),
            "w_gate": router_gate(mlp),
            "w13": ex.w13_weight,
            "w13s": ex.w13_weight_scale,
            "w2": ex.w2_weight,
            "w2s": ex.w2_weight_scale,
            "peers": self.peer_table,
            "rank": self.rank,
            "epoch": self.epoch,
            "layer": i,
            "folds": self.folds,
            "health": self.health,
            "alarm": self.alarm,
            "trace": self.trace if self.traced else 0,
        }
        nxt = self.model.layers[i + 1] if i + 1 < len(self.model.layers) else None
        if nxt is not None and not _is_gdn(nxt):
            a["w_qkv"] = nxt.self_attn.qkv_proj.weight
        if _is_gdn(layer):
            g = layer.linear_attn
            a |= {
                "w_qkvz": g.in_proj_qkvz.weight,
                "w_qkvz_s": _scales(g.in_proj_qkvz),
                "w_ba": g.in_proj_ba.weight,
                "w_conv": g.conv1d.weight,
                "a_log": g.A_log,
                "dt_bias": g.dt_bias,
                "gdn_norm": g.norm.weight,
                "w_out": g.out_proj.weight,
                "w_out_s": _scales(g.out_proj),
            }
        else:
            a["w_out"] = layer.self_attn.o_proj.weight
        return a

    def bind_caches(self) -> None:
        """The GDN states vLLM just bound, into every GDN layer's arguments."""
        for layer, a in zip(self.model.layers, self.args):
            if _is_gdn(layer):
                conv_state, rstate = gdn_states(layer.linear_attn)
                a |= {
                    "conv_state": conv_state,
                    "conv_stride": conv_state.stride(0),
                    "ssm": rstate,
                    "ssm_stride": rstate.stride(0),
                }

    def key(self, i: int, rows: int) -> K.LayerBuild:
        """Layer ``i``'s build at ``rows``."""
        layers = self.model.layers
        layer = layers[i]
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        next_q4 = "none"
        if nxt is not None and _is_gdn(nxt):
            next_q4 = K.q4_of(linear_format(nxt.linear_attn.in_proj_qkvz).fmt) or "none"
        common = dict(
            tokens=rows,
            tp=self.tp,
            rip=self.rip,
            eps=layer.post_attention_layernorm.variance_epsilon,
            next_q4=next_q4,
            qkv_tail=nxt is not None and not _is_gdn(nxt),
            trace=self.traced,
        )
        if not _is_gdn(layer):
            return K.LayerBuild(kind="attn", **common)
        g = layer.linear_attn
        qkvz, out = linear_format(g.in_proj_qkvz), linear_format(g.out_proj)
        first_q4 = "none"
        if i == 0:
            first_q4 = K.q4_of(qkvz.fmt) or "none"
        return K.LayerBuild(
            kind="gdn",
            first=i == 0,
            qkvz=qkvz.fmt,
            qkvz_shuffled=qkvz.shuffled,
            out=out.fmt,
            out_shuffled=out.shuffled,
            first_q4=first_q4,
            gdn_eps=g.norm.eps,
            **common,
        )

    def launch(self, key, i: int, seq: int, dynamic: dict) -> int:
        """Launch ``key`` for layer ``i`` as the step's launch ``seq`` (which
        picks its barrier slot); returns the next ``seq``."""
        a = self.args[i] | dynamic
        a["sync"] = self.slots[seq % 2]
        a["sync_next"] = self.slots[(seq + 1) % 2]
        K.run(key, a)
        return seq + 1


def attention_core(attn, qkv, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3NextAttention.forward`` from qkv_proj's output to o_proj's
    input, without the output gate (the attn launch applies it): the split,
    q / k norms, RoPE and gate copy in one launch, then the attention ->
    (attention output [S, heads HD], gate [S, heads HD])."""
    q_gate, k, v = qkv.split([attn.q_size * 2, attn.kv_size, attn.kv_size], dim=-1)
    rope = attn.rotary_emb
    if positions.ndim == 2:
        # a decode token is text: its M-RoPE positions are all equal, so
        # every section rotates by the same position
        positions = positions[0]
    q, k, gate = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        attn.q_norm.weight,
        attn.k_norm.weight,
        rope.cos_sin_cache,
        positions,
        attn.q_norm.variance_epsilon,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        rope.rotary_dim,
        norm_beta=1.0,
    )
    out = attn.attn(q, k, v)
    rows = qkv.size(0)
    return out.reshape(rows, -1).contiguous(), gate.reshape(rows, -1).contiguous()
