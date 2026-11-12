# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer: vLLM's modules and tensors in, the
kernels' arguments out (``Launcher``), and ``attention_core``, the part of a
full-attention layer that stays vLLM's (q / k norms, RoPE and the attention on
the qkv rows the previous launch projected).

Up to ``K.SPLIT_MAX`` rows a layer's MLP half is ``mono.blocks.layer_block``
(``Launcher.mono_layer``) after the layer's split launch (``LayerBuild.split``)
and the final norm is vLLM's (``gemma_norm``); up to ``MONO_GDN_MAX`` rows a
GDN layer whose in_proj_qkvz is plain MXFP4 runs ``mono.blocks.gdn_block``
instead of its split launch (``Launcher.mono_gdn``).

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
    routed_experts,
    router_gate,
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
    SLOTS,
    launch_slots,
    sync_words,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange
from vllm.platforms import current_platform

# Profiling: set True in the worker before the model is built (for example on
# import of a ``--worker-extension-cls`` module) to build every launch with
# ``LayerBuild.trace``: CTA 0 stamps the clock at each of its phase ends into
# ``Launcher.trace`` ([layers, layer.TRACE_WORDS] int64, s_memrealtime ticks at
# 100 MHz; the phase names are ``layer.phase_names(key)``). Off: no stamps are
# compiled.
TRACE = False

# rows up to which a plain-MXFP4 in_proj_qkvz GDN layer runs gdn_block, faster
# there than the split launch (a graph replay at 1 / 2 rows: 13 / 15 us vs 19
# / 20; at 4 rows they tie)
MONO_GDN_MAX = 2


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
        self.slots = tuple(base + 4 * SLOT_WORDS * i for i in range(SLOTS))
        self.folds = base + 4 * FOLD_BASE
        self.traced = TRACE
        self.trace = torch.zeros(
            len(model.layers) * K.TRACE_WORDS, dtype=torch.int64, device=dev
        )
        self.args = [self._static_args(i) for i in range(len(model.layers))]
        self.launches = len(model.layers)
        self.split_widths = tuple(s for s in widths if s <= K.SPLIT_MAX)
        # the wide launches of a step at each width (their barrier slots)
        self.launch_count = {}
        for s in widths:
            n = sum(not self.mono_gdn_layer(i, s) for i in range(len(model.layers)))
            if n == 1:
                raise ValueError(f"one wide launch a step at {s} rows")
            self.launch_count[s] = n
        if self.split_widths:
            self._init_split(group, dev)

    def _init_split(self, group, dev) -> None:
        """The state of the split widths' ``layer_block`` launches: its scratch
        a width, its peer buffer and health words (``mono.abi``), and the
        hand-off rows between the launches (``h`` the MLP half's output,
        ``res`` the residual, ``core`` the GDN launch's)."""
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
        from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as ML
        from vllm.models.qwen3_5.amd.specialized_decode.mono.abi import (
            PEER_HEAD as MONO_HEAD,
        )
        from vllm.models.qwen3_5.amd.specialized_decode.mono.common.plan import (
            HEALTH_WORDS,
        )
        from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer

        def nbytes(s: int) -> int:
            lay = blocks.layer_scratch_layout(blocks.LayerBuild(tokens=s))
            return (blocks.scratch_bytes(lay) + 255) // 256 * 256

        self.mono_scratch = {
            s: torch.zeros(nbytes(s), dtype=torch.uint8, device=dev)
            for s in self.split_widths
        }
        self.mono_health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
        self.mono_peers = PeerBuffer(
            MONO_HEAD + blocks.peer_bytes(ML.MAX_TOKENS),
            group,
            self.rank,
            self.tp,
            dev,
        )
        self.mono_peers.bytes.zero_()
        self.mono_peer_table = self.mono_peers.addresses + MONO_HEAD
        self.mono_alarm = int(self.mono_peers.addresses[0].item()) + 4 * self.rank
        if self.rip != ML.RIP:
            raise ValueError(f"experts of {self.rip} rows a rank ({ML.RIP} only)")
        n = max(self.split_widths)
        bf = torch.bfloat16
        self.h = torch.empty(n, L.HIDDEN, dtype=bf, device=dev)
        self.res = torch.empty(n, L.HIDDEN, dtype=bf, device=dev)
        self.core = torch.empty(n, L.VALUE_DIM, dtype=bf, device=dev)
        # gdn_block's: its residual out apart from the residual it reads
        self.gdn_res = torch.empty(n, L.HIDDEN, dtype=bf, device=dev)
        self.gdn_scratch = {
            s: torch.zeros(
                blocks.scratch_bytes(
                    blocks.gdn_scratch_layout(blocks.GdnBuild(tokens=s))
                ),
                dtype=torch.uint8,
                device=dev,
            )
            for s in self.split_widths
            if s <= MONO_GDN_MAX
        }

    def view(self, name: str, rows: int, cols: int) -> torch.Tensor:
        """Scratch buffer ``name`` of a ``rows`` step as [rows, cols] bf16."""
        off, n = K.scratch_layout(rows, self.rip)[name]
        assert n >= rows * cols * 2, (name, n, rows, cols)
        return (
            self.scratch[off : off + rows * cols * 2]
            .view(torch.bfloat16)
            .view(rows, cols)
        )

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
        if rows <= K.SPLIT_MAX:
            return self._split_key(i, rows)
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        next_q4 = "none"
        if nxt is not None and _is_gdn(nxt):
            nq = linear_format(nxt.linear_attn.in_proj_qkvz)
            next_q4 = K.q4_of(nq.fmt, nq.shuffled) or "none"
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
            first_q4 = K.q4_of(qkvz.fmt, qkvz.shuffled) or "none"
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

    def _split_key(self, i: int, rows: int) -> K.LayerBuild:
        layer = self.model.layers[i]
        eps = layer.input_layernorm.variance_epsilon
        if not _is_gdn(layer):
            return K.LayerBuild(
                kind="attn",
                tokens=rows,
                tp=self.tp,
                eps=eps,
                trace=self.traced,
                split=True,
            )
        g = layer.linear_attn
        qkvz, out = linear_format(g.in_proj_qkvz), linear_format(g.out_proj)
        return K.LayerBuild(
            kind="gdn",
            tokens=rows,
            tp=self.tp,
            first=i == 0,
            qkvz=qkvz.fmt,
            qkvz_shuffled=qkvz.shuffled,
            out=out.fmt,
            out_shuffled=out.shuffled,
            first_q4=K.q4_of(qkvz.fmt, qkvz.shuffled) or "none",
            rip=self.rip,
            eps=eps,
            gdn_eps=g.norm.eps,
            trace=self.traced,
            split=True,
        )

    def mono_key(self, i: int, rows: int):
        """Layer ``i``'s ``layer_block`` build at a split width: out_proj /
        o_proj's format as ``linear_format`` reads it (an "mxfp4_h32" out_proj
        reads the rotated core the GDN launch leaves)."""
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks

        layer = self.model.layers[i]
        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.self_attn.o_proj
        f = linear_format(o)
        if not _is_gdn(layer) and f.fmt == "mxfp4_h32":
            raise ValueError("a Hadamard-rotated o_proj: its input is not rotated")
        return blocks.LayerBuild(
            tokens=rows,
            eps=layer.post_attention_layernorm.variance_epsilon,
            oq=f.fmt != "bf16",
            og=not _is_gdn(layer),
            shuffled=f.shuffled if f.fmt != "bf16" else True,
        )

    def launch(self, key, i: int, seq: int, dynamic: dict) -> int:
        """Launch ``key`` for layer ``i`` as the step's launch ``seq`` (which
        picks its barrier slots); returns the next ``seq``."""
        a = self.args[i] | dynamic
        sync, nxt = launch_slots(self.launch_count[key.tokens])[seq]
        a["sync"] = self.slots[sync]
        a["sync_next"] = self.slots[nxt]
        K.run(key, a)
        return seq + 1

    def mono_gdn_layer(self, i: int, rows: int) -> bool:
        """Whether layer ``i`` runs ``gdn_block`` at ``rows``: a GDN layer with
        a plain MXFP4 in_proj_qkvz (gdn_block's only format) and an out_proj
        that reads an unrotated core."""
        layer = self.model.layers[i]
        if rows > MONO_GDN_MAX or not _is_gdn(layer):
            return False
        g = layer.linear_attn
        return (
            linear_format(g.in_proj_qkvz).fmt == "mxfp4"
            and linear_format(g.out_proj).fmt != "mxfp4_h32"
        )

    def gdn_key(self, i: int, rows: int, first: bool):
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks

        layer = self.model.layers[i]
        g = layer.linear_attn
        return blocks.GdnBuild(
            tokens=rows,
            first=first,
            eps=layer.input_layernorm.variance_epsilon,
            norm_eps=g.norm.eps,
            shuffled=linear_format(g.in_proj_qkvz).shuffled,
        )

    def mono_gdn(self, i: int, h, residual, st_idx):
        """Layer ``i``'s residual add + input_layernorm and ``linear_attn`` up
        to out_proj's input (``gdn_block``; ``residual`` None on the first
        layer) -> (the core, the residual) in ``core`` and ``gdn_res``."""
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks

        g = self.model.layers[i].linear_attn
        rows = h.size(0)
        conv_state, rstate = gdn_states(g)
        core, res_out = self.core[:rows], self.gdn_res[:rows]
        blocks.gdn_block(
            self.gdn_key(i, rows, residual is None),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=self.model.layers[i].input_layernorm.weight,
            w_qkvz=g.in_proj_qkvz.weight,
            w_qkvzs=g.in_proj_qkvz.weight_scale,
            w_ba=g.in_proj_ba.weight,
            conv_w=g.conv1d.weight,
            conv_state=conv_state.transpose(-1, -2),
            a_log=g.A_log,
            dt_bias=g.dt_bias,
            norm_w=g.norm.weight,
            rstate=rstate,
            st_idx=st_idx,
            core=core,
            scratch=self.gdn_scratch[rows],
            epoch=self.epoch,
            health=self.mono_health,
            alarm=self.mono_alarm,
            layer=i,
        )
        return core, res_out

    def mono_layer(self, i: int, core, gate, residual):
        """Layer ``i``'s out_proj / o_proj, its all-reduce, the residual add +
        post_attention_layernorm and the MoE with its all-reduce
        (``layer_block``) -> (the layer's output, the residual) in ``h`` and
        ``res``."""
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks

        layer = self.model.layers[i]
        rows = core.size(0)
        key = self.mono_key(i, rows)
        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.self_attn.o_proj
        mlp = layer.mlp
        ex = routed_experts(mlp)
        out, res_out = self.h[:rows], self.res[:rows]
        blocks.layer_block(
            key,
            core=core,
            gate=gate,
            residual=residual,
            w_o=o.weight,
            w_os=o.weight_scale if key.oq else None,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.mono_scratch[rows],
            peers=self.mono_peer_table,
            rank=self.rank,
            epoch=self.epoch,
            health=self.mono_health,
            alarm=self.mono_alarm,
            layer=blocks.LAYER_SLOT + i,
        )
        return out, res_out


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


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _gemma_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``GemmaRMSNorm.forward_native(x, residual)`` in one launch: this forward
    is not torch.compiled, so the CustomOp would run as ~10 eager kernels."""
    t = x.float() + residual.float()
    res = t.to(x.dtype)
    t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (t * (weight.float() + 1.0)).to(x.dtype), res


def gemma_norm(norm, x, residual) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``GemmaRMSNorm`` with the residual add -> (x, the residual)."""
    return _gemma_add_rms_norm(x, residual, norm.weight, norm.variance_epsilon)
