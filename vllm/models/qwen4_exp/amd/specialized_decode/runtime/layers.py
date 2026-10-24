# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer, vLLM's modules and tensors in, the
kernels' arguments out, and ``qsa_core`` (vLLM's QSA attention up to o_proj's
input). ``Launcher`` holds the per-rank state every launch reads: the scratch
of each width, the peer buffer, the step epoch and the health words."""

import torch

from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    gdn_conv_state,
    mxfp4_shuffled,
    o_proj,
)
from vllm.platforms import current_platform


def _buf(n: int, dev) -> torch.Tensor:
    return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)


class Launcher:
    """Per-rank launch state. Two scratch sets a width: the attention half of
    a layer (its HC front, the GDN block) and the MLP half (its HC front, the
    MoE block), so a launch never reuses the scratch the previous one may
    still poll. Mailbox tags carry ``epoch``, bumped once a step (a CUDA graph
    captures the bump), so no buffer is zeroed between steps.

    Every in-kernel wait is bounded (``mono.abi``): a wait that gives
    up records itself in ``health``, sets this rank's word in the output rank's
    ``flags`` (the peer buffer's first ``PEER_HEAD`` bytes on rank 0) and marks
    the rank FAILED, so its later waits exit at once."""

    def __init__(self, widths, tp: int, rank: int, group, dev):
        from vllm.models.qwen4_exp.amd.specialized_decode.mono.abi import PEER_HEAD
        from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import (
            HEALTH_WORDS,
        )
        from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import PeerBuffer

        self.tp, self.rank = tp, rank

        def front_bytes(s):
            return max(
                front.scratch_bytes(front.FrontBuild(tokens=s, tp=tp, self_msq=m))
                for m in (False, True)
            )

        def attn_bytes(s):
            gdn = blocks.gdn_scratch_layout(blocks.GdnBuild(tokens=s, tp=tp))
            return max(blocks.scratch_bytes(gdn), front_bytes(s))

        def mlp_bytes(s):
            moe = blocks.moe_scratch_layout(blocks.MoeBuild(tokens=s, tp=tp))
            return max(blocks.scratch_bytes(moe), front_bytes(s))

        self.attn_scratch = {s: _buf(attn_bytes(s), dev) for s in widths}
        self.mlp_scratch = {s: _buf(mlp_bytes(s), dev) for s in widths}
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
        # collective: every TP rank builds its executor at the same point
        self.peers = PeerBuffer(
            PEER_HEAD + blocks.peer_bytes(L.MAX_TOKENS, tp), group, rank, tp, dev
        )
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * tp].view(torch.int32)

    def hc_front(self, layer, hc, state, msq, scratch):
        """``hc.mix(state)`` -> (block input [S, H], injection logits [S, HC]);
        ``msq``: the state's sums of squares the previous block wrote, None to
        compute them."""
        s = state.size(0)
        x = state.new_empty(s, L.HIDDEN)
        inj = state.new_empty(s, L.HC)
        front.hc_front(
            self.front_key(hc, s, msq is None),
            state=state,
            msq_in=msq,
            hc_w=hc.hc_norm.weight,
            w_down=hc.input_mix_weight_down_block_inject.weight,
            w_up=hc.input_mix_weight_up.weight,
            inj_out=inj,
            bin_out=x,
            scratch=scratch[s],
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=blocks.HC_SLOT + layer.layer_idx,
        )
        return x, inj

    def gdn(self, layer, md, x) -> torch.Tensor:
        """``linear_attn`` up to out_proj's input -> core [S, CORE]."""
        a = layer.linear_attn
        conv_state, rstate = gdn_conv_state(a)
        s = x.size(0)
        core = x.new_empty(s, L.Layout(self.tp).CORE)
        blocks.gdn_block(
            self.gdn_key(layer, s),
            x=x,
            w_qkvz=a.in_proj_qkvz.weight,
            w_qkvzs=a.in_proj_qkvz.weight_scale,
            w_ba=a.in_proj_ba.weight,
            conv_w=a.conv1d.weight,
            conv_state=conv_state,
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=rstate,
            st_idx=md[a.prefix].non_spec_state_indices_tensor,
            core=core,
            scratch=self.attn_scratch[s],
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=layer.layer_idx,
        )
        return core

    def oproj(self, layer, core, gate, state, inj):
        """o_proj, its all-reduce and the attention HC's combine -> (the next
        state, its sums of squares)."""
        o = o_proj(layer)
        key = self.oproj_key(layer, core.size(0))
        out, msq = _block_out(state)
        blocks.oproj_block(
            key,
            core=core,
            gate=gate,
            w_o=o.weight,
            w_os=o.weight_scale if key.oq else None,
            state=state,
            inj=inj,
            out=out,
            msq_out=msq,
            peers=self.peer_table,
            rank=self.rank,
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=blocks.OPROJ_SLOT + layer.layer_idx,
        )
        return out, msq

    def moe(self, layer, x, state, inj):
        """The MoE, its all-reduce and the MLP HC's combine -> (the next state,
        its sums of squares)."""
        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        out, msq = _block_out(state)
        blocks.moe_block(
            blocks.MoeBuild(tokens=x.size(0), tp=self.tp),
            x=x,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            state=state,
            inj=inj,
            out=out,
            msq_out=msq,
            scratch=self.mlp_scratch[x.size(0)],
            peers=self.peer_table,
            rank=self.rank,
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=blocks.MOE_SLOT + layer.layer_idx,
        )
        return out, msq

    # the build keys, also what ``SpecializedDecode`` compiles at build
    def front_key(self, hc, s: int, self_msq: bool) -> front.FrontBuild:
        return front.FrontBuild(
            tokens=s, tp=self.tp, self_msq=self_msq, eps=hc.config.rms_norm_eps
        )

    def gdn_key(self, layer, s: int) -> blocks.GdnBuild:
        a = layer.linear_attn
        return blocks.GdnBuild(
            tokens=s,
            tp=self.tp,
            norm_eps=a.norm.eps,
            shuffled=bool(mxfp4_shuffled(a.in_proj_qkvz)),
        )

    def oproj_key(self, layer, s: int) -> blocks.OprojBuild:
        shuffled = mxfp4_shuffled(o_proj(layer))
        return blocks.OprojBuild(
            tokens=s,
            tp=self.tp,
            oq=shuffled is not None,
            og=layer.layer_type != "linear_attention",
            shuffled=shuffled is not False,
        )


def _block_out(state):
    """A block's outputs: the next state and its sums of squares."""
    s = state.size(0)
    msq = torch.empty(blocks.NRED, s, L.HC, dtype=torch.float32, device=state.device)
    return torch.empty_like(state), msq


def qsa_core(attn, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen4ExpQSAAttention.forward`` up to the output gate (the o_proj block
    applies it to o_proj's input) -> (attention output [S, heads HD], gate)."""
    from vllm.models.qwen4_exp.amd import qsa

    qkv, _ = attn.qkv_proj(x)
    n = x.shape[0]
    name = qsa._encode_layer_name(attn.layer_name)
    if attn.use_fused_qsa_prepare:
        out = qkv.new_empty(n, attn.num_heads, attn.head_dim)
        gate = torch.empty_like(out)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen4_exp_qsa_prepare_with_output(
                x, positions, qkv, out, gate, name
            )
        else:
            qsa.qwen4_exp_qsa_prepare_with_output(x, positions, qkv, out, gate, name)
        return out.view(n, -1), gate.view(n, -1)
    q, k, v, gate = attn._project_qkv_gate(qkv, positions)
    query = q.view(n, attn.num_heads, attn.head_dim)
    key = k.view(n, attn.num_kv_heads, attn.head_dim)
    value = v.view(n, attn.num_kv_heads, attn.head_dim)
    out = torch.empty_like(query)
    if current_platform.opaque_attention_op():
        torch.ops.vllm.qwen4_exp_qsa_with_output(
            x, positions, query, key, value, out, name
        )
    else:
        qsa.qwen4_exp_qsa_with_output(x, positions, query, key, value, out, name)
    return out.view(n, -1), gate.contiguous()
