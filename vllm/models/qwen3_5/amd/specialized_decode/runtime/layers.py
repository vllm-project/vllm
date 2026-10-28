# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer, vLLM's modules and tensors in, the
kernels' arguments out, and the parts of a layer that stay vLLM's: a full
attention layer's input norm and attention up to o_proj's input
(``attention_core``) and the final norm (``gemma_norm``). ``Launcher`` holds
the per-rank state every launch reads: the scratch of each width, the peer
buffer, the step epoch and the health words."""

import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks
from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.runtime.guards import (
    gdn_conv_state,
    mxfp4_shuffled,
    o_proj,
    routed_experts,
)
from vllm.platforms import current_platform


def _buf(n: int, dev) -> torch.Tensor:
    return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)


class Launcher:
    """Per-rank launch state. Two scratch sets a width, one for each launch of
    a layer, so a launch never reuses the scratch the previous one may still
    poll. Mailbox tags carry ``epoch``, bumped once a step (a CUDA graph
    captures the bump), so no buffer is zeroed between steps.

    Every in-kernel wait is bounded (``mono.abi``): a wait that gives
    up records itself in ``health``, sets this rank's word in the output rank's
    ``flags`` (the peer buffer's first ``PEER_HEAD`` bytes on rank 0) and marks
    the rank FAILED, so its later waits exit at once."""

    def __init__(self, widths, rank: int, group, dev):
        from vllm.models.qwen3_5.amd.specialized_decode.mono.abi import PEER_HEAD
        from vllm.models.qwen3_5.amd.specialized_decode.mono.common.plan import (
            HEALTH_WORDS,
        )
        from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import PeerBuffer

        self.rank = rank

        def gdn_bytes(s):
            return blocks.scratch_bytes(
                blocks.gdn_scratch_layout(blocks.GdnBuild(tokens=s))
            )

        def layer_bytes(s):
            return blocks.scratch_bytes(
                blocks.layer_scratch_layout(blocks.LayerBuild(tokens=s))
            )

        self.gdn_scratch = {s: _buf(gdn_bytes(s), dev) for s in widths}
        self.layer_scratch = {s: _buf(layer_bytes(s), dev) for s in widths}
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=dev)
        # collective: every TP rank builds its executor at the same point
        self.peers = PeerBuffer(
            PEER_HEAD + blocks.peer_bytes(L.MAX_TOKENS), group, rank, L.TP, dev
        )
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * L.TP].view(torch.int32)

    def gdn(self, layer, md, h, residual):
        """A GDN layer's residual add + input_layernorm and ``linear_attn`` up
        to out_proj's input -> (core [S, CORE], the residual)."""
        a = layer.linear_attn
        conv_state, rstate = gdn_conv_state(a)
        s = h.size(0)
        core = h.new_empty(s, L.CORE)
        res_out = torch.empty_like(h)
        blocks.gdn_block(
            self.gdn_key(layer, s, residual is None),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
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
            scratch=self.gdn_scratch[s],
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=layer.layer_idx,
        )
        return core, res_out

    def layer(self, layer, core, gate, residual):
        """o_proj and its all-reduce, the residual add +
        post_attention_layernorm, the MoE and its all-reduce -> (the layer's
        output, the residual)."""
        o = o_proj(layer)
        mlp = layer.mlp
        ex = routed_experts(mlp)
        key = self.layer_key(layer, core.size(0))
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
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
            scratch=self.layer_scratch[core.size(0)],
            peers=self.peer_table,
            rank=self.rank,
            epoch=self.epoch,
            health=self.health,
            alarm=self.alarm,
            layer=blocks.LAYER_SLOT + layer.layer_idx,
        )
        return out, res_out

    # the build keys, also what ``SpecializedDecode`` compiles at build
    def gdn_key(self, layer, s: int, first: bool) -> blocks.GdnBuild:
        a = layer.linear_attn
        return blocks.GdnBuild(
            tokens=s,
            first=first,
            eps=layer.input_layernorm.variance_epsilon,
            norm_eps=a.norm.eps,
            shuffled=bool(mxfp4_shuffled(a.in_proj_qkvz)),
        )

    def layer_key(self, layer, s: int) -> blocks.LayerBuild:
        shuffled = mxfp4_shuffled(o_proj(layer))
        return blocks.LayerBuild(
            tokens=s,
            eps=layer.post_attention_layernorm.variance_epsilon,
            oq=shuffled is not None,
            og=layer.layer_type != "linear_attention",
            shuffled=shuffled is not False,
        )


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
    if residual is None:
        return norm(x), x
    return _gemma_add_rms_norm(x, residual, norm.weight, norm.variance_epsilon)


def attention_core(attn, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen3NextAttention.forward`` up to the output gate (the layer block
    applies it to o_proj's input), the split, q / k norms, RoPE and gate copy
    in one launch -> (attention output [S, heads HD], gate)."""
    qkv, _ = attn.qkv_proj(x)
    q_gate, k, v = qkv.split([attn.q_size * 2, attn.kv_size, attn.kv_size], dim=-1)
    rope = attn.rotary_emb
    if positions.ndim == 2:
        # a decode token is text: its M-RoPE positions are all equal
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
    return attn.attn(q, k, v), gate
