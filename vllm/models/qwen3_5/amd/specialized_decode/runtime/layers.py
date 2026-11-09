# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launches of one decoder layer, vLLM's modules and tensors in, the
kernels' arguments out, and the parts of a layer that stay vLLM's: a full
attention layer's input norm and attention up to o_proj's input
(``attention_core``) and the final norm (``gemma_norm``). With the MXFP6
projections, the attention front kernel (``mono/afront.py``) runs that
layer's input norm, qkv_proj, q / k norms, RoPE and the FP8 KV cache write
instead (``Launcher.attention_front``). ``Launcher`` holds the per-rank state
every launch reads: the scratch of each width, the peer buffer, the step
epoch and the health words."""

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.models.qwen3_5.amd.specialized_decode.mono import afront, blocks
from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.runtime.guards import (
    gdn_conv_state,
    mxfp4_shuffled,
    o_proj,
    routed_experts,
)
from vllm.models.qwen3_5.amd.specialized_decode.runtime.mxfp6 import (
    fp6_copy,
    rm_weight,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


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
        # The mailbox through which the attention front kernel hands q and k
        # to the CTAs that run their norms and RoPE.
        self.af_scratch = _buf(afront.SCRATCH_BYTES, dev)

    def gdn(self, layer, md, h, residual):
        """A GDN layer's residual add + input_layernorm and ``linear_attn`` up
        to out_proj's input -> (core [S, CORE], the residual)."""
        a = layer.linear_attn
        conv_state, rstate = gdn_conv_state(a)
        s = h.size(0)
        core = h.new_empty(s, L.CORE)
        res_out = torch.empty_like(h)
        fp6 = fp6_copy(a.in_proj_qkvz)
        if fp6 is not None:
            w_qkvz, w_qkvzs = fp6
        else:
            w_qkvz, w_qkvzs = a.in_proj_qkvz.weight, a.in_proj_qkvz.weight_scale
        blocks.gdn_block(
            self.gdn_key(layer, s, residual is None),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=w_qkvz,
            w_qkvzs=w_qkvzs,
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
        fp6 = fp6_copy(o)
        if fp6 is not None:
            w_o, w_os = fp6
        else:
            w_o, w_os = o.weight, o.weight_scale if key.oq else None
        blocks.layer_block(
            key,
            core=core,
            gate=gate,
            residual=residual,
            w_o=w_o,
            w_os=w_os,
            w_rm=rm_weight(mlp) if key.rm else None,
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

    def attention_front(self, layer, h, residual, positions):
        """A full-attention layer with an MXFP6 qkv_proj up to o_proj's input:
        hidden + residual -> input_layernorm -> the MXFP6 qkv_proj -> the q
        and k norms, RoPE and the split, and the FP8 KV cache write when
        ``_af_kv`` allows it, in one launch -> the attention -> (attention
        output, gate, hidden + residual)."""
        attn = layer.self_attn
        fp6 = fp6_copy(attn.qkv_proj)
        assert fp6 is not None
        if positions.ndim == 2:
            # a decode token is text: its M-RoPE positions are all equal
            positions = positions[0]
        s = h.size(0)
        q = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
        gate = torch.empty_like(q)
        k = torch.empty(s, afront.HEAD, dtype=h.dtype, device=h.device)
        v = torch.empty_like(k)
        res_out = torch.empty_like(h)
        kv = _af_kv(attn.attn, s)
        cache = {}
        if kv is not None:
            names = ("k_cache", "v_cache", "slot_mapping", "k_scale", "v_scale")
            cache = dict(zip(names, kv))
        afront.attn_front(
            self.af_key(layer, s, None if kv is None else kv[0].size(1)),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w=fp6[0],
            ws=fp6[1],
            q_out=q,
            k_out=k,
            v_out=v,
            gate_out=gate,
            qn_w=attn.q_norm.weight,
            kn_w=attn.k_norm.weight,
            cos_sin=attn.rotary_emb.cos_sin_cache,
            positions=positions,
            scratch=self.af_scratch,
            epoch=self.epoch,
            # The attention front has its own scratch, so the layer's index
            # tells its launches of one step apart.
            slot=layer.layer_idx,
            health=self.health,
            alarm=self.alarm,
            **cache,
        )
        if kv is None:
            return attn.attn(q, k, v), gate, res_out
        return _attention_q_only(attn.attn, q), gate, res_out

    def af_key(self, layer, s: int, page: int | None) -> afront.AFBuild:
        """The attention front build of ``layer`` at ``s`` rows. ``page`` is
        the tokens of a block of the KV cache views that the kernel writes,
        None when vLLM writes the cache."""
        attn = layer.self_attn
        rope = attn.rotary_emb
        return afront.AFBuild(
            tokens=s,
            eps=layer.input_layernorm.variance_epsilon,
            qk=True,
            kv=page is not None,
            qk_eps=attn.q_norm.variance_epsilon,
            rot=rope.rotary_dim,
            cs_f32=rope.cos_sin_cache.dtype == torch.float32,
            page=64 if page is None else page,
        )

    # the build keys, also what ``SpecializedDecode`` compiles at build
    def gdn_key(self, layer, s: int, first: bool) -> blocks.GdnBuild:
        a = layer.linear_attn
        q6 = fp6_copy(a.in_proj_qkvz) is not None
        return blocks.GdnBuild(
            tokens=s,
            first=first,
            eps=layer.input_layernorm.variance_epsilon,
            norm_eps=a.norm.eps,
            shuffled=q6 or bool(mxfp4_shuffled(a.in_proj_qkvz)),
            q6=q6,
        )

    def layer_key(self, layer, s: int) -> blocks.LayerBuild:
        shuffled = mxfp4_shuffled(o_proj(layer))
        oq = shuffled is not None
        return blocks.LayerBuild(
            tokens=s,
            eps=layer.post_attention_layernorm.variance_epsilon,
            oq=oq,
            og=layer.layer_type != "linear_attention",
            shuffled=shuffled is not False,
            o6=fp6_copy(o_proj(layer)) is not None,
            # rm needs the layer's rm weight, which is made from the MXFP6
            # o_proj values, so an MXFP4 o_proj runs without it. From 6 rows
            # the router tasks after the norm are faster (the layer stage
            # benchmark).
            rm=not oq and s <= blocks.RM_MAX and rm_weight(layer.mlp) is not None,
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


def _af_kv(attn_layer, s):
    """(k_cache, v_cache, slot_mapping, k_scale, v_scale) for the KV cache
    write in the attention front kernel, or None when vLLM must write the
    cache. The kernel writes reshape_and_cache_flash's FP8 e4m3 values into
    the unshuffled [blocks, block_size, 1, 256] views, one slot for each
    row."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends import rocm_aiter_fa

    why = None
    if not isinstance(attn_layer.impl, rocm_aiter_fa.AiterFlashAttentionImpl):
        why = f"the attention backend {type(attn_layer.impl).__name__}"
    elif (
        attn_layer.kv_cache_dtype not in ("fp8", "fp8_e4m3")
        or current_platform.fp8_dtype() != torch.float8_e4m3fn
    ):
        why = f"the KV cache dtype {attn_layer.kv_cache_dtype}"
    elif rocm_aiter_ops.is_shuffle_kv_cache_enabled():
        why = "the shuffled KV cache layout"
    elif attn_layer.kv_sharing_target_layer_name is not None:
        why = "a KV cache shared with another layer"
    elif attn_layer.query_quant is not None:
        why = "a quantized query"
    slots = None
    if why is None:
        mapping = get_forward_context().slot_mapping
        if isinstance(mapping, dict):
            slots = mapping.get(attn_layer.layer_name)
        # vLLM can leave the pad rows of the batch out of slot_mapping. The
        # kernel would then read slots past its end.
        if slots is None or slots.dtype != torch.int64 or slots.shape != (s,):
            shape = None if slots is None else tuple(slots.shape)
            why = f"a slot_mapping of shape {shape} for {s} rows"
    if why is None:
        k_cache, v_cache = attn_layer.impl._split_kv_cache(attn_layer.kv_cache)
        if k_cache.shape[2:] != (1, 256) or k_cache.element_size() != 1:
            why = f"KV cache views of shape {tuple(k_cache.shape)}"
    if why is not None:
        logger.info_once(
            "Qwen3.8 specialized decode: vLLM writes the KV cache (%s)", why
        )
        return None
    logger.info_once(
        "Qwen3.8 specialized decode: attn_front writes the FP8 KV cache (blocks "
        "of %d tokens)",
        k_cache.size(1),
    )
    return k_cache, v_cache, slots, attn_layer._k_scale, attn_layer._v_scale


def _attention_q_only(attn_layer, q) -> torch.Tensor:
    """Attention.forward for a q whose k and v the attention front kernel
    already wrote into the KV cache. It is the direct call of
    Attention.forward with key and value None, which skips vLLM's KV cache
    write, and which the backend takes as a layer that projects only q. The
    custom op that Attention.forward calls on ROCm takes no None for key and
    value, so this calls the op's Python function."""
    from vllm.model_executor.layers.attention.attention import (
        unified_attention_with_output,
    )

    out = torch.empty(
        q.size(0),
        attn_layer.num_heads * attn_layer.head_size_v,
        dtype=q.dtype,
        device=q.device,
    )
    unified_attention_with_output(
        q.view(-1, attn_layer.num_heads, attn_layer.head_size),
        None,
        None,
        out.view(-1, attn_layer.num_heads, attn_layer.head_size_v),
        attn_layer.layer_name,
    )
    return out
