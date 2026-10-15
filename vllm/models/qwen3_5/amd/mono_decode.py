# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM side of the Qwen3.8 mono decode (``VLLM_ROCM_MONO_DECODE=1``, MI355X, TP8).

A pure decode step of at most ``MAX_TOKENS`` rows runs each decoder layer as
two persistent launches:

- K1 (``gdn_pre``, GDN layers): the residual add + input_layernorm ->
  in_proj_qkvz / in_proj_ba -> conv1d -> the gated delta rule -> the gated
  norm, writing out_proj's input.
- K2 (``layer_post``, every layer): out_proj / o_proj + its TP all-reduce + the
  residual add + post_attention_layernorm + the MoE (router, top-10, MXFP4
  experts, the gated shared expert) + the MoE all-reduce. A full-attention
  layer runs vLLM's attention up to o_proj's input, then K2.

The kernels read the weights as the stock flags leave them: the shared expert
bf16 or fused as MXFP4 expert 512 (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``);
in_proj_qkvz and out_proj / o_proj bf16 or online MXFP4 on the ASM FP4 GEMM
(``VLLM_ROCM_USE_AITER_FP4_ASM_GEMM``).

With MTP (``mono_base``), a GDN layer of a verify step runs the input norm and
vLLM's GDN core instead of K1, then K2; the drafter reads the final norm's
output, as from the stock model.

The step is chosen whole in the model's outer forward, which no torch.compile
traces; a FULL decode graph captures it per width. Any other step (prefill,
mixed, more rows) takes the compiled model.

The all-reduces run in-kernel over a peer buffer every TP rank maps. Mailbox
tags carry a device epoch bumped once a step (captured with it), so no buffer
is zeroed between steps. Each width and kernel has its own scratch.
"""

from itertools import islice

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen3_5.amd import mono_base as B
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.platforms import current_platform

enabled = B.enabled


class MonoDecode(B.QwenMonoDecode):
    """Bound to one ``Qwen3_5Model``; built at model init on every rank."""

    LABEL = "Qwen3.8"
    TP_SIZES = (L.TP,)
    MAX_TOKENS = L.MAX_TOKENS
    BLOCKS = L.BLOCKS
    MTP = True

    def unsupported(self, model) -> str | None:
        from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

        c = model.config
        shape = (
            c.hidden_size == L.HIDDEN
            and c.linear_num_key_heads == L.NK * L.TP
            and c.linear_num_value_heads == L.NV * L.TP
            and c.linear_key_head_dim == L.HD
            and c.linear_value_head_dim == L.HD
            and c.linear_conv_kernel_dim == L.CONV_W
            and c.num_experts == L.E
            and c.num_experts_per_tok == L.TOPK
            and c.moe_intermediate_size == L.RI * L.TP
            and c.shared_expert_intermediate_size == L.SI * L.TP
            and c.hidden_act == "silu"
            and getattr(c, "norm_topk_prob", True)
        )
        if not shape:
            return "not the Qwen3.8 shape"
        if model.start_layer != 0 or model.end_layer != len(model.layers):
            return "partial layers"
        for layer in model.layers:
            mlp = layer.mlp
            if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
                return "layer scale or sequence parallel"
            if not isinstance(mlp, Qwen3NextSparseMoeBlock):
                return "dense MLP"
            fused = mlp.is_fused_shared_expert_enabled
            if not fused and (mlp.shared_expert is None or mlp.replicate_shared_expert):
                return "shared expert not TP sharded"
            if layer.layer_type == "linear_attention":
                if not B.gdn_variant(layer.linear_attn, ("silu", "swish")):
                    return "GDN layer variant"
            else:
                a = layer.self_attn
                if a.num_heads * a.head_dim != L.CORE:
                    return "attention width"
                rope = a.rotary_emb
                if not (
                    a.attn_output_gate
                    and getattr(rope, "is_neox_style", False)
                    and getattr(rope, "dtype", None) in (torch.float16, torch.bfloat16)
                ):
                    return "attention without the gated NeoX RoPE front"
        return None

    def buffer_bytes(self):
        from vllm.models.qwen3_5.amd.mono import gdn, layer

        def k1(s):
            return max(
                gdn.scratch_bytes(gdn.K1Build(tokens=s, first=f, qq=q))
                for f in (False, True)
                for q in (False, True)
            )

        def k2(s):
            return max(
                layer.scratch_bytes(layer.K2Build(tokens=s, fse=f))
                for f in (False, True)
            )

        return k1, k2, layer.peer_bytes(L.MAX_TOKENS)

    def warm_up(self, dev) -> None:
        with torch.inference_mode():
            for s in (1, 2):
                x = torch.zeros(s, L.HIDDEN, dtype=torch.bfloat16, device=dev)
                _gemma_norm(self.model.norm, x, x)

    def weights_unsupported(self) -> str | None:
        for layer in self.model.layers:
            i = layer.layer_idx
            mlp = layer.mlp
            fused = mlp.is_fused_shared_expert_enabled
            why = B.experts_error(B.routed_experts(mlp), L.E + int(fused))
            if why is not None:
                return why
            dense: dict[str, torch.Tensor] = {}
            if layer.layer_type == "linear_attention":
                a = layer.linear_attn
                if B.asm_mxfp4(a.in_proj_qkvz):
                    if a.in_proj_qkvz.weight.shape != (L.QKVZ, L.HIDDEN // 2):
                        return f"layer {i}: in_proj_qkvz shape"
                else:
                    dense["in_proj_qkvz"] = a.in_proj_qkvz.weight
                dense |= {
                    "in_proj_ba": a.in_proj_ba.weight,
                    "conv1d": a.conv1d.weight,
                    "dt_bias": a.dt_bias,
                    "norm": a.norm.weight,
                }
                if a.A_log.dtype != torch.float32:
                    return "A_log not fp32"
            o = B.o_proj(layer)
            fp4_o = B.asm_mxfp4(o)
            if o.weight.shape != (L.HIDDEN, L.CORE // 2 if fp4_o else L.CORE):
                return f"layer {i}: o_proj shape {tuple(o.weight.shape)}"
            if not fp4_o:
                dense["o_proj"] = o.weight
            dense |= {
                "input_layernorm": layer.input_layernorm.weight,
                "post_attention_layernorm": layer.post_attention_layernorm.weight,
                "gate": mlp.gate.weight,
                "shared_expert_gate": mlp.shared_expert_gate.weight,
            }
            if not fused:
                dense["shared_expert.gate_up_proj"] = (
                    mlp.shared_expert.gate_up_proj.weight
                )
                dense["shared_expert.down_proj"] = mlp.shared_expert.down_proj.weight
            why = B.dense_error(i, dense)
            if why is not None:
                return why
        return None

    def eligible(self, input_ids, intermediate_tensors, inputs_embeds) -> bool:
        if inputs_embeds is not None or self.model.aux_hidden_state_layers:
            return False
        return super().eligible(input_ids, intermediate_tensors)

    def forward(self, input_ids, positions) -> torch.Tensor:
        model = self.model
        s = input_ids.size(0)
        self.first_step(s)
        self.epoch.add_(1)
        md = get_forward_context().attn_metadata
        assert isinstance(md, dict)
        h = model.embed_input_ids(input_ids)
        residual: torch.Tensor | None = None
        for layer in islice(model.layers, model.start_layer, model.end_layer):
            gate = None
            if layer.layer_type == "linear_attention" and not self.spec:
                core = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
                res_attn = torch.empty_like(h)
                self._k1(layer, md, h, residual, res_attn, core)
            else:
                x, res_attn = _gemma_norm(layer.input_layernorm, h, residual)
                if layer.layer_type == "linear_attention":
                    core = B.gdn_core(layer.linear_attn, x)
                else:
                    core, gate = _attention_core(layer.self_attn, x, positions)
            h, residual = self._k2(layer, core, res_attn, gate)
        h, _ = _gemma_norm(model.norm, h, residual)
        return h

    def _k1(self, layer, md, h, residual, res_out, core) -> None:
        from vllm.models.qwen3_5.amd.mono import gdn

        a = layer.linear_attn
        m = md[a.prefix]
        conv_state, rstate = a.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        s = h.size(0)
        qq = B.asm_mxfp4(a.in_proj_qkvz)
        gdn.gdn_pre(
            gdn.K1Build(
                tokens=s,
                first=residual is None,
                eps=layer.input_layernorm.variance_epsilon,
                norm_eps=a.norm.eps,
                qq=qq,
            ),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=a.in_proj_qkvz.weight,
            w_qkvzs=a.in_proj_qkvz.weight_scale if qq else None,
            w_ba=a.in_proj_ba.weight,
            conv_w=a.conv1d.weight,
            conv_state=conv_state,
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=rstate,
            st_idx=m.non_spec_state_indices_tensor,
            core=core,
            scratch=self.k1_scratch[s],
            epoch=self.epoch,
            layer=layer.layer_idx,
        )

    def _k2(self, layer, core, residual, gate) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.models.qwen3_5.amd.mono import layer as k2

        mlp = layer.mlp
        ex = B.routed_experts(mlp)
        o = B.o_proj(layer)
        oq = B.asm_mxfp4(o)
        fused = mlp.is_fused_shared_expert_enabled
        s = core.size(0)
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        k2.layer_post(
            k2.K2Build(
                tokens=s,
                eps=layer.post_attention_layernorm.variance_epsilon,
                fse=fused,
                oq=oq,
                og=gate is not None,
            ),
            core=core,
            gate=gate,
            residual=residual,
            w_o=o.weight,
            w_os=o.weight_scale if oq else None,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w_sgu=None if fused else mlp.shared_expert.gate_up_proj.weight,
            w_sd=None if fused else mlp.shared_expert.down_proj.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.k2_scratch[s],
            peers=self.peers.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=k2.TAG_SLOTS // 2 + layer.layer_idx,
        )
        return out, res_out


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _gemma_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``GemmaRMSNorm.forward_native(x, residual)`` in one launch: this forward
    is not torch.compiled, so the CustomOp runs as ~10 eager kernels."""
    t = x.float() + residual.float()
    res = t.to(x.dtype)
    t = t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (t * (weight.float() + 1.0)).to(x.dtype), res


def _gemma_norm(norm, x, residual):
    if residual is None:
        return norm(x), x
    return _gemma_add_rms_norm(x, residual, norm.weight, norm.variance_epsilon)


def _attention_core(attn, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3NextAttention.forward up to the output gate (K2 applies it to
    o_proj's input), with the split, q / k norms, RoPE and gate copy in one
    launch as the model's CUDA path runs them -> (attention output, gate)."""
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
