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

The step is chosen whole in the model's outer forward, which no torch.compile
traces; a FULL decode graph captures it per width. Any other step (prefill,
mixed, spec decode, more rows) takes the compiled model.

The all-reduces run in-kernel over a peer buffer every TP rank maps. Mailbox
tags carry a device epoch bumped once a step (captured with it), so no buffer
is zeroed between steps. Each width and kernel has its own scratch.
"""

from itertools import islice
from typing import Any

import torch

from vllm import envs
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.platforms import current_platform

logger = init_logger(__name__)


def enabled() -> bool:
    if not envs.VLLM_ROCM_MONO_DECODE or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        logger.warning_once("VLLM_ROCM_MONO_DECODE needs gfx950 (MI355X); ignored")
        return False
    from vllm.config import get_current_vllm_config

    if get_current_vllm_config().kv_transfer_config is not None:
        # the kernels spin-wait on every CTA being resident; a KV connector's
        # copy kernels on another stream can hold the CUs and deadlock them
        logger.warning_once("VLLM_ROCM_MONO_DECODE is off with a KV connector")
        return False
    return True


def _routed_experts(mlp):
    return getattr(mlp.experts, "routed_experts", mlp.experts)


def _unsupported(model, vllm_config) -> str | None:
    """Why the kernels cannot serve this model as built, else None."""
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    c = model.config
    if get_tp_group().world_size != L.TP or get_pp_group().world_size != 1:
        return "needs TP8 without PP"
    if vllm_config.lora_config is not None:
        return "LoRA"
    if vllm_config.speculative_config is not None:
        return "spec decode"
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
    if model.is_fused_shared_expert_enabled:
        return "fused shared expert"
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        return "partial layers"
    for layer in model.layers:
        mlp = layer.mlp
        if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
            return "layer scale or sequence parallel"
        if not isinstance(mlp, Qwen3NextSparseMoeBlock):
            return "dense MLP"
        if mlp.shared_expert is None or mlp.replicate_shared_expert:
            return "shared expert not TP sharded"
        if mlp.enable_eplb or _routed_experts(mlp).use_ep:
            return "expert parallel"
        if layer.layer_type == "linear_attention":
            a = layer.linear_attn
            gdn = (
                hasattr(a, "in_proj_qkvz")
                and not a.disable_tp_for_ba_proj
                and a.qkvz_layout == "flat"
                and a.activation == "silu"
                and a.conv1d.bias is None
                and a.norm.activation in ("silu", "swish")
                and a.norm.group_size is None
                and a.norm.norm_before_gate
            )
            if not gdn:
                return "GDN layer variant"
        else:
            a = layer.self_attn
            if a.num_heads * a.head_dim != L.CORE:
                return "attention width"
    return None


class MonoDecode:
    """Bound to one ``Qwen3_5Model``; built at model init on every rank."""

    def __init__(self, model, vllm_config):
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.qwen3_5.amd.mono import gdn, layer

        self.model = model
        why = _unsupported(model, vllm_config)
        self.ok = why is None
        self._weights_ok: bool | None = None
        if not self.ok:
            logger.info_once("Qwen3.8 mono decode off: %s", why)
            return
        self._probe = next(
            ly.linear_attn.prefix
            for ly in model.layers
            if ly.layer_type == "linear_attention"
        )
        dev = model.embed_tokens.weight.device

        def buf(n: int) -> torch.Tensor:
            return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)

        widths = range(1, L.MAX_TOKENS + 1)
        self.k1_scratch = {
            s: buf(
                max(
                    gdn.scratch_bytes(gdn.K1Build(tokens=s, first=f))
                    for f in (False, True)
                )
            )
            for s in widths
        }
        self.k2_scratch = {
            s: buf(layer.scratch_bytes(layer.K2Build(tokens=s))) for s in widths
        }
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        tp = get_tp_group()
        self.rank = tp.rank_in_group
        # collective: every rank builds the model, so every rank gets here
        self.peers = PeerBuffer(
            layer.peer_bytes(L.MAX_TOKENS),
            tp.cpu_group,
            tp.rank_in_group,
            tp.world_size,
            dev,
        )
        self.peers.bytes.zero_()
        logger.info_once(
            "Qwen3.8 mono decode on for decode steps of <= %d rows", L.MAX_TOKENS
        )

    def weights_ok(self) -> bool:
        """After loading: every weight in the dtype and layout the kernels read."""
        if self._weights_ok is None:
            why = self._weights_unsupported()
            if why is not None:
                logger.warning_once("Qwen3.8 mono decode off: %s", why)
            self._weights_ok = why is None
        return self._weights_ok

    def _weights_unsupported(self) -> str | None:
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

        bf = torch.bfloat16
        for layer in self.model.layers:
            mlp = layer.mlp
            ex = _routed_experts(mlp)
            backend = getattr(ex.quant_method, "mxfp4_backend", None)
            if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
                return f"routed experts on {backend}, not AITER_MXFP4_MXFP4"
            if not getattr(ex.w13_weight, "is_shuffled", False):
                return "routed experts not shuffled"
            if ex.w13_weight.shape[0] != L.E:
                return "routed experts not all local"
            if layer.layer_type == "linear_attention":
                a = layer.linear_attn
                dense: tuple[torch.Tensor, ...] = (
                    a.in_proj_qkvz.weight,
                    a.in_proj_ba.weight,
                    a.conv1d.weight,
                    a.dt_bias,
                    a.norm.weight,
                    a.out_proj.weight,
                )
                if a.A_log.dtype != torch.float32:
                    return "A_log not fp32"
                w_o = a.out_proj.weight
            else:
                dense = (layer.self_attn.o_proj.weight,)
                w_o = layer.self_attn.o_proj.weight
            dense += (
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
                mlp.gate.weight,
                mlp.shared_expert_gate.weight,
                mlp.shared_expert.gate_up_proj.weight,
                mlp.shared_expert.down_proj.weight,
            )
            if any(w.dtype != bf for w in dense):
                return f"layer {layer.layer_idx}: a dense weight is not bf16"
            if w_o.shape != (L.HIDDEN, L.CORE):
                return f"layer {layer.layer_idx}: o_proj shape {tuple(w_o.shape)}"
        return None

    def eligible(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        """A pure decode step of at most ``MAX_TOKENS`` rows."""
        if not self.ok or intermediate_tensors is not None or inputs_embeds is not None:
            return False
        if input_ids is None or not 1 <= input_ids.size(0) <= L.MAX_TOKENS:
            return False
        if self.model.aux_hidden_state_layers:
            return False
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return False
        m: Any = md.get(self._probe)
        if m is None or m.spec_sequence_masks is not None:
            return False
        if m.num_prefills != 0 or m.num_spec_decodes != 0 or m.num_decodes < 1:
            return False
        idx = m.non_spec_state_indices_tensor
        s = input_ids.size(0)
        # the graph's pad rows read NULL_BLOCK_ID entries, so the indices cover s
        if idx is None or idx.dtype != torch.int32 or idx.numel() < s:
            return False
        if not idx.is_contiguous():
            return False
        return self.weights_ok()

    def forward(self, input_ids, positions) -> torch.Tensor:
        model = self.model
        s = input_ids.size(0)
        logger.info_once("Qwen3.8 mono decode: first mono step (%d rows)", s)
        self.epoch.add_(1)
        md = get_forward_context().attn_metadata
        assert isinstance(md, dict)
        h = model.embed_input_ids(input_ids)
        residual: torch.Tensor | None = None
        for layer in islice(model.layers, model.start_layer, model.end_layer):
            if layer.layer_type == "linear_attention":
                core = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
                res_attn = torch.empty_like(h)
                self._k1(layer, md, h, residual, res_attn, core)
            else:
                if residual is None:
                    res_attn = h
                    x = layer.input_layernorm(h)
                else:
                    x, res_attn = layer.input_layernorm(h, residual)
                core = _attention_core(layer.self_attn, x, positions)
            h, residual = self._k2(layer, core, res_attn)
        h, _ = model.norm(h, residual)
        return h

    def _k1(self, layer, md, h, residual, res_out, core) -> None:
        from vllm.models.qwen3_5.amd.mono import gdn

        a = layer.linear_attn
        m = md[a.prefix]
        conv_state, rstate = a.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        s = h.size(0)
        gdn.gdn_pre(
            gdn.K1Build(
                tokens=s,
                first=residual is None,
                eps=layer.input_layernorm.variance_epsilon,
                norm_eps=a.norm.eps,
            ),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=a.in_proj_qkvz.weight,
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

    def _k2(self, layer, core, residual) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.models.qwen3_5.amd.mono import layer as k2

        mlp = layer.mlp
        ex = _routed_experts(mlp)
        if layer.layer_type == "linear_attention":
            w_o = layer.linear_attn.out_proj.weight
        else:
            w_o = layer.self_attn.o_proj.weight
        s = core.size(0)
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        k2.layer_post(
            k2.K2Build(tokens=s, eps=layer.post_attention_layernorm.variance_epsilon),
            core=core,
            residual=residual,
            w_o=w_o,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w_sgu=mlp.shared_expert.gate_up_proj.weight,
            w_sd=mlp.shared_expert.down_proj.weight,
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


def _attention_core(attn, x, positions) -> torch.Tensor:
    """Qwen3NextAttention.forward up to o_proj's input."""
    qkv, _ = attn.qkv_proj(x)
    q, k, v, gate = attn._project_qkv_gate(qkv, positions)
    o = attn.attn(q, k, v)
    if gate is not None:
        o = o * torch.sigmoid(gate)
    return o.contiguous()
