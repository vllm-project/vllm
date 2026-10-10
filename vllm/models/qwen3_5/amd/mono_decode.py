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
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.platforms import current_platform

logger = init_logger(__name__)


CDNA_VERSIONS = (4,)


def enabled() -> bool:
    return envs.VLLM_ROCM_MONO_DECODE and current_platform.is_rocm()


def _routed_experts(mlp):
    return getattr(mlp.experts, "routed_experts", mlp.experts)


def _refusals(vllm_config) -> list[str]:
    """Every reason the kernels cannot run this deployment: an explicit opt-in
    that cannot run is an error."""
    from vllm.platforms.rocm import get_cdna_version

    pc = vllm_config.parallel_config
    why = []
    if get_cdna_version() not in CDNA_VERSIONS:
        why.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
    if get_tp_group().world_size != L.TP:
        why.append(f"needs tensor parallel size {L.TP}")
    if get_pp_group().world_size != 1:
        why.append("needs no pipeline parallelism")
    if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
        why.append("needs tensor parallelism only: no expert or data parallelism")
    if pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1:
        why.append("needs no context parallelism")
    if vllm_config.lora_config is not None:
        why.append("needs no LoRA")
    if vllm_config.speculative_config is not None:
        why.append("needs no speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        # a KV connector's copies on another stream can hold CUs a resident
        # grid spin-waits on
        why.append("needs no KV connector")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < L.BLOCKS:
        # every CTA of a launch stays resident: a partitioned GPU deadlocks
        why.append(f"needs {L.BLOCKS} compute units, the GPU has {cus}")
    return why


def _unsupported(model) -> str | None:
    """Why the kernels cannot serve this model variant, else None."""
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
            rope = a.rotary_emb
            if not (
                a.attn_output_gate
                and getattr(rope, "is_neox_style", False)
                and getattr(rope, "dtype", None) in (torch.float16, torch.bfloat16)
            ):
                return "attention without the gated NeoX RoPE front"
    return None


class MonoDecode:
    """Bound to one ``Qwen3_5Model``; built at model init on every rank."""

    def __init__(self, model, vllm_config):
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.qwen3_5.amd.mono import gdn, layer

        self.model = model
        refused = _refusals(vllm_config)
        if refused:
            raise ValueError(f"VLLM_ROCM_MONO_DECODE {'; '.join(refused)}.")
        why = _unsupported(model)
        self.ok = why is None
        self._weights_checked = False
        if not self.ok:
            logger.info_once("Qwen3.8 mono decode off for this model: %s", why)
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
        # compile the attention front's helpers here: a decode graph's first
        # call can be its capture
        with torch.inference_mode():
            for s in (1, 2):
                x = torch.zeros(s, L.HIDDEN, dtype=torch.bfloat16, device=dev)
                o = torch.zeros(s, L.CORE, dtype=torch.bfloat16, device=dev)
                _gemma_norm(model.norm, x, x)
                _sigmoid_gate(o, o)
        logger.info_once(
            "Qwen3.8 mono decode on for decode steps of <= %d rows", L.MAX_TOKENS
        )

    def check_weights(self) -> None:
        """At the first decode step, after loading: every weight in the dtype
        and layout the kernels read, else an error."""
        if not self._weights_checked:
            why = self._weights_unsupported()
            if why is not None:
                raise ValueError(f"VLLM_ROCM_MONO_DECODE {why}.")
            self._weights_checked = True

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
        self.check_weights()
        return True

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
                x, res_attn = _gemma_norm(layer.input_layernorm, h, residual)
                core = _attention_core(layer.self_attn, x, positions)
            h, residual = self._k2(layer, core, res_attn)
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


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _sigmoid_gate(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return o * torch.sigmoid(gate)


def _gemma_norm(norm, x, residual):
    if residual is None:
        return norm(x), x
    return _gemma_add_rms_norm(x, residual, norm.weight, norm.variance_epsilon)


def _attention_core(attn, x, positions) -> torch.Tensor:
    """Qwen3NextAttention.forward up to o_proj's input, with the split, q / k
    norms, RoPE and gate copy in one launch as the model's CUDA path runs them."""
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
    return _sigmoid_gate(attn.attn(q, k, v), gate)
