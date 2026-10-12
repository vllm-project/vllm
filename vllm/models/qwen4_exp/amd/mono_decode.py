# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM side of the Qwen3.8-Flash-Next mono decode (``VLLM_ROCM_MONO_DECODE=1``,
MI355X, TP2).

A pure decode step of at most ``MAX_TOKENS`` rows runs each decoder layer as
two persistent launches on the materialized multi-stream state:

- K1 (``gdn_pre``, GDN layers): the attention HC's norm and mix -> in_proj_qkvz
  / in_proj_ba -> conv1d -> the gated delta rule -> the gated norm, writing
  out_proj's input. A QSA layer runs K1's HC front alone (``hc_front``), then
  vLLM's QSA attention up to o_proj's input.
- K2 (``layer_post``, every layer): out_proj / o_proj + its TP all-reduce + the
  attention HC combine, the MLP HC's norm and mix, the MoE (router, top-10, the
  MXFP4 experts with the shared expert fused as expert 512) + the MoE
  all-reduce + the MLP HC combine, writing the next state and its sums of
  squares (the next front's norm).

Stock defers each HC combine to the next mix; K2 materializes it with the same
bf16 rounding, so the PLE layer and the final mixer read a plain state.

The step is chosen whole in the model's outer forward, which no torch.compile
traces; a FULL decode graph captures it per width. Any other step (prefill,
mixed, spec decode, more rows) takes the compiled model.

The all-reduces run in-kernel over a peer buffer every TP rank maps. Mailbox
tags carry a device epoch bumped once a step (captured with it), so no buffer
is zeroed between steps. Each width and kernel has its own scratch.
"""

from typing import Any

import torch

from vllm import envs
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen4_exp.amd.mono import layout as L
from vllm.platforms import current_platform

logger = init_logger(__name__)


CDNA_VERSIONS = (4,)


def enabled() -> bool:
    return envs.VLLM_ROCM_MONO_DECODE and current_platform.is_rocm()


def _asm_mxfp4(linear) -> bool:
    """An online-MXFP4 linear on AITER's ASM FP4 GEMM: its weight and scales
    shuffled as the kernels' fp4 GEMVs read them."""
    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    qm = linear.quant_method
    return (
        isinstance(qm, Mxfp4OnlineLinearMethod)
        and isinstance(qm.kernel, AiterMxfp4LinearKernel)
        and qm.kernel.use_asm_gemm
    )


def _o_proj(layer):
    if layer.layer_type == "linear_attention":
        return layer.linear_attn.out_proj
    return layer.self_attn.o_proj


def _refusals(vllm_config) -> list[str]:
    """Every reason the kernels cannot run this deployment: an explicit opt-in
    that cannot run is an error."""
    from vllm.platforms.rocm import get_cdna_version

    pc = vllm_config.parallel_config
    why = []
    if get_cdna_version() not in CDNA_VERSIONS:
        why.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
    if get_tp_group().world_size not in L.TPS:
        why.append(f"needs tensor parallel size {'/'.join(map(str, L.TPS))}")
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
    if not envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:
        why.append("needs VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < L.BLOCKS:
        # every CTA of a launch stays resident: a partitioned GPU deadlocks
        why.append(f"needs {L.BLOCKS} compute units, the GPU has {cus}")
    return why


def _unsupported(model) -> str | None:
    """Why the kernels cannot serve this model variant, else None."""
    from vllm.models.qwen4_exp.amd.model import Qwen4ExpSparseMoeBlock
    from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention

    c = model.config
    shape = (
        c.hidden_size == L.HIDDEN
        and c.hc_count == L.HC
        and c.hc_lowrank == L.LORA
        and c.linear_num_key_heads == 16
        and c.linear_num_value_heads == 48
        and c.linear_key_head_dim == L.HD
        and c.linear_value_head_dim == L.HD
        and c.linear_conv_kernel_dim == L.CONV_W
        and c.num_experts == L.E
        and c.num_experts_per_tok == L.TOPK
        and c.moe_intermediate_size == L.MOE_I
        and c.shared_expert_intermediate_size == L.MOE_I
        and c.hidden_act == "silu"
        and getattr(c, "norm_topk_prob", True)
    )
    if not shape:
        return "not the Qwen3.8-Flash-Next shape"
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        return "partial layers"
    if model._mtp_hidden_buffer is not None:
        return "MTP hidden states"
    mixer = model.hyper_connection_mixer
    if mixer is None or mixer.use_combine:
        return "no final HC mixer"
    for layer in model.layers:
        mlp = layer.mlp
        if not isinstance(mlp, Qwen4ExpSparseMoeBlock):
            return "dense MLP"
        if not mlp.is_fused_shared_expert_enabled:
            return "shared expert not fused"
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            if not hc.use_combine or hc.pad_size != L.DOWN_ROWS - L.LORA - L.HC:
                return "HC variant"
        if layer.layer_type == "linear_attention":
            a = layer.linear_attn
            gdn = (
                hasattr(a, "in_proj_qkvz")
                and not a.disable_tp_for_ba_proj
                and a.qkvz_layout == "flat"
                and a.activation == "silu"
                and a.conv1d.bias is None
                and a.norm.activation == "sigmoid"
                and a.norm.group_size is None
                and a.norm.norm_before_gate
            )
            if not gdn:
                return "GDN layer variant"
        else:
            a = layer.self_attn
            if not isinstance(a, Qwen4ExpQSAAttention):
                return "attention without the QSA indexer"
            if a.num_heads * a.head_dim != L.Layout(get_tp_group().world_size).CORE:
                return "attention width"
            if not a.attn_output_gate:
                return "attention without the output gate"
    return None


class MonoDecode:
    """Bound to one ``Qwen4ExpModel``; built at model init on every rank."""

    def __init__(self, model, vllm_config):
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.qwen4_exp.amd.mono import k1, k2

        self.model = model
        refused = _refusals(vllm_config)
        if refused:
            raise ValueError(f"VLLM_ROCM_MONO_DECODE {'; '.join(refused)}.")
        why = _unsupported(model)
        self.ok = why is None
        self._weights_checked = False
        if not self.ok:
            logger.info_once(
                "Qwen3.8-Flash-Next mono decode off for this model: %s", why
            )
            return
        tp = get_tp_group()
        self.tp = tp.world_size
        self.lay = L.Layout(self.tp)
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
                    k1.scratch_bytes(
                        k1.K1Build(tokens=s, tp=self.tp, self_msq=m, front_only=f)
                    )
                    for m in (False, True)
                    for f in (False, True)
                )
            )
            for s in widths
        }
        self.k2_scratch = {
            s: buf(k2.scratch_bytes(k2.K2Build(tokens=s, tp=self.tp))) for s in widths
        }
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.rank = tp.rank_in_group
        # collective: every rank builds the model, so every rank gets here
        self.peers = PeerBuffer(
            k2.peer_bytes(L.MAX_TOKENS, self.tp),
            tp.cpu_group,
            tp.rank_in_group,
            tp.world_size,
            dev,
        )
        self.peers.bytes.zero_()
        logger.info_once(
            "Qwen3.8-Flash-Next mono decode on for decode steps of <= %d rows",
            L.MAX_TOKENS,
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
        lay = self.lay
        fp4s = (torch.uint8, torch.float4_e2m1fn_x2)
        for layer in self.model.layers:
            i = layer.layer_idx
            mlp = layer.mlp
            ex = mlp.experts.routed_experts
            backend = getattr(ex.quant_method, "mxfp4_backend", None)
            if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
                return f"routed experts on {backend}, not AITER_MXFP4_MXFP4"
            if not getattr(ex.w13_weight, "is_shuffled", False):
                return "routed experts not shuffled"
            if ex.w13_weight.shape != (L.E + 1, 2 * lay.RIP, L.HIDDEN // 2):
                return f"layer {i}: w13 {tuple(ex.w13_weight.shape)}"
            if ex.w2_weight.shape != (L.E + 1, L.HIDDEN, lay.RIP // 2):
                return f"layer {i}: w2 {tuple(ex.w2_weight.shape)}"
            dense: dict[str, torch.Tensor] = {
                "gate": mlp.gate.weight,
                "shared_expert_gate": mlp.shared_expert_gate.weight,
            }
            for name in ("attn_hyper_connection", "mlp_hyper_connection"):
                hc = getattr(layer, name)
                dense[f"{name}.hc_norm"] = hc.hc_norm.weight
                dense[f"{name}.down"] = hc.input_mix_weight_down_block_inject.weight
                dense[f"{name}.up"] = hc.input_mix_weight_up.weight
            if layer.layer_type == "linear_attention":
                a = layer.linear_attn
                if not _asm_mxfp4(a.in_proj_qkvz):
                    return f"layer {i}: in_proj_qkvz not online MXFP4 on the ASM GEMM"
                dense |= {
                    "in_proj_ba": a.in_proj_ba.weight,
                    "conv1d": a.conv1d.weight,
                    "dt_bias": a.dt_bias,
                    "norm": a.norm.weight,
                }
                if a.A_log.dtype != torch.float32:
                    return "A_log not fp32"
            o = _o_proj(layer)
            if _asm_mxfp4(o):
                if (
                    o.weight.shape != (L.HIDDEN, lay.CORE // 2)
                    or o.weight.dtype not in fp4s
                ):
                    return f"layer {i}: o_proj {tuple(o.weight.shape)}"
            else:
                dense["o_proj"] = o.weight
            for name, w in dense.items():
                if w.dtype != bf:
                    return (
                        f"layer {i}: {name} is {w.dtype}, neither bf16 nor in a "
                        "quantized layout the kernels read"
                    )
        return None

    def eligible(self, input_ids, intermediate_tensors):
        """A pure decode step of at most ``MAX_TOKENS`` rows."""
        if not self.ok or intermediate_tensors is not None:
            return False
        if input_ids is None or not 1 <= input_ids.size(0) <= L.MAX_TOKENS:
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

    def forward(
        self, input_ids, positions, inputs_embeds, query_start_loc, ngram_context
    ):
        """``inputs_embeds``: the multimodal wrapper's embeddings of the step's
        tokens (decode rows are sampled text tokens, so the token embeddings),
        else None."""
        model = self.model
        s = input_ids.size(0)
        logger.info_once("Qwen3.8-Flash-Next mono decode: first mono step (%d rows)", s)
        self.epoch.add_(1)
        md = get_forward_context().attn_metadata
        assert isinstance(md, dict)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = inputs_embeds.repeat(1, L.HC)
        msq: torch.Tensor | None = None
        for layer in model.layers:
            if layer.ple is not None:
                state = layer.ple(state, input_ids, query_start_loc, ngram_context)
                msq = None
            inj = torch.empty(s, L.HC, dtype=state.dtype, device=state.device)
            if layer.layer_type == "linear_attention":
                core = torch.empty(
                    s, self.lay.CORE, dtype=state.dtype, device=state.device
                )
                self._k1(layer, md, state, msq, inj, core)
                gate = None
            else:
                x = torch.empty(s, L.HIDDEN, dtype=state.dtype, device=state.device)
                self._front(layer, state, msq, inj, x)
                core, gate = _qsa_core(layer.self_attn, x, positions)
            state, msq = self._k2(layer, core, gate, state, inj)
        _, h, _ = model.hyper_connection_mixer.mix(state)
        return h

    def _hc_args(self, hc, state, msq, inj) -> dict:
        return {
            "state": state,
            "msq_in": msq,
            "hc_w": hc.hc_norm.weight,
            "w_down": hc.input_mix_weight_down_block_inject.weight,
            "w_up": hc.input_mix_weight_up.weight,
            "inj_out": inj,
        }

    def _front(self, layer, state, msq, inj, x) -> None:
        from vllm.models.qwen4_exp.amd.mono import k1

        hc = layer.attn_hyper_connection
        s = state.size(0)
        k1.hc_front(
            k1.K1Build(
                tokens=s,
                tp=self.tp,
                self_msq=msq is None,
                front_only=True,
                eps=hc.config.rms_norm_eps,
            ),
            **self._hc_args(hc, state, msq, inj),
            bin_out=x,
            scratch=self.k1_scratch[s],
            epoch=self.epoch,
            layer=layer.layer_idx,
        )

    def _k1(self, layer, md, state, msq, inj, core) -> None:
        from vllm.models.qwen4_exp.amd.mono import k1

        a = layer.linear_attn
        hc = layer.attn_hyper_connection
        m = md[a.prefix]
        conv_state, rstate = a.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        s = state.size(0)
        k1.gdn_pre(
            k1.K1Build(
                tokens=s,
                tp=self.tp,
                self_msq=msq is None,
                eps=hc.config.rms_norm_eps,
                norm_eps=a.norm.eps,
            ),
            **self._hc_args(hc, state, msq, inj),
            w_qkvz=a.in_proj_qkvz.weight,
            w_qkvzs=a.in_proj_qkvz.weight_scale,
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

    def _k2(self, layer, core, gate, state, inj) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.models.qwen4_exp.amd.mono import hc as mhc
        from vllm.models.qwen4_exp.amd.mono import k2

        mlp = layer.mlp
        ex = mlp.experts.routed_experts
        hc = layer.mlp_hyper_connection
        o = _o_proj(layer)
        oq = _asm_mxfp4(o)
        s = core.size(0)
        out = torch.empty_like(state)
        msq = torch.empty(mhc.NRED, s, L.HC, dtype=torch.float32, device=state.device)
        k2.layer_post(
            k2.K2Build(
                tokens=s,
                tp=self.tp,
                eps=hc.config.rms_norm_eps,
                oq=oq,
                og=gate is not None,
            ),
            core=core,
            gate=gate,
            state=state,
            inj_a=inj,
            w_o=o.weight,
            w_os=o.weight_scale if oq else None,
            hc_w=hc.hc_norm.weight,
            w_down=hc.input_mix_weight_down_block_inject.weight,
            w_up=hc.input_mix_weight_up.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            msq_out=msq,
            scratch=self.k2_scratch[s],
            peers=self.peers.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=k2.TAG_SLOTS // 2 + layer.layer_idx,
        )
        return out, msq


def _qsa_core(attn, x, positions) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``Qwen4ExpQSAAttention.forward`` up to the output gate (K2 applies it to
    o_proj's input) -> (attention output [S, heads HD], gate or None)."""
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
    return out.view(n, -1), None if gate is None else gate.contiguous()
