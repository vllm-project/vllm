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
(``VLLM_ROCM_USE_AITER_FP4_ASM_GEMM``). With ``VLLM_QWEN38_MONO_FP6=1`` they
read MXFP6 copies of the bf16 in_proj_qkvz and out_proj / o_proj instead
(``pack_fp6``).

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

import os
from itertools import islice

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen3_5.amd import mono_base as B
from vllm.models.qwen3_5.amd.mono import layout as L
from vllm.platforms import current_platform

logger = init_logger(__name__)

enabled = B.enabled

# VLLM_QWEN38_MONO_FP6=1 gives the kernels MXFP6 E2M3 copies of the GDN
# in_proj_qkvz and of every out_proj and o_proj, which pack_fp6 makes after
# loading. K1 then multiplies in_proj_qkvz in MXFP6 (q6) and K2 its out_proj
# or o_proj (o6), both with the bf16 inputs. The bf16 weights stay for the
# other paths. With VLLM_QWEN38_FAKE_MXFP6_PROJ=1 they hold the same MXFP6
# values, so a prefill step and a mono decode step use the same weights.
FP6 = os.environ.get("VLLM_QWEN38_MONO_FP6", "0") == "1"


@torch.no_grad()
def pack_fp6(model, rows: int = 1024) -> None:
    """Packs the MXFP6 copies of VLLM_QWEN38_MONO_FP6: for every layer the
    out_proj or o_proj, in a GDN layer in_proj_qkvz and in a full-attention
    layer qkv_proj (if it has no bias), if its weight is bf16. A copy is
    ``mono_fp6`` = (weight bytes, scale bytes) of the linear, in the layout
    of moe.fp6_pack.

    The multi-thread loader calls load_weights once for each batch of
    shards, and each call ends here. So a later call packs into the same
    buffers again, and the last call packs every weight after all of it has
    loaded. The packing goes ``rows`` rows at a time, which keeps the
    temporaries small. A slice of whole 16-row groups is one contiguous byte
    range of both buffers, at 3/4 and 1/32 bytes a weight."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.qwen3_5.amd.mono.moe import fp6_pack

    count, nbytes = 0, 0
    for layer in model.layers:
        linears = [B.o_proj(layer)]
        if layer.layer_type == "linear_attention":
            linears.append(layer.linear_attn.in_proj_qkvz)
        elif getattr(layer.self_attn.qkv_proj, "bias", None) is None:
            # The attention front kernel (afront.py) reads qkv_proj in MXFP6.
            linears.append(layer.self_attn.qkv_proj)
        for lin in linears:
            w = getattr(lin, "weight", None)
            if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
                continue
            n, k = w.shape
            fp6 = getattr(lin, "mono_fp6", None)
            if fp6 is None:
                fp6 = (
                    torch.empty(n * k * 3 // 4, dtype=torch.uint8, device=w.device),
                    torch.empty(n * k // 32, dtype=torch.uint8, device=w.device),
                )
                lin.mono_fp6 = fp6
            for r0 in range(0, n, rows):
                part_w, part_s = fp6_pack(w.data[r0 : r0 + rows])
                at_w, at_s = r0 * k * 3 // 4, r0 * k // 32
                fp6[0][at_w : at_w + part_w.numel()].copy_(part_w)
                fp6[1][at_s : at_s + part_s.numel()].copy_(part_s)
            count += 1
            nbytes += fp6[0].numel() + fp6[1].numel()
    if count == 0:
        raise RuntimeError(
            "VLLM_QWEN38_MONO_FP6=1 found no bf16 in_proj_qkvz, out_proj or o_proj "
            "to pack. Serve them in bf16, for example with online MXFP4 "
            "targets that name only the shared expert."
        )
    if get_tensor_model_parallel_rank() == 0:
        logger.info(
            "Qwen3.8 mono decode: packed MXFP6 copies of %d projections, %.2f GB a GPU",
            count,
            nbytes / 1e9,
        )
    pack_rm(model)


@torch.no_grad()
def pack_rm(model) -> None:
    """Makes ``mono_rm``, the router weight of K2's rm (rm_pack in layer.py),
    for every layer whose out_proj or o_proj weight is bf16. It is made from
    the values that K2 multiplies: the MXFP6 values when the linear has an
    MXFP6 copy (mono_fp6), else the bf16 weight. pack_fp6 calls it at its
    end, so a later call of load_weights makes it again from the weights
    loaded by then, and the last call from all of them."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.qwen3_5.amd.mono.layer import rm_pack
    from vllm.models.qwen3_5.amd.mxfp6 import round_e2m3

    rank = get_tensor_model_parallel_rank()
    count = 0
    for layer in model.layers:
        o = B.o_proj(layer)
        w = getattr(o, "weight", None)
        gate = getattr(layer.mlp, "gate", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or gate is None:
            continue
        # round_e2m3 gives the MXFP6 values of fp6_pack. It leaves values
        # that VLLM_QWEN38_FAKE_MXFP6_PROJ already rounded unchanged.
        vals = (
            round_e2m3(w.data) if getattr(o, "mono_fp6", None) is not None else w.data
        )
        layer.mlp.mono_rm = rm_pack(
            gate.weight, layer.post_attention_layernorm.weight, vals, rank
        )
        count += 1
    if rank == 0:
        logger.info(
            "Qwen3.8 mono decode: made the rm router weights of %d layers", count
        )


class MonoDecode(B.QwenMonoDecode):
    """Bound to one ``Qwen3_5Model``; built at model init on every rank."""

    LABEL = "Qwen3.8"
    TP_SIZES = (L.TP,)
    MAX_TOKENS = L.MAX_TOKENS
    BLOCKS = L.BLOCKS
    MTP = True

    def __init__(self, model, vllm_config):
        super().__init__(model, vllm_config)
        if self.ok:
            from vllm.models.qwen3_5.amd.mono import afront

            # The mailbox through which the attention front kernel hands q
            # and k to the CTAs that run their norms and RoPE.
            n = (afront.SCRATCH_BYTES + 255) // 256 * 256
            self.af_scratch = torch.zeros(
                n, dtype=torch.uint8, device=self.epoch.device
            )

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
        return self._fp6_missing() if FP6 else None

    def _fp6_missing(self) -> str | None:
        """With VLLM_QWEN38_MONO_FP6=1 every layer must have the MXFP6 copy
        of its out_proj or o_proj, every GDN layer that of in_proj_qkvz and
        every full-attention layer that of qkv_proj. Without that check a
        layer whose weight was not bf16 would quietly run in its old
        format."""
        n_o = n_q = n_a = 0
        for layer in self.model.layers:
            if getattr(B.o_proj(layer), "mono_fp6", None) is None:
                return (
                    f"VLLM_QWEN38_MONO_FP6 layer {layer.layer_idx}: no MXFP6 out_proj "
                    "or o_proj"
                )
            n_o += 1
            if layer.layer_type == "linear_attention":
                if getattr(layer.linear_attn.in_proj_qkvz, "mono_fp6", None) is None:
                    return (
                        f"VLLM_QWEN38_MONO_FP6 layer {layer.layer_idx}: no MXFP6 "
                        "in_proj_qkvz"
                    )
                n_q += 1
            elif getattr(layer.self_attn.qkv_proj, "mono_fp6", None) is None:
                return (
                    f"VLLM_QWEN38_MONO_FP6 layer {layer.layer_idx}: no MXFP6 qkv_proj"
                )
            else:
                n_a += 1
        logger.info_once(
            "Qwen3.8 mono decode: MXFP6 in_proj_qkvz in %d layers, MXFP6 out_proj or "
            "o_proj in %d layers, MXFP6 qkv_proj in %d layers",
            n_q,
            n_o,
            n_a,
        )
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
            elif layer.layer_type == "linear_attention":
                x, res_attn = _gemma_norm(layer.input_layernorm, h, residual)
                core = B.gdn_core(layer.linear_attn, x)
            else:
                fp6 = getattr(layer.self_attn.qkv_proj, "mono_fp6", None)
                if fp6 is not None and residual is not None:
                    # The attention front kernel runs the residual add, the
                    # norm, the MXFP6 qkv_proj, the q and k norms and RoPE in
                    # one launch, and also the KV cache write where it can.
                    # It works row by row, so it also serves verify steps.
                    core, gate, res_attn = self._attn_front(
                        layer, h, residual, fp6, positions
                    )
                else:
                    x, res_attn = _gemma_norm(layer.input_layernorm, h, residual)
                    qkv, _ = layer.self_attn.qkv_proj(x)
                    core, gate = _attention_core(layer.self_attn, qkv, positions)
            h, residual = self._k2(layer, core, res_attn, gate)
        h, _ = _gemma_norm(model.norm, h, residual)
        return h

    def _attn_front(self, layer, h, residual, fp6, positions):
        """The attention front kernel: hidden + residual -> input_layernorm
        -> the MXFP6 qkv_proj -> the q and k norms, RoPE and the split, and
        the FP8 KV cache write when _af_kv allows it -> the attention ->
        (attention output, gate, hidden + residual)."""
        from vllm.models.qwen3_5.amd.mono import afront

        attn = layer.self_attn
        rope = attn.rotary_emb
        s = h.size(0)
        if positions.ndim == 2:
            # a decode token is text: its M-RoPE positions are all equal
            positions = positions[0]
        q = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
        gate = torch.empty_like(q)
        k = torch.empty(s, afront.HEAD, dtype=h.dtype, device=h.device)
        v = torch.empty_like(k)
        res_out = torch.empty_like(h)
        kv = self._af_kv(attn.attn, s)
        cache = {}
        if kv is not None:
            names = ("k_cache", "v_cache", "slot_mapping", "k_scale", "v_scale")
            cache = dict(zip(names, kv))
        afront.attn_front(
            afront.AFBuild(
                tokens=s,
                eps=layer.input_layernorm.variance_epsilon,
                qk=True,
                kv=kv is not None,
                qk_eps=attn.q_norm.variance_epsilon,
                rot=rope.rotary_dim,
                cs_f32=rope.cos_sin_cache.dtype == torch.float32,
                page=kv[0].size(1) if kv is not None else 64,
            ),
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
            cos_sin=rope.cos_sin_cache,
            positions=positions,
            scratch=self.af_scratch,
            epoch=self.epoch,
            slot=layer.layer_idx,
            **cache,
        )
        if kv is None:
            return attn.attn(q, k, v), gate, res_out
        return _attention_q_only(attn.attn, q), gate, res_out

    def _af_kv(self, attn_layer, s):
        """(k_cache, v_cache, slot_mapping, k_scale, v_scale) for the KV
        cache write in the attention front kernel, or None when vLLM must
        write the cache. The kernel writes reshape_and_cache_flash's FP8
        e4m3 values into the unshuffled [blocks, block_size, 1, 256] views,
        one slot for each row."""
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
            # vLLM can leave the pad rows of the batch out of slot_mapping.
            # The kernel would then read slots past its end.
            if slots is None or slots.dtype != torch.int64 or slots.shape != (s,):
                shape = None if slots is None else tuple(slots.shape)
                why = f"a slot_mapping of shape {shape} for {s} rows"
        if why is None:
            k_cache, v_cache = attn_layer.impl._split_kv_cache(attn_layer.kv_cache)
            if k_cache.shape[2:] != (1, 256) or k_cache.element_size() != 1:
                why = f"KV cache views of shape {tuple(k_cache.shape)}"
        if why is not None:
            logger.info_once("Qwen3.8 mono decode: vLLM writes the KV cache (%s)", why)
            return None
        logger.info_once(
            "Qwen3.8 mono decode: attn_front writes the FP8 KV cache (blocks of %d "
            "tokens)",
            k_cache.size(1),
        )
        return k_cache, v_cache, slots, attn_layer._k_scale, attn_layer._v_scale

    def _k1(self, layer, md, h, residual, res_out, core) -> None:
        from vllm.models.qwen3_5.amd.mono import gdn

        a = layer.linear_attn
        m = md[a.prefix]
        conv_state, rstate = a.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        s = h.size(0)
        # The MXFP6 copy (pack_fp6) runs on the qq path of the MXFP4
        # in_proj, with its own weight loads and GEMV (q6).
        fp6 = getattr(a.in_proj_qkvz, "mono_fp6", None)
        qq = fp6 is not None or B.asm_mxfp4(a.in_proj_qkvz)
        if fp6 is not None:
            w_qkvz, w_qkvzs = fp6
        else:
            w_qkvz = a.in_proj_qkvz.weight
            w_qkvzs = a.in_proj_qkvz.weight_scale if qq else None
        gdn.gdn_pre(
            gdn.K1Build(
                tokens=s,
                first=residual is None,
                eps=layer.input_layernorm.variance_epsilon,
                norm_eps=a.norm.eps,
                qq=qq,
                q6=fp6 is not None,
            ),
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
        fp6 = getattr(o, "mono_fp6", None)
        oq = fp6 is None and B.asm_mxfp4(o)
        if fp6 is not None:
            w_o, w_os = fp6
        else:
            w_o = o.weight
            w_os = o.weight_scale if oq else None
        fused = mlp.is_fused_shared_expert_enabled
        s = core.size(0)
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        # rm needs the layer's mono_rm (pack_rm), which is made from bf16 or
        # MXFP6 o_proj values, so the fp4 o_proj runs without it. From 6 rows
        # the router tasks after the norm are faster (K2 stage benchmark).
        w_rm = getattr(mlp, "mono_rm", None)
        rm = s < 6 and w_rm is not None and not oq
        k2.layer_post(
            k2.K2Build(
                tokens=s,
                eps=layer.post_attention_layernorm.variance_epsilon,
                fse=fused,
                oq=oq,
                o6=fp6 is not None,
                og=gate is not None,
                rm=rm,
            ),
            core=core,
            gate=gate,
            residual=residual,
            w_o=w_o,
            w_os=w_os,
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
            w_rm=w_rm if rm else None,
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


def _attention_core(attn, qkv, positions) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3NextAttention.forward from qkv_proj's output up to the output
    gate (K2 applies it to o_proj's input), with the split, q / k norms, RoPE
    and gate copy in one launch as the model's CUDA path runs them ->
    (attention output, gate)."""
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
