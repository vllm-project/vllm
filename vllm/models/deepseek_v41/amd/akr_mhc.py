# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 mHC routes on amd_kernels (hidden 5120, 4 streams).

- `mhc_seam`: post-mix of the previous sublayer + pre-mix, Sinkhorn and
  RMSNorm of the next one, in place of the AITER mega-mHC seam. It also emits
  the layer input as MXFP8 wherever a consumer reads those codes (the
  attention's fused wq_a + wkv, the fused EP8 experts, the shared expert's
  gate_up), so no quantize launch sits between the seam and those GEMMs; on
  request also the mean of the new streams (the DSpark auxiliary hidden
  state).
- `mhc_tail`: post-mix, collapse and final RMSNorm after the last layer.
- `mhc_wo_b`: wo_b GEMM with the attention post-mix as its epilogue, then the
  FFN seam's gate finisher.
- `mhc_ffn`: shared-expert down GEMM plus the routed output with the FFN
  post-mix as its epilogue, then the next layer's attention seam.

Each decision is on the static row count, so a captured shape takes one branch
in every layer. The seam, the fused epilogues and the finisher share per-device
scratch: they run serially on one stream (the stock aux-stream gate overlap is
off where the seam serves).
"""

from typing import Any, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd import akr_config, akr_kernels

logger = init_logger(__name__)

HC = 4
HIDDEN = 5120
_ROUTES = ("mhc_seam", "mhc_tail", "mhc_wo_b", "mhc_ffn")


class SeamCarry(NamedTuple):
    """What a layer whose FFN epilogue chain already ran the next attention
    seam returns in place of its FFN output: the next layer's normed input,
    its attention pre-mix and, before an auxiliary layer, the stream mean. The
    accompanying residual is already post-mixed."""

    layer_input: torch.Tensor
    attn_pre: torch.Tensor
    aux: torch.Tensor | None
    # the layer input as MXFP8, where the attention's first GEMM reads codes
    codes: QuantizedActivation | None = None


class Step(NamedTuple):
    want_aux: bool
    next_layer: Any
    chain_aux: bool


def attached_codes(x: torch.Tensor) -> QuantizedActivation | None:
    """The MXFP8 copy of `x` that AITER's MXFP8 mHC seam attached (where that
    seam ran), with byte scales as the amd_kernels entries read them."""
    from vllm.models.deepseek_v4.amd.model import mxfp8_input

    qa = mxfp8_input(x)
    if qa is None or qa.scale.element_size() != 1 or qa.data.dim() != 2:
        return None
    return akr_config.quant_activation(
        qa.data, qa.scale.view(torch.uint8), qa.orig_dtype
    )


def attach_codes(x: torch.Tensor, codes: QuantizedActivation | None) -> None:
    """Hand the seam's MXFP8 copy of `x` to the stock linears that read it."""
    if codes is not None:
        from vllm.models.deepseek_v4.amd.model import attach_mxfp8_input

        attach_mxfp8_input(x, codes)


def seam_rows_ok(m: int, with_aux: bool = False) -> bool:
    if with_aux:
        return 1 <= m <= akr_config.MHC_SEAM_AUX_MAX_ROWS
    return akr_config.GATES["mhc_seam"](m)


class AkrMhc:
    """`DeepseekV4DecoderLayer.akr_mhc`: packed gate weights, the seam and the
    fused chains of one decoder layer."""

    def __init__(self, layer, vllm_config) -> None:
        cfg = vllm_config.model_config.hf_config
        self.layer = layer
        self.shape_ok = cfg.hc_mult == HC and cfg.hidden_size == HIDDEN
        # the attention's fused wq_a + wkv linear reads MXFP8 codes (set at load)
        self.attn_codes = False
        self.no_ubatching = not vllm_config.parallel_config.use_ubatching
        self.ready = False
        self.attn_pack: torch.Tensor | None = None
        self.ffn_pack: torch.Tensor | None = None
        # fused chains
        self.gate_attn: torch.Tensor | None = None
        self.gate_ffn: torch.Tensor | None = None
        self.wo_b: Any = None
        self.shared_down: Any = None
        # requests for the next seam, and what it returned
        self.want_quant = False
        self.want_aux = False
        self.last_quant: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_aux: torch.Tensor | None = None
        # per step, set by the model's layer loop
        self._step = Step(False, None, False)
        self.aux_out: torch.Tensor | None = None

    def plan_step(self, *, want_aux: bool, next_layer, chain_aux: bool) -> None:
        self._step = Step(want_aux, next_layer, chain_aux)

    def begin_step(self) -> Step:
        step, self._step = self._step, Step(False, None, False)
        self.aux_out = None
        self.last_quant = None
        return step

    # ---------------------------------------------------------------- load

    def post_load(self) -> None:
        layer = self.layer
        if not self.shape_ok or not layer.hc_attn_fn.is_cuda:
            return
        probe = layer.attn_norm.weight.new_empty(1, HC, HIDDEN)
        if probe.dtype != torch.bfloat16 or not akr_kernels.mhc_seam_ok(probe):
            return
        self.attn_pack = akr_kernels.mhc_seam_gate(layer.hc_attn_fn.data, HC, HIDDEN)
        self.ffn_pack = akr_kernels.mhc_seam_gate(layer.hc_ffn_fn.data, HC, HIDDEN)
        wqa_wkv = getattr(getattr(layer, "attn", None), "fused_wqa_wkv", None)
        self.attn_codes = (
            wqa_wkv is not None
            and not getattr(layer, "use_sequence_parallel", False)
            and akr_config.takes_mxfp8_codes(wqa_wkv)
        )
        self.ready = True

    def post_load_chains(self) -> None:
        """The fused epilogues read the stock linears' preshuffled weights."""
        layer = self.layer
        if akr_config.on("mhc_wo_b") and hasattr(layer.attn, "forward_codes"):
            self.wo_b = akr_kernels.postmix_weight(layer.attn.wo_b, routed=False)
        if self.wo_b is not None:
            self.gate_ffn = akr_kernels.postmix_gate(
                layer.hc_ffn_fn.data, self.wo_b, False
            )
        shared = layer.ffn.shared_experts
        if akr_config.on("mhc_ffn") and shared is not None and shared.akr is not None:
            self.shared_down = akr_kernels.postmix_weight(shared.down_proj, routed=True)
        if self.shared_down is not None:
            self.gate_attn = akr_kernels.postmix_gate(
                layer.hc_attn_fn.data, self.shared_down, True
            )

    # ---------------------------------------------------------------- decisions

    def seam_ok(self, m: int, with_aux: bool = False) -> bool:
        return (
            self.ready
            and self.no_ubatching
            and akr_config.on("mhc_seam")
            and seam_rows_ok(m, with_aux)
        )

    def attn_wants_codes(self, m: int) -> bool:
        """Whether the attention seam emits the layer input as MXFP8: wherever
        it runs and the stock fused wq_a + wkv linear reads codes (seam with
        the MXFP8 epilogue + the stock GEMM on its codes beat the bf16 seam +
        the dense_gemm entry by 0.5-2.3 us at the seam's row counts)."""
        return self.attn_codes and self.seam_ok(m)

    def wo_b_chain_ok(self, m: int) -> bool:
        layer = self.layer
        return (
            self.wo_b is not None
            and akr_config.serves("mhc_wo_b", m)
            and self.seam_ok(m)
            and not layer.use_sequence_parallel
            and layer.mono is None
            and akr_kernels.postmix_ok(m, self.wo_b, False)
        )

    def ffn_chain_ok(self, m: int, nxt) -> bool:
        layer = self.layer
        nxt_mhc = None if nxt is None else nxt.akr_mhc
        return (
            nxt_mhc is not None
            and nxt_mhc.gate_attn is not None
            and nxt.engram is None
            and self.shared_down is not None
            and akr_config.serves("mhc_ffn", m)
            and self.seam_ok(m)
            and not layer.use_sequence_parallel
            and layer.mono is None
            and akr_kernels.postmix_ok(m, self.shared_down, True)
        )

    # ---------------------------------------------------------------- chains

    def wo_b_chain(
        self,
        z: QuantizedActivation,
        residual,
        post_mix,
        res_mix,
        attn_pre,
        *,
        quant: bool,
        dtype: torch.dtype,
    ):
        """wo_b with the attention post-mix as its epilogue, then the FFN seam:
        (residual, post, comb, layer_input, ffn_pre, MXFP8 input or None)."""
        layer = self.layer
        residual, partial_x, gates, _ = akr_kernels.postmix_gates_gemm(
            z, self.wo_b, residual, post_mix, res_mix, attn_pre, self.gate_ffn
        )
        out = akr_kernels.mhc_gate_finish(
            gates,
            partial_x,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.ffn_norm,
            layer,
            quant=quant,
        )
        quant_out = (
            akr_config.quant_activation(out[4], out[5], dtype) if quant else None
        )
        return residual, out[0], out[1], out[2], out[3], quant_out

    def ffn_chain(
        self, nxt, shared_act, routed, residual, post_mix, res_mix, ffn_pre, *, aux
    ):
        """Shared down + routed add with the FFN post-mix as the epilogue, then
        the next layer's attention seam: (residual, post, comb, SeamCarry)."""
        codes = nxt.akr_mhc.attn_wants_codes(residual.shape[0])
        residual, partial_x, gates, mean = akr_kernels.postmix_gates_gemm(
            shared_act,
            self.shared_down,
            residual,
            post_mix,
            res_mix,
            ffn_pre,
            nxt.akr_mhc.gate_attn,
            routed=routed,
            aux_mean=aux,
        )
        out = akr_kernels.mhc_gate_finish(
            gates,
            partial_x,
            nxt.hc_attn_scale,
            nxt.hc_attn_base,
            nxt.attn_norm,
            nxt,
            quant=codes,
        )
        carry = SeamCarry(
            out[2],
            out[3],
            mean,
            akr_config.quant_activation(out[4], out[5], out[2].dtype)
            if codes
            else None,
        )
        return residual, out[0], out[1], carry

    # ---------------------------------------------------------------- the seam

    def pre(
        self,
        stock,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        pre_mix: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        sublayer_out: torch.Tensor | None = None,
        post_layer_mix: torch.Tensor | None = None,
        comb_res_mix: torch.Tensor | None = None,
        fn_packed: torch.Tensor | None = None,
    ):
        """Drop-in for `MHCPreDelayedOp.__call__` that falls back to `stock`.
        `want_quant` / `want_aux` ask for the extra outputs, read back from
        `last_quant` / `last_aux`."""
        want_quant, want_aux = self.want_quant, self.want_aux
        self.want_quant = self.want_aux = False
        self.last_quant = self.last_aux = None
        has_post = sublayer_out is not None
        pack = (
            self.attn_pack
            if fn is self.layer.hc_attn_fn
            else self.ffn_pack
            if fn is self.layer.hc_ffn_fn
            else None
        )
        if (
            pack is None
            or x is not None
            or norm_weight is None
            or residual.dim() != 3
            or not residual.is_contiguous()
            or (sublayer_out is not None and not sublayer_out.is_contiguous())
            or not self.seam_ok(residual.shape[0], with_aux=want_aux and has_post)
        ):
            return stock(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix=pre_mix,
                x=x,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                sublayer_out=sublayer_out,
                post_layer_mix=post_layer_mix,
                comb_res_mix=comb_res_mix,
                fn_packed=fn_packed,
            )
        out = akr_kernels.mhc_seam(
            residual,
            pack,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult=hc_post_mult_value,
            norm_eps=norm_eps,
            sinkhorn_repeat=sinkhorn_repeat,
            pre_mix=pre_mix,
            sublayer_out=sublayer_out,
            post_mix=post_layer_mix,
            comb_mix=comb_res_mix,
            quant=want_quant,
            aux_mean=want_aux and has_post,
        )
        i = 5
        if want_quant:
            self.last_quant = (out[i], out[i + 1])
            i += 2
        if want_aux and has_post:
            self.last_aux = out[i]
        return out[:5]

    def quant_output(self, dtype: torch.dtype) -> QuantizedActivation | None:
        """The last seam's MXFP8 output, if it was asked for."""
        if self.last_quant is None:
            return None
        return akr_config.quant_activation(*self.last_quant, dtype)


def create(layer, vllm_config) -> AkrMhc | None:
    if not any(akr_config.on(r) for r in _ROUTES):
        return None
    return AkrMhc(layer, vllm_config)


def defer_aux(model, nxt: int, layer_ids: range, m: int) -> bool:
    """Whether the auxiliary hidden state after layer `nxt - 1` comes from the
    seam of layer `nxt` (or, for the last layer, from the model tail) instead
    of a stock post-mix + mean."""
    if not akr_config.on("mhc_seam") or model.use_sequence_parallel:
        return False
    if nxt == model.end_layer:
        return tail_ok(model, m) and model.decoder_replay_layers is None
    if nxt not in layer_ids or nxt == model.decoder_replay_start:
        return False
    layer = model.layers[nxt]
    return (
        layer.akr_mhc is not None
        and layer.akr_mhc.ready
        and layer.engram is None
        and seam_rows_ok(m, with_aux=True)
    )


# ---------------------------------------------------------------- model tail


def tail_ok(model, m: int) -> bool:
    return (
        model.akr_tail_ready
        and akr_config.serves("mhc_tail", m)
        and not model.use_sequence_parallel
    )


def model_tail(
    model, x, residual, post_mix, res_mix, pre_mix, aux_hidden_states, final_aux
):
    """The end of `DeepseekV4Model.forward` on the last PP rank: post-mix,
    the DSpark hidden-buffer copy, collapse and final norm (and the stream
    mean of the last auxiliary layer if `_run_layers` deferred it)."""
    keep = model._mtp_hidden_buffer is not None
    if model.end_layer in model.aux_hidden_state_layers and final_aux is not None:
        out, res, _ = akr_kernels.mhc_tail(
            final_aux, pre_mix, model.norm, keep_residual=keep
        )
        if keep:
            res = final_aux
    else:
        out, res, mean = akr_kernels.mhc_tail(
            residual,
            pre_mix,
            model.norm,
            sublayer_out=x,
            post_mix=post_mix,
            comb_mix=res_mix,
            keep_residual=keep,
            aux_mean=model.akr_tail_aux,
        )
        if model.akr_tail_aux:
            aux_hidden_states.append(mean)
    if keep:
        model._mtp_hidden_buffer[: out.shape[0]].copy_(res.flatten(1))
    return (out, aux_hidden_states) if aux_hidden_states else out


def post_load(model) -> None:
    from vllm.models.deepseek_v41.amd import akr_hooks

    inner: Any = getattr(model, "model", None)
    layers = [
        layer
        for layer in getattr(inner, "layers", ())
        if getattr(layer, "akr_mhc", None) is not None
    ]
    for layer in layers:
        layer.akr_mhc.post_load()
    ready = [layer for layer in layers if layer.akr_mhc.ready]
    target = not akr_hooks.is_draft(model)
    if not ready:
        if target:
            for r in _ROUTES:
                akr_config.fail(r, "no 4-stream, 5120-wide decoder layer")
        return
    if target and (akr_config.on("mhc_wo_b") or akr_config.on("mhc_ffn")):
        for layer in ready:
            layer.akr_mhc.post_load_chains()
    if (
        target
        and akr_config.on("mhc_tail")
        and inner.norm.weight.dtype == torch.bfloat16
        and akr_kernels.mhc_tail_ok(inner.norm.weight.new_empty(1, HC, HIDDEN))
    ):
        inner.akr_tail_ready = True
    akr_kernels.mhc_warmup(
        HC, HIDDEN, ready[0].hc_attn_fn.device, tail=target and inner.akr_tail_ready
    )
    logger.info(
        "amd_kernels mHC (%s): %d layers",
        "target" if target else "draft",
        len(ready),
    )
