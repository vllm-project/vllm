# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer kernels: each takes and returns what
``KimiDecoderLayer.forward`` does with AttnRes, ``(prefix_sum, block_residual,
prefix_delta)``.

- ``WholeLayer`` (KDA layers): K1 (AttnRes, in_proj, f_b, spec conv, the KDA
  recurrence, the gated norm) writes o_proj's input; K2 runs o_proj, its
  all-reduce, the MLP AttnRes and the latent MoE.
- ``FfnLayer`` (MLA layers): M1a (AttnRes, the query prologue, the KV write
  and, at DCP8, the query gather), AITER's decode, M1b (at DCP8 the LSE
  combine, W_UV and the gate), then K2.
- ``MlaOpsLayer`` (MLA layers M1 cannot serve, ``m1_reasons``): vLLM's AttnRes
  and MLA attention ops, then K2.

Weights are views of the loaded parameters, taken at every launch.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import Step


def _k2_key(layer, runner: MonoRunner, tokens: int, reset: bool, m1b: bool = False):
    """``m1b``: an MLA layer's K2 that runs its M1b first (``runner.fuse_m1b``)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2

    m = layer.mlp
    fused = (
        dict(
            m1b_qlen=runner.num_spec + 1,
            m1b_ranks=runner.dcp,
            m1b_off=runner.mla_post_off,
            m1b_splits=m1c_splits(runner, tokens),
            m1c=m1c_at(runner, tokens),
        )
        if m1b
        else {}
    )
    return k2.K2Build(
        tokens=tokens,
        nblocks=layer.prev_valid_blocks + (1 if reset else 0),
        reset=reset,
        eps=layer.mlp_res_norm.variance_epsilon,
        out_eps=layer.post_attention_layernorm.variance_epsilon,
        ln_eps=m.routed_expert_norm.variance_epsilon,
        **fused,
        **runner.fault,
    )


def m1c_at(runner: MonoRunner, width: int) -> bool:
    """``runner.m1c`` at this width: M1c's KV splits must share out evenly over
    the requests (qlen 4 has 80: a 12-row width, 3 requests, runs the asm
    decode instead)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn

    q = runner.num_spec + 1
    return runner.m1c and mla_attn.splits(q) % (width // q) == 0


def m1c_splits(runner: MonoRunner, width: int) -> int:
    """M1b's ``splits``: M1c's, when K2 runs it at this width (else 0)."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_attn

    return mla_attn.splits(runner.num_spec + 1) if m1c_at(runner, width) else 0


def _k2(
    layer,
    runner: MonoRunner,
    core,
    prefix,
    blocks,
    reset=False,
    m1b_in=None,
    m1c_in=None,
):
    """K2 from ``core`` (o_proj's input): ``prefix`` updated in place (written,
    when ``reset``), the MoE output returned. ``m1b_in``: ``mla_post``'s inputs,
    when this launch runs M1b and writes ``core`` itself; ``m1c_in``: M1c's
    (``mla_attn.host_args``'s), when it runs M1c before."""
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    m = layer.mlp
    ex = getattr(m.experts, "routed_experts", m.experts)
    w_up = m.routed_expert_up_proj.weight
    if not getattr(m.routed_output_transform, "row_sharded", False):
        w_up = w_up.narrow(0, runner.rank * moe.UP_N, moe.UP_N)
    out = torch.empty_like(prefix)
    k2.k2_launch(
        _k2_key(layer, runner, core.size(0), reset, m1b_in is not None),
        core=core,
        w_o=layer.self_attn.o_proj.weight,
        prefix=prefix,
        blocks=blocks,
        ares_nw=layer.mlp_res_norm.weight,
        ares_qk=layer.mlp_res_proj.weight.view(HIDDEN),
        in_nw=layer.post_attention_layernorm.weight,
        w_gate=m.gate.weight,
        bias=m.gate.e_score_correction_bias,
        w_ld=m.routed_expert_down_proj.weight,
        w_sgu=m.shared_experts.gate_up_proj.weight,
        w_sd=m.shared_experts.down_proj.weight,
        w13=ex.w13_weight,
        w13s=ex.w13_weight_scale,
        w2=ex.w2_weight,
        w2s=ex.w2_weight_scale,
        ln_w=m.routed_expert_norm.weight,
        w_up=w_up,
        out=out,
        layer=128 + layer.layer_idx,
        m1b_in=m1b_in,
        m1c_in=m1c_in,
        **runner.launch_args(),
        **runner.peer_args(),
    )
    return out


def compile_build(key) -> None:
    """Compiles one launch's kernel (a ``K2Build``, ``KdaPreBuild``,
    ``MlaPreBuild`` or ``MlaPostBuild``) without running it: FlyDSL's
    compile-only mode returns before the launch."""
    import os

    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
        kda,
        mla,
        mla_back,
    )

    mod = {
        kda.KdaPreBuild: kda,
        k2.K2Build: k2,
        mla.MlaPreBuild: mla,
        mla_back.MlaPostBuild: mla_back,
    }[type(key)]
    launch = mod.build(key)
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*[0] * len(mod.ABI.names), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


class _Layer:
    def __init__(self, layer, runner: MonoRunner, width: int):
        self.layer = layer
        self.runner = runner
        self.width = width
        # the step's first launch bumps the epoch (a graph captures it)
        self.first = layer.layer_idx == runner.first_layer

    def builds(self) -> list:
        """The builds this layer launches at its width, compiled before capture."""
        reset = getattr(self.layer, "is_block_write_layer", False)
        return [_k2_key(self.layer, self.runner, self.width, reset)]


class WholeLayer(_Layer):
    def _k1_key(self, qlen: int, state_len: int, delta: bool):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

        layer = self.layer
        attn = layer.self_attn
        return kda.KdaPreBuild(
            tokens=self.width,
            qlen=qlen,
            nblocks=layer.prev_valid_blocks,
            delta=delta,
            state_len=state_len,
            eps=layer.self_attention_res_norm.variance_epsilon,
            out_eps=layer.input_layernorm.variance_epsilon,
            onorm_eps=attn.o_norm.eps,
            lower_bound=attn.gate_lower_bound,
            write_idx=layer.block_write_idx if layer.is_block_write_layer else -1,
        )

    def builds(self) -> list:
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

        q = self.runner.num_spec + 1
        # the conv state keeps CONV_W - 1 + num_spec entries; a layer after
        # another always has a pending delta
        k1 = self._k1_key(q, kda.CONV_W - 1 + self.runner.num_spec, True)
        return [k1, *super().builds()]

    def caches(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The conv state as [slots, dim, state_len] (a view) and the recurrent
        state."""
        conv_state, rstate = self.layer.self_attn.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        return conv_state, rstate

    def launch(self, step: "Step", positions, prefix_sum, block_residual, prefix_delta):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

        layer, runner = self.layer, self.runner
        attn = layer.self_attn
        m: Any = step.attn_metadata[attn.prefix]
        if self.first:
            runner.step_begin()
        conv_state, rstate = self.caches()
        s = prefix_sum.size(0)
        core = prefix_sum.new_empty(s, kda.PROJ)
        kda.kda_pre(
            self._k1_key(
                m.spec_state_indices_tensor.size(-1),
                conv_state.size(-1),
                prefix_delta is not None,
            ),
            prefix=prefix_sum,
            delta=prefix_delta,
            blocks=block_residual,
            ares_nw=layer.self_attention_res_norm.weight,
            ares_qk=layer.self_attention_res_proj.weight.view(kda.HIDDEN),
            in_nw=layer.input_layernorm.weight,
            w_in=attn.in_proj_qkvgfab.weight,
            w_fb=attn.f_b_proj.weight,
            conv_w=attn.conv1d.weight.view(kda.QKV, 4),
            conv_state=conv_state,
            a_log=attn.A_log,
            dt_bias=attn.dt_bias,
            on_w=attn.o_norm.weight,
            rstate=rstate,
            st_idx=m.spec_state_indices_tensor,
            num_acc=m.num_accepted_tokens,
            core_out=core,
            layer=layer.layer_idx,
            **runner.launch_args(),
        )
        if layer.is_block_write_layer:
            # the prefix restarts at the attention's output
            prefix_sum = torch.empty_like(prefix_sum)
            out = _k2(layer, runner, core, prefix_sum, block_residual, reset=True)
        else:
            out = _k2(layer, runner, core, prefix_sum, block_residual)
        return prefix_sum, block_residual, out


class FfnLayer(_Layer):
    def __init__(self, layer, runner: MonoRunner, width: int):
        super().__init__(layer, runner, width)
        self.wrapper = layer.self_attn.mla_attn

    def _m1a_key(self, delta: bool, qheads: int | None = None):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla

        layer, w = self.layer, self.wrapper
        if qheads is None:
            qheads = self._cprr_heads()
        return mla.MlaPreBuild(
            tokens=self.width,
            nblocks=layer.prev_valid_blocks,
            delta=delta,
            eps=layer.self_attention_res_norm.variance_epsilon,
            out_eps=layer.input_layernorm.variance_epsilon,
            q_eps=w.q_a_layernorm.variance_epsilon,
            kv_eps=w.kv_a_layernorm.variance_epsilon,
            ranks=self.runner.dcp,
            peer_off=self.runner.mla_peer_off,
            qheads=qheads,
            qbf16=m1c_at(self.runner, self.width),
        )

    def _cprr_heads(self) -> int:
        """The head count AITER's cprr decode pads the gathered query to, 0 if
        it doesn't pad. The warm-up's guess; ``launch`` reads the step's."""
        import vllm.envs as envs
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import NH
        from vllm.v1.attention.backends.mla import rocm_aiter_mla as am

        n = self.runner.dcp * NH
        asm = am._asm_dcp_verify_configured(
            self.runner.dcp, self.runner.interleave, True
        )
        if (
            not asm
            or self.runner.num_spec + 1 < am._MIN_CPRR_QLEN
            or envs.VLLM_ROCM_AITER_MLA_DCP_VERIFY.lower() == "segmented"
        ):
            return 0
        heads = am._asm_dcp_verify_heads(n)
        return heads if heads > n else 0

    def _m1b_key(self, qlen: int):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_back

        return mla_back.MlaPostBuild(
            tokens=self.width,
            qlen=qlen,
            ranks=self.runner.dcp,
            peer_off=self.runner.mla_post_off,
            splits=m1c_splits(self.runner, self.width),
        )

    def builds(self) -> list:
        # a layer after another always has a pending delta
        if self.runner.fuse_m1b:
            k2 = _k2_key(self.layer, self.runner, self.width, False, True)
            return [self._m1a_key(True), k2]
        m1b = self._m1b_key(self.runner.num_spec + 1)
        return [self._m1a_key(True), m1b, *super().builds()]

    def caches(self) -> torch.Tensor:
        """The fp8 MLA cache [blocks, block_size, 576]."""
        return self.wrapper.mla_attn.kv_cache

    def launch(self, step: "Step", positions, prefix_sum, block_residual, prefix_delta):
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla as m1
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_back
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
        from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla
        from vllm.platforms import current_platform

        layer, w, runner = self.layer, self.wrapper, self.runner
        a = w.mla_attn
        if self.first:
            runner.step_begin()
        md, _, kv_cache, slots = mla.context(w)
        w_qkv, w_g = mla.qkv_gate_weights(w)
        s = prefix_sum.size(0)
        m1c = m1c_at(runner, self.width)
        # AITER's cprr pad (e.g. 96 -> 128 heads) M1a writes in place of
        # forward_mqa's copy (mla.attend hands it over)
        qh = md.decode.asm_decode_num_heads if runner.dcp > 1 else 0
        qh = qh if qh > runner.dcp * m1.NH else 0
        q8 = prefix_sum.new_empty(
            s, qh or runner.dcp * m1.NH, m1.ENTRY,
            dtype=torch.bfloat16 if m1c else current_platform.fp8_dtype(),
        )  # fmt: skip
        gate = prefix_sum.new_empty(s, m1.GROWS)
        m1.mla_pre(
            self._m1a_key(prefix_delta is not None, qh),
            prefix=prefix_sum,
            delta=prefix_delta,
            blocks=block_residual,
            ares_nw=layer.self_attention_res_norm.weight,
            ares_qk=layer.self_attention_res_proj.weight.view(HIDDEN),
            in_nw=layer.input_layernorm.weight,
            w_qkv=w_qkv,
            w_g=w_g,
            q_nw=w.q_a_layernorm.weight,
            kv_nw=w.kv_a_layernorm.weight,
            w_qb=w.q_b_proj.weight,
            w_k=a.W_K,
            w_k_scale=a.W_K_scale,
            kv_cache=kv_cache,
            slots=slots[:s],
            gate_out=gate,
            q_out=q8,
            layer=layer.layer_idx,
            **runner.launch_args(),
            **runner.peer_args(),
        )
        # DCP1: K2 runs the attention (M1c), its KV splits' partials split 0's views
        m1c_in = None
        if m1c:
            m1c_in, po, pl = mla.m1c_inputs(w, q8, kv_cache, md)
            o, lse = po[0], pl[0]
        else:
            o, lse = mla.attend(w, q8, kv_cache, md)
        core = torch.empty_like(gate)
        m1b_in = dict(
            o=o,
            lse=lse,
            seq_lens=md.decode.seq_lens,
            qsl=md.query_start_loc,
            w_v=a.W_V,
            w_v_scale=a.W_V_scale,
            gate=gate,
        )
        if runner.fuse_m1b:
            out = _k2(
                layer, runner, core, prefix_sum, block_residual, m1b_in=m1b_in,
                m1c_in=m1c_in,
            )  # fmt: skip
            return prefix_sum, block_residual, out
        mla_back.mla_post(
            self._m1b_key(runner.num_spec + 1),
            core=core,
            layer=layer.layer_idx,
            **m1b_in,
            **runner.launch_args(),
            **runner.peer_args(),
        )
        out = _k2(layer, runner, core, prefix_sum, block_residual)
        return prefix_sum, block_residual, out


class MlaOpsLayer(_Layer):
    """An MLA layer whose attention runs as vLLM's ops (o_proj's input), then
    K2: for deployments M1 cannot serve."""

    def launch(self, step: "Step", positions, prefix_sum, block_residual, prefix_delta):
        from vllm.models.kimi_k3.amd.linear import _apply_attn_res

        layer = self.layer
        if self.first:
            self.runner.step_begin()
        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
            layer.prev_valid_blocks,
            delta=prefix_delta,
            output_norm=layer.input_layernorm,
            quant_key=layer.self_attn.get_input_quant_key(),
        )
        core = layer.self_attn.mla_attn.forward(
            positions, hidden_states, skip_o_proj=True
        )
        out = _k2(layer, self.runner, core, prefix_sum, block_residual)
        return prefix_sum, block_residual, out
