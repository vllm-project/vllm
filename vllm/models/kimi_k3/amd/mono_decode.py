# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM side of the Kimi-K3 mono decode (``VLLM_ROCM_MONO_DECODE=1``, MI355X, TP8).

A spec-verify step runs each decoder layer as two persistent launches:

- K1 (``kda_pre``, KDA layers): AttnRes -> in_proj -> f_b -> conv -> the KDA
  recurrence -> the gated norm, writing o_proj's input.
- K2 (``k2``, every MoE layer): o_proj + its TP all-reduce + the MLP AttnRes +
  the latent MoE (router, top-16, MXFP4 experts, shared experts, the latent
  all-reduce, up_proj) + the final all-reduce. An MLA layer runs vLLM's
  attention up to o_proj's input, then K2.

Any other batch (prefill, mixed, more than ``MAX_TOKENS`` rows) takes the
original path; both paths take and return the same tensors.

The all-reduces run in-kernel over a peer buffer every TP rank maps. Mailbox
tags carry a device epoch the model bumps once a forward (``step_begin``, which
a CUDA graph captures), so no buffer is zeroed between steps.
"""

from typing import Any

import torch
from einops import rearrange

from vllm import envs
from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd.mono import runner
from vllm.models.kimi_k3.amd.mono.runner import MAX_TOKENS
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


def step_begin() -> None:
    runner.step_begin()


def _routed_experts(m):
    return getattr(m.experts, "routed_experts", m.experts)


class MonoMoe:
    """Bound to one KimiMoE (TP8, latent MoE, shared experts)."""

    def __init__(self, moe_mod):
        from vllm.models.kimi_k3.amd.mono.stages import moe

        self.m = moe_mod
        self.ok = (
            moe_mod.tp_size == moe.TP
            and moe_mod.use_latent_moe
            and moe_mod.shared_experts is not None
            and moe_mod.routed_scaling_factor == 1.0
        )
        self._w_up: torch.Tensor | None = None
        if not self.ok:
            logger.info_once("K3 mono: MoE layer unsupported, original path")
            return
        dev = moe_mod.gate.weight.device
        runner.alloc_scratch(dev)
        runner.peers(dev)

    def w_up(self) -> torch.Tensor:
        """This rank's up_proj rows: the weight itself when it is row sharded,
        else its slice of the replicated weight."""
        if self._w_up is None:
            from vllm.models.kimi_k3.amd.mono.stages import moe

            m = self.m
            w = m.routed_expert_up_proj.weight
            if not getattr(m.routed_output_transform, "row_sharded", False):
                w = w.narrow(0, get_tp_group().rank_in_group * moe.UP_N, moe.UP_N)
            self._w_up = w
        return self._w_up

    def eligible(self, x: torch.Tensor) -> bool:
        return (
            self.ok
            and x.size(0) <= MAX_TOKENS
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.models.kimi_k3.amd.mono.stages import moe

        m = self.m
        ex = _routed_experts(m)
        out = torch.empty_like(x)
        moe.moe(
            moe.MoeBuild(tokens=x.size(0), eps=m.routed_expert_norm.variance_epsilon),
            x=x,
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
            w_up=self.w_up(),
            out=out,
            queue=runner.queue(),
            **runner.launch_args(),
            layer=128 + m.layer_idx,
        )
        return out


class MonoKda:
    """Bound to one KDA ``KimiDecoderLayer``."""

    def __init__(self, layer):
        from vllm.models.kimi_k3.amd.mono.attention import kda as kda_pre

        attn = layer.self_attn
        self.layer = layer
        # a FULL graph would bake one step's metadata into the launch
        self.ok = (
            is_breakable_cudagraph_enabled()
            and layer.prev_valid_blocks >= 1
            and attn.local_num_heads == kda_pre.NH
            and attn.head_dim == kda_pre.HD
            and attn.in_proj_qkvgfab.weight.shape == (kda_pre.NPROJ, kda_pre.HIDDEN)
            and attn.conv_size == 4
        )
        if self.ok:
            runner.alloc_scratch(attn.in_proj_qkvgfab.weight.device)
        else:
            logger.info_once("K3 mono: KDA layer unsupported, original path")

    def weights_ok(self) -> bool:
        layer, attn = self.layer, self.layer.self_attn
        bf = torch.bfloat16
        return (
            attn.in_proj_qkvgfab.weight.dtype == bf
            and attn.f_b_proj.weight.dtype == bf
            and attn.o_norm.weight.dtype == bf
            and layer.input_layernorm.weight.dtype == bf
            and layer.self_attention_res_norm.weight.dtype == bf
            and layer.self_attention_res_proj.weight.dtype == bf
            and attn.conv1d.weight.dtype == torch.float32
            and attn.A_log.dtype == torch.float32
            and attn.dt_bias.dtype == torch.float32
        )

    def core(self, prefix_sum, block_residual, prefix_delta, positions):
        """K1 (or the original path) -> the gated core rows (o_proj's input)."""
        from vllm.models.kimi_k3.amd.mono.attention.kda import PROJ

        core = prefix_sum.new_empty(prefix_sum.size(0), PROJ)
        _mono_kda_pre(self, prefix_sum, block_residual, prefix_delta, positions, core)
        return core


def _spec_len(m, s, prefix_sum, prefix_delta, block_residual) -> int | None:
    """The uniform spec length of a pure spec-verify step K1 can serve, else
    None."""
    if m is None or m.spec_sequence_masks is None:
        return None
    if not prefix_sum.is_contiguous():
        return None
    if prefix_delta is not None and not prefix_delta.is_contiguous():
        return None
    if block_residual.stride(-1) != 1:
        return None
    if m.num_prefills != 0 or m.num_decodes != 0 or m.num_spec_decodes < 1:
        return None
    idx = m.spec_state_indices_tensor
    L = idx.size(-1)
    if m.uniform_spec_sequence_length != L or m.num_actual_tokens != s:
        return None
    if m.num_spec_decodes * L != s or s > MAX_TOKENS:
        return None
    if idx.dtype != torch.int32 or idx.stride(-1) != 1:
        return None
    if m.num_accepted_tokens is None or m.num_accepted_tokens.dtype != torch.int32:
        return None
    return L


@eager_break_during_capture
def _mono_kda_pre(
    mono_kda: "MonoKda", prefix_sum, block_residual, prefix_delta, positions, core
):
    from vllm.models.kimi_k3.amd.mono.attention import kda as kda_pre

    layer = mono_kda.layer
    attn = layer.self_attn
    s = prefix_sum.size(0)
    md = get_forward_context().attn_metadata
    m: Any = md.get(attn.prefix) if isinstance(md, dict) else None
    L = _spec_len(m, s, prefix_sum, prefix_delta, block_residual)
    if L is None or not mono_kda.weights_ok():
        _original(layer, prefix_sum, block_residual, prefix_delta, core)
        return
    conv_state, rstate = attn.kv_cache
    if conv_state.dtype != torch.bfloat16 or rstate.dtype != torch.float32:
        _original(layer, prefix_sum, block_residual, prefix_delta, core)
        return
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    key = kda_pre.KdaPreBuild(
        tokens=s,
        qlen=L,
        nblocks=layer.prev_valid_blocks,
        delta=prefix_delta is not None,
        state_len=conv_state.size(-1),
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        onorm_eps=attn.o_norm.eps,
        lower_bound=attn.gate_lower_bound,
        write_idx=layer.block_write_idx if layer.is_block_write_layer else -1,
    )
    assert kda_pre.scratch_bytes(key) <= runner.scratch_bytes()
    kda_pre.kda_pre(
        key,
        prefix=prefix_sum,
        delta=prefix_delta,
        blocks=block_residual,
        ares_nw=layer.self_attention_res_norm.weight,
        ares_qk=layer.self_attention_res_proj.weight.view(kda_pre.HIDDEN),
        in_nw=layer.input_layernorm.weight,
        w_in=attn.in_proj_qkvgfab.weight,
        w_fb=attn.f_b_proj.weight,
        conv_w=attn.conv1d.weight.view(kda_pre.QKV, 4),
        conv_state=conv_state,
        a_log=attn.A_log,
        dt_bias=attn.dt_bias,
        on_w=attn.o_norm.weight,
        rstate=rstate,
        st_idx=m.spec_state_indices_tensor,
        num_acc=m.num_accepted_tokens,
        core_out=core,
        scratch=runner.scratch(),
        layer=layer.layer_idx,
        epoch=runner.epoch(),
    )


def _original(layer, prefix_sum, block_residual, prefix_delta, core):
    """The unfused path up to o_proj: KimiDecoderLayer.forward_attn_residual's
    AttnRes, then KimiK3DeltaAttention.forward minus o_proj."""
    from vllm.models.kimi_k3.amd.linear import _apply_attn_res

    attn = layer.self_attn
    hidden_states = _apply_attn_res(
        prefix_sum,
        block_residual,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
        layer.prev_valid_blocks,
        delta=prefix_delta,
        output_norm=layer.input_layernorm,
        block_write_idx=layer.block_write_idx if layer.is_block_write_layer else -1,
    )
    assert isinstance(hidden_states, torch.Tensor)  # no quant_key: unquantized
    num_tokens = hidden_states.size(0)
    projected = attn.in_proj_qkvgfab(hidden_states)[0]
    split_sizes = [
        3 * attn.local_projection_size,
        attn.local_projection_size,
        attn.head_dim,
        attn.local_num_heads,
    ]
    if attn.in_proj_padding:
        split_sizes.append(attn.in_proj_padding)
    mixed_qkv, g_proj_states, f_a, beta = projected.split(split_sizes, dim=-1)[:4]
    g1 = rearrange(attn.f_b_proj(f_a)[0], "n (h d) -> 1 n h d", d=attn.head_dim)
    g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=attn.head_dim)
    core_attn_out = torch.empty(
        (1, num_tokens, attn.local_num_heads, attn.head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # the custom op's body: this already runs inside an eager break
    fwd = type(attn)._forward
    fwd = getattr(fwd, "__wrapped__", fwd)
    fwd(
        attn,
        mixed_qkv=mixed_qkv,
        g1=g1,
        g2=g2,
        beta=beta.unsqueeze(0),
        core_attn_out=core_attn_out,
    )
    core.copy_(rearrange(core_attn_out, "1 n h d -> n (h d)"))


class MonoK2:
    """Bound to a ``KimiDecoderLayer`` whose MLP is a mono-capable KimiMoE:
    o_proj + its all-reduce + the MLP AttnRes + the MoE in one launch."""

    def __init__(self, layer):
        self.layer = layer
        self.moe = layer.mlp._mono_moe

    def eligible(self, core: torch.Tensor) -> bool:
        return core.size(0) <= MAX_TOKENS and core.dtype == torch.bfloat16

    def forward(self, core, prefix_sum, block_residual, reset=False):
        """``reset`` (a block-write layer): ``prefix_sum`` is the new prefix's
        buffer, written with the all-reduced o_proj; the MLP AttnRes counts the
        block written this layer."""
        from vllm.models.kimi_k3.amd.mono import layer as k2
        from vllm.models.kimi_k3.amd.mono.attention.kda import HIDDEN

        layer = self.layer
        m = layer.mlp
        ex = _routed_experts(m)
        out = torch.empty_like(prefix_sum)
        k2.k2_launch(
            k2.K2Build(
                tokens=core.size(0),
                nblocks=layer.prev_valid_blocks + (1 if reset else 0),
                reset=reset,
                eps=layer.mlp_res_norm.variance_epsilon,
                out_eps=layer.post_attention_layernorm.variance_epsilon,
                ln_eps=m.routed_expert_norm.variance_epsilon,
            ),
            core=core,
            w_o=layer.self_attn.o_proj.weight,
            prefix=prefix_sum,
            blocks=block_residual,
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
            w_up=self.moe.w_up(),
            out=out,
            **runner.launch_args(),
            layer=128 + layer.layer_idx,
        )
        return out
