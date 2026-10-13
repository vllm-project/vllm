# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM side of the Kimi-K3 mono decode (``VLLM_ROCM_MONO_DECODE=1``, MI355X, TP8).

A uniform spec-verify step selects persistent kernels by row count:

- K1 (``kda_pre``, KDA layers): AttnRes -> in_proj -> f_b -> conv -> the KDA
  recurrence -> the gated norm, writing o_proj's input.
- K2 (``k2``, every MoE layer): o_proj + its TP all-reduce + the MLP AttnRes +
  the latent MoE (router, top-16, MXFP4 experts, shared experts, the latent
  all-reduce, up_proj) + the final all-reduce. An MLA layer runs vLLM's
  attention up to o_proj's input, then K2.

KDA steps of at most 16 rows overlap K1 and K2 in one launch. Selected larger
batches through 48 rows use a separate K2 with packed expert tokens and
shape-specific front-stage pipelining. Other larger batches retain the
production MoE. KDA batches above 29 rows retain the
production input projection and fuse the post-projection KDA. K1 needs pure
uniform verification with at most 8 tokens per request; other attention
batches take the original path.

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
from vllm.config import get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd.mono import runner
from vllm.models.kimi_k3.amd.mono.common.tiles import TILE_TOKENS
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
    if get_current_vllm_config().kv_transfer_config is not None:
        # the kernels spin-wait on every CTA being resident; a KV connector's
        # copy kernels on another stream can hold the CUs and deadlock them
        logger.warning_once("VLLM_ROCM_MONO_DECODE is off with a KV connector")
        return False
    return True


def local_post_enabled(config) -> bool:
    if not envs.VLLM_ROCM_MONO_DECODE or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    transfer = config.kv_transfer_config
    if transfer is not None and transfer.kv_connector != "SimpleCPUOffloadConnector":
        return False
    spec = config.speculative_config
    return on_gfx950() and (spec is None or not spec.enable_adaptive_verification)


def serialized_offload_kda(config) -> bool:
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != "SimpleCPUOffloadConnector":
        return False
    extra = transfer.kv_connector_extra_config or {}
    return (
        extra.get("serialize_copies_with_compute") is True
        and extra.get("kv_offload_backend", "cpu") == "cpu"
        and local_post_enabled(config)
        and not config.parallel_config.use_ubatching
    )


def kda_enabled() -> bool:
    return enabled() or serialized_offload_kda(get_current_vllm_config())


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
            and 1 <= x.size(0) <= 16
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
        self._launches: dict = {}
        self._packed_projection: torch.Tensor | None = None
        self._overlap_arguments: dict = {}
        self._overlap_launches: dict = {}
        config = get_current_vllm_config()
        self.compilation_config = config.compilation_config
        self.serialized_offload = serialized_offload_kda(config)
        self.ok = (
            (is_breakable_cudagraph_enabled() or self.serialized_offload)
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
        if (
            prefix_sum.size(0) >= 30
            and not self.serialized_offload
            and self.layer.self_attn.use_local_post
            and self.weights_ok()
        ):
            from vllm.models.kimi_k3.amd.linear import _apply_attn_res

            layer = self.layer
            x = _apply_attn_res(
                prefix_sum,
                block_residual,
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
                layer.prev_valid_blocks,
                delta=prefix_delta,
                output_norm=layer.input_layernorm,
                block_write_idx=(
                    layer.block_write_idx if layer.is_block_write_layer else -1
                ),
            )
            projected = layer.self_attn.in_proj_qkvgfab(x)[0]
            post_projection(layer.self_attn, projected, core)
            return core
        _mono_kda_pre(self, prefix_sum, block_residual, prefix_delta, positions, core)
        return core

    def fused(self, prefix, blocks, delta, positions):
        from vllm.models.kimi_k3.amd.mono.attention.kda import PROJ

        core = prefix.new_empty(prefix.size(0), PROJ)
        next_prefix = (
            torch.empty_like(prefix) if self.layer.is_block_write_layer else prefix
        )
        out = torch.empty_like(prefix)
        _mono_kda_fused(self, prefix, blocks, delta, positions, core, next_prefix, out)
        return next_prefix, blocks, out

    def launch(self, key, *, stream, **values):
        """Compile once; metadata pointers and the stream remain runtime inputs."""
        import flydsl.compiler as flyc

        from vllm.models.kimi_k3.amd.mono.attention import kda

        args = (*kda.ABI.pack(values), stream)
        compiled = self._launches.get(key)
        if compiled is None:
            if key.pipeline:
                logger.info_once(
                    "K3 mono K1 selected: M=%d L=%d pipeline=%d state_rows=%d",
                    key.tokens,
                    key.qlen,
                    key.pipeline,
                    key.state_rows,
                )
            # compile() executes the launch while materializing the callable.
            compiled = flyc.compile(kda.build(key), *args)
            if compiled is not None:
                self._launches[key] = compiled
        else:
            compiled(*args)

    def projection_weight(self, pipeline: int) -> torch.Tensor:
        weight = self.layer.self_attn.in_proj_qkvgfab.weight
        if pipeline & 33554432:
            if self._packed_projection is None:
                from vllm.models.kimi_k3.amd.mono.attention.parallel import (
                    pack_projection_weights,
                )

                self._packed_projection = pack_projection_weights(weight)
            return self._packed_projection
        return weight


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
    if not 1 <= L <= TILE_TOKENS:
        return None
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
    prepared = _prepare_kda(
        mono_kda, prefix_sum, block_residual, prefix_delta, positions, core
    )
    if prepared is None:
        _original(mono_kda.layer, prefix_sum, block_residual, prefix_delta, core)
        return
    key, values = prepared
    mono_kda.launch(key, **values)


def _prepare_kda(
    mono_kda: "MonoKda", prefix_sum, block_residual, prefix_delta, positions, core
):
    from vllm.models.kimi_k3.amd.mono.attention import kda as kda_pre

    layer = mono_kda.layer
    attn = layer.self_attn
    s = prefix_sum.size(0)
    md = get_forward_context().attn_metadata
    m: Any = md.get(attn.prefix) if isinstance(md, dict) else None
    L = _spec_len(m, s, prefix_sum, prefix_delta, block_residual)
    if (
        get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        and not mono_kda.compilation_config.cudagraph_mode.separate_routine()
    ):
        L = None
    if L is None or not mono_kda.weights_ok():
        return None
    conv_state, rstate = attn.kv_cache
    if conv_state.dtype != torch.bfloat16 or rstate.dtype != torch.float32:
        return None
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    pipeline, state_rows = {
        (12, 6): (35667968, 0),
        (18, 6): (35667968, 0),
        (24, 6): (2113536, 0),
        (16, 8): (33562624, 0),
        (24, 8): (35667968, 32),
    }.get((s, L), (0, s if L == 8 and s in (32, 64) else 0))
    if s <= 16 and L in (2, 4, 6, 8):
        pipeline = 35667968 if L == 6 else 33562624
        state_rows = 16
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
        state_rows=state_rows,
        pipeline=pipeline,
    )
    assert kda_pre.scratch_bytes(key) <= runner.scratch_bytes()
    assert attn.in_proj_qkvgfab.weight.is_contiguous()
    assert rstate.shape[1:] == (kda_pre.NH, kda_pre.HD, kda_pre.HD)
    assert rstate.stride(-1) == 1 and rstate.stride(-2) == kda_pre.HD
    assert rstate.stride(-3) == kda_pre.HD * kda_pre.HD
    return key, dict(
        prefix=prefix_sum.data_ptr(),
        delta=kda_pre.ptr(prefix_delta),
        blocks=block_residual.data_ptr(),
        blk_sm=block_residual.stride(0),
        blk_sr=block_residual.stride(1),
        ares_nw=layer.self_attention_res_norm.weight.data_ptr(),
        ares_qk=layer.self_attention_res_proj.weight.data_ptr(),
        in_nw=layer.input_layernorm.weight.data_ptr(),
        w_in=mono_kda.projection_weight(pipeline).data_ptr(),
        w_fb=attn.f_b_proj.weight.data_ptr(),
        conv_w=attn.conv1d.weight.data_ptr(),
        conv_st=conv_state.data_ptr(),
        cs_seq=conv_state.stride(0),
        cs_dim=conv_state.stride(1),
        cs_tok=conv_state.stride(2),
        a_log=attn.A_log.data_ptr(),
        dt_bias=attn.dt_bias.data_ptr(),
        on_w=attn.o_norm.weight.data_ptr(),
        rstate=rstate.data_ptr(),
        rs_seq=rstate.stride(0),
        st_idx=m.spec_state_indices_tensor.data_ptr(),
        st_stride=m.spec_state_indices_tensor.stride(0),
        num_acc=m.num_accepted_tokens.data_ptr(),
        core_out=core.data_ptr(),
        x_dbg=0,
        proj_dbg=0,
        scratch=runner.scratch().data_ptr(),
        layer=layer.layer_idx,
        epoch=runner.epoch().data_ptr(),
        tl=0,
        stream=torch.cuda.current_stream(),
    )


@eager_break_during_capture
def _mono_kda_fused(mono_kda, prefix, blocks, delta, positions, core, next_prefix, out):
    import flydsl.compiler as flyc

    from vllm.models.kimi_k3.amd.mono import overlap

    layer = mono_kda.layer
    post = layer._mono_k2
    prepared = _prepare_kda(mono_kda, prefix, blocks, delta, positions, core)
    capture_mode = get_forward_context().cudagraph_runtime_mode
    if prepared is None or capture_mode == CUDAGraphMode.FULL:
        if prepared is None:
            _original(layer, prefix, blocks, delta, core)
        else:
            key, values = prepared
            mono_kda.launch(key, **values)
        out.copy_(
            post.forward(core, next_prefix, blocks, reset=layer.is_block_write_layer)
        )
        return
    first_key, first = prepared
    if first_key.qlen not in (2, 4, 6, 8):
        mono_kda.launch(first_key, **first)
        out.copy_(
            post.forward(core, next_prefix, blocks, reset=layer.is_block_write_layer)
        )
        return
    key = overlap.OverlapBuild(
        first_key,
        post.build_key(core.size(0), layer.is_block_write_layer),
        k1_workers=256,
        reuse_workers=True,
        tagged_core=True,
    )
    assert overlap.scratch_bytes(key) <= runner.scratch_bytes()
    second = post.arguments(core, next_prefix, blocks, out)
    second.update(blk_sm=blocks.stride(0), blk_sr=blocks.stride(1))
    values = overlap.argument_values(first, second, reset=key.k2.reset)
    cache_key = key, values
    arguments = mono_kda._overlap_arguments.get(cache_key)
    if arguments is None:
        arguments = torch.tensor(values, dtype=torch.int64, device=prefix.device)
        mono_kda._overlap_arguments[cache_key] = arguments
    state = runner.launch_args()
    args = (
        arguments.data_ptr(),
        state["scratch"].data_ptr(),
        state["peers"].data_ptr(),
        state["epoch"].data_ptr(),
        state["rank"],
        layer.layer_idx,
        0,
        torch.cuda.current_stream(),
    )
    compiled = mono_kda._overlap_launches.get(key)
    if compiled is None:
        compiled = flyc.compile(overlap.build(key), *args)
        if compiled is not None:
            mono_kda._overlap_launches[key] = compiled
        logger.info_once(
            "K3 mono full K1+K2 selected: M=%d L=%d reset=%s",
            key.k1.tokens,
            key.k1.qlen,
            key.k2.reset,
        )
    else:
        compiled(*args)


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
    projected = attn.in_proj_qkvgfab(hidden_states)[0]
    _original_projected(attn, projected, core)


def _original_projected(attn, projected, core):
    num_tokens = projected.size(0)
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
    core_attn_out = core.view(1, num_tokens, attn.local_num_heads, attn.head_dim)
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


@eager_break_during_capture
def post_projection(attn, projected, core):
    """Fuse a uniform spec batch inside independent CTAs, or run the original ops."""
    from vllm.models.kimi_k3.amd.mono.attention import kda, local_post

    context = get_forward_context()
    md = context.attn_metadata
    graph_mode = attn.compilation_config.cudagraph_mode
    m: Any = md.get(attn.prefix) if isinstance(md, dict) else None
    rows = projected.size(0)
    length = None if m is None else m.uniform_spec_sequence_length
    if (
        (
            context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and (graph_mode is None or not graph_mode.separate_routine())
        )
        or m is None
        or m.num_prefills
        or m.num_decodes
        or m.spec_sequence_masks is None
        or length is None
        or not 1 <= length <= 8
        or m.spec_state_indices_tensor.size(-1) != length
        or rows % length
        or m.num_spec_decodes * length != rows
        or m.num_actual_tokens != rows
        or m.spec_state_indices_tensor.dtype != torch.int32
        or m.spec_state_indices_tensor.stride(-1) != 1
        or m.num_accepted_tokens is None
        or m.num_accepted_tokens.dtype != torch.int32
        or attn.local_num_heads != kda.NH
        or attn.head_dim != kda.HD
        or attn.conv_size != 4
        or projected.dtype != torch.bfloat16
        or projected.stride() != (kda.NPROJ, 1)
        or attn.f_b_proj.weight.dtype != torch.bfloat16
        or attn.o_norm.weight.dtype != torch.bfloat16
        or attn.conv1d.weight.dtype != torch.float32
        or attn.A_log.dtype != torch.float32
        or attn.dt_bias.dtype != torch.float32
        or attn.gate_lower_bound is None
    ):
        _original_projected(attn, projected, core)
        return
    conv, state = attn.kv_cache
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    if (
        conv.dtype != torch.bfloat16
        or state.dtype != torch.float32
        or conv.size(-1) != length + kda.CONV_W - 2
    ):
        _original_projected(attn, projected, core)
        return
    assert state.shape[1:] == (kda.NH, kda.HD, kda.HD)
    assert state[0].is_contiguous()
    local_post.run(
        local_post.KdaPostBuild(
            rows, length, conv.size(-1), attn.o_norm.eps, attn.gate_lower_bound
        ),
        projected,
        {
            "w_fb": attn.f_b_proj.weight,
            "conv_w": attn.conv1d.weight,
            "conv_state": conv,
            "a_log": attn.A_log,
            "dt_bias": attn.dt_bias,
            "on_w": attn.o_norm.weight,
            "rstate": state,
            "st_idx": m.spec_state_indices_tensor,
            "num_acc": m.num_accepted_tokens,
        },
        core,
    )


class MonoK2:
    """Bound to a ``KimiDecoderLayer`` whose MLP is a mono-capable KimiMoE:
    o_proj + its all-reduce + the MLP AttnRes + the MoE in one launch."""

    def __init__(self, layer):
        self.layer = layer
        self.moe = layer.mlp._mono_moe

    def eligible(self, core: torch.Tensor) -> bool:
        return (
            (
                1 <= core.size(0) <= 16
                or core.size(0) in (18, 20, 24, 28, 30, 32, 36, 40, 42, 48)
            )
            and core.dtype == torch.bfloat16
            and core.is_contiguous()
        )

    def build_key(self, rows, reset=False):
        from vllm.models.kimi_k3.amd.mono import layer as k2

        layer = self.layer
        tuned = 2 <= rows <= 48
        wide = rows > 16
        nblocks = layer.prev_valid_blocks + int(reset)
        stream_attn = wide and (rows > 32 or nblocks >= 8)
        front_panels = rows >= 28
        return k2.K2Build(
            tokens=rows,
            nblocks=nblocks,
            reset=reset,
            eps=layer.mlp_res_norm.variance_epsilon,
            out_eps=layer.post_attention_layernorm.variance_epsilon,
            ln_eps=layer.mlp.routed_expert_norm.variance_epsilon,
            producer_latent_quant=tuned,
            balanced_oproj=tuned and rows < 30,
            down_batch=2 if tuned else 5,
            inter_vec=2 if tuned else 1,
            trim_oproj=rows == 6,
            stream_oproj=rows == 6,
            wide=wide,
            compact_lds=wide,
            cache_wide_inter=wide and rows <= 32,
            stream_attn=stream_attn,
            sparse_routes=wide,
            stream_x=front_panels,
            front_panels=front_panels,
            shared_down_first=rows == 48,
            shared_h_once=rows == 48,
            tail_panels=rows == 48,
        )

    def arguments(self, core, prefix_sum, block_residual, out):
        from vllm.models.kimi_k3.amd.mono.attention.kda import HIDDEN

        layer, m = self.layer, self.layer.mlp
        ex = _routed_experts(m)
        return dict(
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
        )

    def forward(self, core, prefix_sum, block_residual, reset=False):
        from vllm.models.kimi_k3.amd.mono import layer as k2

        out = torch.empty_like(prefix_sum)
        k2.k2_launch(
            self.build_key(core.size(0), reset),
            **self.arguments(core, prefix_sum, block_residual, out),
            **runner.launch_args(),
            layer=128 + self.layer.layer_idx,
        )
        return out
