# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 TP prefill with a sequence-parallel residual stream (ROCm, TP8+DCP8).

Between two attention cores every per-token op otherwise runs replicated on
all T tokens of every rank, joined by three [T, *] all-reduces per layer
(o_proj, routed latent, MoE output). For batches of at least
VLLM_K3_SP_PREFILL_MIN_TOKENS tokens the residual stream stays token-sharded
instead: rank r owns rows [r R, (r+1) R) of the batch padded to a multiple of
64 (R = padded / TP), and per layer

  attn_res(own rows) -> all-gather -> attention on all tokens (o_proj without
  its all-reduce) -> reduce-scatter -> attn_res(own rows) -> MoE block:
    router, top-k and the routed input projection + MXFP4 quantization on own
    rows -> one all-gather of the packed routing input -> routed experts on all
    tokens (the layer's own modular kernel) -> latent reduce-scatter by column
    blocks (+ per-row sum of squares) beside the shared experts, whose input is
    all-gathered beside the routed experts -> RMSNorm of this rank's latent
    columns and the matching row slice of up_proj, added into the shared
    partial -> one reduce-scatter to own rows.
  The dense layer runs its MLP on all tokens between an all-gather and a
  reduce-scatter.

The bytes on the links stay about the same; what goes is the replicated
per-token work, and two of the collectives run beside compute. Every sum is
fp32 over the TP partials in a fixed order (ops/sp_collectives.py), and the
token-wise math is unchanged. Collectives run on a high-priority side stream.
"""

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.linear import KimiLinearModel

logger = init_logger(__name__)

# Rows are padded to TP x the collectives' 8-row tile.
_ROW_ALIGN = 64
# Workgroups of a collective on the critical path / beside compute.
_CTAS = 128
_CTAS_OVERLAP = 96

_state: "_SPState | None" = None


@triton.jit
def _norm_cols(
    x_ptr, ss_ptr, w_ptr, out_ptr, rows, N, FULL, eps,
    BR: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    """RMSNorm of a column block given each row's full-width sum of squares."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BN)
    m = (r < rows)[:, None] & (c < N)[None, :]
    off = r[:, None].to(tl.int64) * N + c[None, :]
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    rs = tl.rsqrt(tl.load(ss_ptr + r, mask=r < rows, other=1.0) / FULL + eps)
    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + off, (x * rs[:, None] * w[None, :]).to(tl.bfloat16), mask=m)


class _SPState:
    def __init__(self, max_rows: int):
        from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
        from vllm.models.kimi_k3.amd.ops.sp_collectives import SPCollectives

        tp = get_tp_group()
        self.tp, self.rank, self.world = (
            tp,
            get_tensor_model_parallel_rank(),
            tp.world_size,
        )
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.max_rows = max_rows
        # Symmetric buffers: "x" for all-gathers, "p" for reduce-scatter slots
        # and the routing all-gather, "v" for the fp32 vector all-reduce.
        self.comm = SPCollectives(
            tp.device_group.group_name, self.rank, self.world, max_rows, self.dev
        )
        self.side = torch.cuda.Stream(priority=-1)

    def on_side(self, fn) -> torch.cuda.Event:
        """Run fn on the side stream after all work issued so far. Tensors it
        reads must stay referenced until the current stream waits on the
        returned event (the allocator orders reuse per stream only)."""
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            fn()
        ev = torch.cuda.Event()
        ev.record(self.side)
        return ev

    def rs_rows(self, x: torch.Tensor, rows: int) -> torch.Tensor:
        out = torch.empty(
            rows // self.world, x.shape[1], dtype=torch.bfloat16, device=self.dev
        )
        ev = self.on_side(lambda: self.comm.rs_rows(x, "p", _CTAS, rows, out))
        torch.cuda.current_stream().wait_event(ev)
        return out

    def ag(self, x_own, name: str = "x", ctas: int = _CTAS):
        """All-gather of own rows; x_own may be a QuantizedActivation."""
        main = torch.cuda.current_stream()
        if isinstance(x_own, QuantizedActivation):
            R = x_own.data.shape[0]
            T = R * self.world
            data = x_own.data.contiguous().view(torch.uint8).view(torch.bfloat16)
            sc = torch.zeros(R, 64, dtype=torch.bfloat16, device=self.dev)
            sc.view(torch.float32)[:, 0] = x_own.scale.reshape(R).float()
            d_out = torch.empty(T, data.shape[1], dtype=torch.bfloat16, device=self.dev)
            s_out = torch.empty(T, 64, dtype=torch.bfloat16, device=self.dev)

            def fn():
                self.comm.ag_rows(data, name, ctas, d_out)
                self.comm.ag_rows(sc, "p", ctas, s_out)

            main.wait_event(self.on_side(fn))
            scale = s_out.view(torch.float32)[:, 0]
            scale = scale.reshape(T, *x_own.scale.shape[1:]).to(x_own.scale.dtype)
            return QuantizedActivation(
                d_out.view(torch.uint8).view(x_own.data.dtype),
                scale,
                x_own.orig_dtype,
                torch.Size((T, *x_own.orig_shape[1:])),
                x_own.quant_key,
            )
        out, ev = self.ag_async(x_own, name, ctas)
        main.wait_event(ev)
        return out

    def ag_async(self, x_own: torch.Tensor, name: str, ctas: int):
        x_own = x_own.contiguous()
        out = torch.empty(
            x_own.shape[0] * self.world,
            x_own.shape[1],
            dtype=torch.bfloat16,
            device=self.dev,
        )
        return out, self.on_side(lambda: self.comm.ag_rows(x_own, name, ctas, out))


def _first_rows(x, n: int):
    if isinstance(x, QuantizedActivation):
        if x.data.shape[0] == n:
            return x
        return QuantizedActivation(
            x.data[:n],
            x.scale[:n],
            x.orig_dtype,
            torch.Size((n, *x.orig_shape[1:])),
            x.quant_key,
        )
    return x[:n]


class _NoReduce:
    """A RowParallelLinear without its all-reduce for the block's duration."""

    def __init__(self, linear):
        self.linear = linear

    def __enter__(self):
        self.linear.reduce_results = False

    def __exit__(self, *exc):
        self.linear.reduce_results = True


def _moe_supported(moe) -> bool:
    from vllm.models.kimi_k3.amd import latent_moe_runner as lmr

    runner = moe.experts
    if runner._quantized_routed_input is None:
        runner._quantized_routed_input = lmr.enable_quantized_routed_input(runner)
    t = runner.routed_output_transform
    return bool(
        runner._quantized_routed_input
        and runner._tail_shardable
        and runner._up_proj_preshard
        and t is not None
        and t.norm is not None
        and moe.shared_experts is not None
        and runner.routed_scaling_factor == 1.0
    )


def _up_proj_rows(sp: _SPState, runner) -> torch.Tensor:
    """This rank's [hidden, latent / TP] slice of up_proj (the columns that
    multiply its latent block), gathered once from the column shards."""
    w = getattr(runner, "_k3_sp_up_proj", None)
    if w is None:
        t = runner.routed_output_transform
        full = sp.tp.all_gather(t.up_proj.weight.contiguous(), dim=0)
        cw = full.shape[1] // sp.world
        cols = slice(sp.rank * cw, (sp.rank + 1) * cw)
        w = runner._k3_sp_up_proj = full[:, cols].contiguous()
        runner._k3_sp_norm_weight = t.norm.weight[cols].contiguous()
    return w


def _moe(sp: _SPState, moe, y: torch.Tensor, rows: int) -> torch.Tensor:
    """KimiMoE on own rows y [R, H] -> own rows of its reduced output."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    from vllm.model_executor.layers.linear import ReplicatedLinear

    runner = moe.experts
    proj = runner.routed_input_transform
    R, H = y.shape
    T = R * sp.world
    main = torch.cuda.current_stream()

    # Router, top-k and the routed input projection on own rows; one
    # all-gather carries the MXFP4 latent, its scales and the routing.
    logits, _ = moe.gate(y)
    topk_w, topk_ids = runner.router.select_experts(
        hidden_states=y,
        router_logits=logits,
        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
    )
    lat_own = ReplicatedLinear.forward(proj, y)[0]
    vals, scales = per_1x32_mx_quant_hip(
        lat_own, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    L, k = lat_own.shape[1], topk_ids.shape[1]
    fields = (vals, scales, topk_ids.to(torch.int32), topk_w.to(torch.float32))
    widths = (L // 2, L // 32, 4 * k, 4 * k)  # bytes per row
    packed = torch.empty(
        R, cdiv(sum(widths), 128) * 64, dtype=torch.bfloat16, device=sp.dev
    )
    pb, o = packed.view(torch.uint8), 0
    for t, n in zip(fields, widths):
        pb[:, o : o + n] = t.contiguous().view(torch.uint8).view(R, n)
        o += n
    gathered = sp.ag(packed, "p")
    # The shared experts' input, gathered beside the routed experts.
    x, ev_x = sp.ag_async(y, "x", _CTAS_OVERLAP)
    gb, o, parts = gathered.view(torch.uint8), 0, []
    for n in widths:
        parts.append(gb[:, o : o + n].contiguous())
        o += n
    vals_all = parts[0].view(dtypes.fp4x2)
    scales_all = parts[1].view(dtypes.fp8_e8m0)
    ids_all = parts[2].view(torch.int32).to(topk_ids.dtype)
    w_all = parts[3].view(torch.float32).to(topk_w.dtype)

    # Routed experts on all tokens through the layer's modular kernel: its
    # Kimi prepare step asks the projection for the quantized input, which is
    # the gathered one.
    runner.routed_experts._ensure_moe_quant_config_init()
    proj.forward_quantized = lambda a1: (vals_all, scales_all)
    proj.min_sharded_tokens = 0
    try:
        lat = runner.routed_experts.forward_modular(
            x=torch.empty(T, H, dtype=y.dtype, device=sp.dev),
            topk_weights=w_all,
            topk_ids=ids_all,
            shared_experts=None,
            shared_experts_input=None,
        )
    finally:
        del proj.forward_quantized
        del proj.min_sharded_tokens
    lat = lat[..., : runner.moe_config.hidden_dim].contiguous()

    up = _up_proj_rows(sp, runner)
    cw = lat.shape[1] // sp.world
    lat_cols = torch.empty(T, cw, dtype=torch.bfloat16, device=sp.dev)
    ss_part = torch.empty(T, dtype=torch.float32, device=sp.dev)
    ss = torch.empty(T, dtype=torch.float32, device=sp.dev)

    def latent_rs(lat=lat):
        sp.comm.rs_cols(lat, "p", _CTAS_OVERLAP, out=lat_cols, ss=ss_part)
        sp.comm.allreduce_vec(ss_part, "v", out=ss)

    ev_lat = sp.on_side(latent_rs)
    main.wait_event(ev_x)
    shared = moe.shared_experts(x)  # [T, H] partial: reduce_results=False
    main.wait_event(ev_lat)
    lat_norm = torch.empty_like(lat_cols)
    norm = runner.routed_output_transform.norm
    _norm_cols[(triton.cdiv(T, 8),)](
        lat_cols, ss, runner._k3_sp_norm_weight, lat_norm, T, cw, lat.shape[1],
        norm.variance_epsilon, BR=8, BN=triton.next_power_of_2(cw),
    )  # fmt: skip
    shared.addmm_(lat_norm, up.t())
    out = sp.rs_rows(shared, rows)
    # packed, gathered, x and lat were read on the side stream; the waits above
    # ordered those reads before any reuse of their memory.
    del packed, gathered, x, lat
    return out


def _dense_mlp(sp: _SPState, mlp, y, rows: int) -> torch.Tensor:
    x = sp.ag(y)
    with _NoReduce(mlp.down_proj):
        partial = mlp(x)
    return sp.rs_rows(partial, rows)


def _layer(
    sp, lin, layer, positions, prefix_sum, block_residual, prefix_delta, T, rows
):
    """KimiDecoderLayer.forward_attn_residual on own rows."""
    attn_quant_key = (
        layer.self_attn.get_input_quant_key()
        if isinstance(layer.self_attn, (lin.KimiK3DeltaAttention, lin.KimiMLAAttention))
        else None
    )
    x = lin._apply_attn_res(
        prefix_sum,
        block_residual,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
        layer.prev_valid_blocks,
        delta=prefix_delta,
        output_norm=layer.input_layernorm,
        block_write_idx=(layer.block_write_idx if layer.is_block_write_layer else -1),
        quant_key=attn_quant_key,
    )
    if layer.is_block_write_layer:
        prefix_sum = None
    with _NoReduce(layer.self_attn.o_proj):
        partial = layer._run_self_attn(positions, _first_rows(sp.ag(x), T))
    hidden = sp.rs_rows(partial, rows)
    if prefix_sum is None:
        prefix_sum, prefix_delta = hidden, None
    else:
        prefix_delta = hidden
    mlp_valid_blocks = layer.prev_valid_blocks + (
        1 if layer.is_block_write_layer else 0
    )
    y = lin._apply_attn_res(
        prefix_sum,
        block_residual,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
        mlp_valid_blocks,
        delta=prefix_delta,
        output_norm=layer.post_attention_layernorm,
        quant_key=(
            layer.mlp.get_input_quant_key()
            if isinstance(layer.mlp, lin.KimiMLP)
            else None
        ),
    )
    if isinstance(layer.mlp, lin.KimiMoE):
        out = _moe(sp, layer.mlp, y, rows)
    else:
        out = _dense_mlp(sp, layer.mlp, y, rows)
    return prefix_sum, block_residual, out


def eligible(
    model: "KimiLinearModel", num_tokens: int, intermediate_tensors, step
) -> bool:
    if (
        not envs.VLLM_K3_SP_PREFILL
        or num_tokens < envs.VLLM_K3_SP_PREFILL_MIN_TOKENS
        or step is not None
        or intermediate_tensors is not None
    ):
        return False
    ok = getattr(model, "_k3_sp_supported", None)
    if ok is None:
        from vllm.distributed import get_pp_group
        from vllm.models.kimi_k3.amd import linear as lin

        ok = (
            model.config.attn_res_block_size is not None
            and get_pp_group().world_size == 1
        )
        for layer in model.layers[model.start_layer : model.end_layer]:
            if ok and isinstance(layer.mlp, lin.KimiMoE):
                ok = _moe_supported(layer.mlp)
        model._k3_sp_supported = ok
        logger.info_once(
            "Kimi-K3 sequence-parallel prefill %s (batches of >= %d tokens).",
            "enabled" if ok else "unsupported by this configuration",
            envs.VLLM_K3_SP_PREFILL_MIN_TOKENS,
            scope="local",
        )
    return ok


def forward(
    model: "KimiLinearModel",
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
):
    """KimiLinearModel.forward (attn-res, single PP rank) with the residual
    stream token-sharded across TP. Returns full-batch outputs like it."""
    global _state
    from vllm.config import get_current_vllm_config
    from vllm.models.kimi_k3.amd import linear as lin

    hidden = (
        inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids)
    )
    T, H = hidden.shape
    rows = cdiv(T, _ROW_ALIGN) * _ROW_ALIGN
    if _state is None:
        # Symmetric buffers are collective allocations: this first call must
        # be eager (the profile run's max-size batch), never under capture.
        mnbt = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        _state = _SPState(cdiv(max(mnbt, rows), _ROW_ALIGN) * _ROW_ALIGN)
    sp = _state
    assert rows <= sp.max_rows, (rows, sp.max_rows)
    R = rows // sp.world
    lo = min(sp.rank * R, T)
    hi = min(lo + R, T)
    h = torch.zeros(R, H, dtype=hidden.dtype, device=hidden.device)
    h[: hi - lo].copy_(hidden[lo:hi])
    nblocks = cdiv(model.end_layer, model.config.attn_res_block_size)
    residual = h.new_empty(R, nblocks, H)
    prefix_delta = None
    aux = [hidden] if model.start_layer in model.aux_hidden_state_layers else []
    for layer_idx, layer in enumerate(
        model.layers[model.start_layer : model.end_layer], start=model.start_layer
    ):
        h, residual, prefix_delta = _layer(
            sp, lin, layer, positions, h, residual, prefix_delta, T, rows
        )
        if (layer_idx + 1) in model.aux_hidden_state_layers:
            aux.append(sp.ag(h + prefix_delta)[:T])
    out = lin._apply_attn_res(
        h,
        residual,
        model.output_attn_res_proj,
        model.output_attn_res_norm,
        nblocks,
        delta=prefix_delta,
    )
    out = sp.ag(out)[:T]
    if aux:
        return out, aux
    return out
