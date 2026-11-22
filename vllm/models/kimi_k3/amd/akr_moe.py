# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `moe` / `gemm` routes for the Kimi-K3.1 latent MoE decode step (TP8,
M <= 192).

One MoE layer (`KimiMoE.forward`) on the kdamoe kernels:

  router GEMM + sigmoid/bias top-k + BM=16 sort aux
     (moe/latent_moe_common.router_sigmoid_bias_topk)
  -> latent down projection + MXFP4 quant
     (gemm/bf16_common.bf16_latent_down_mxfp4_quant)
  -> shared experts fork (stock KimiMLP, aux stream)   [see `shared_mode`]
  -> own routed experts on the aiter weight layouts
     (moe/latent_moe_common.mxfp4_moe_experts)
  -> shared experts join
  -> latent all-reduce + RMSNorm + up_proj shard
     (moe/latent_moe_common.allreduce_latent_rmsnorm_up_proj)
     (the 16-slot sum folded in for M <= 64)
  -> [M, 7168] TP partial (shared + this rank's up-projected columns); the final
     all-reduce is left to the next layer's fused all-reduce + AttnRes (akr_attnres) or
     done here.

Sub-switches (akr_switch.SWITCHES): ROUTER, DOWN_QUANT, EXPERTS, LATENT_AR, SLOT_FOLD. A
step that is off, or whose entry rejects the layer (the `*_weight_reason` /
`*_unsupported_reason` of each entry, checked at load on the layer's weights at 1 and
`max_rows()` rows), is replaced by its stock counterpart inside the same chain;
`EXPERTS=0` hands the whole layer back to the stock runner, whose tail can still take
the latent AR kernel (`tail_stock_experts`).

Weights: no copies. The router and down-projection weights stay in the checkpoint (row)
layout (measured within 0-4% of the load-time shuffled layout; a shuffled copy would
cost 0.9 GB / 4.7 GB per rank and break the stock fallbacks); the experts' weights are
views of vLLM's aiter-layout parameters (`moe_weights`).

Extra memory per rank (measured, test_akr_moe_layer_ranks8.py): ~170 MiB of main-stream
workspaces (experts slot buffer 118 MB + q 30 MB, router slabs, aux tuples per row
count), lent to each graph-capture stream (`swap_stream_workspaces`,
akr_runtime.register_stream_workspaces) instead of a set per stream, plus the 8-rank
comm buffers (AttnRes mailboxes 14 MiB, latent tail 3.5 MiB).
"""

import dataclasses

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_runtime, akr_switch

logger = init_logger(__name__)

MAX_M = 192
SLOT_FOLD_MAX_M = 64


def _moe():
    from amd_kernels.moe import latent_moe_common

    return latent_moe_common


def _gemm():
    from amd_kernels.gemm import bf16_common

    return bf16_common


@dataclasses.dataclass
class LayerState:
    weights: object  # latent_moe_common.moe_weights
    router_ok: bool
    down_ok: bool
    tail_ok: bool
    norm_weight: torch.Tensor | None
    up_weight: torch.Tensor | None
    eps: float
    hidden: int
    topk: int


def _topk(moe) -> int:
    return moe.experts.moe_config.experts_per_token


# (device index, M) -> aux tuple of the router (shared by all layers: they run in order)
_AUX: dict = {}


def _aux(moe, m: int, device, topk: int) -> dict:
    key = (device.index, m)
    aux = _AUX.get(key)
    if aux is None:
        aux = _AUX[key] = _moe().router_aux(
            moe.gate.weight.data, m, device, topk=topk, zero_moe_buf=False
        )
    return aux


def static_unsupported_reason(moe, config) -> str | None:
    """Config-level reasons (known at construction) this layer cannot run the chain: the
    routing / activation semantics the kernels implement, and an experts shape the entry
    covers."""
    if moe.routed_scaling_factor != 1.0 or not config.moe_renormalize:
        return "needs renormalised top-k with scale 1.0"
    if config.moe_router_activation_func != "sigmoid":
        return "needs the sigmoid router"
    if (
        config.hidden_act != "situ"
        or config.activation_situ_beta != 4.0
        or config.activation_situ_linear_beta != 25.0
    ):
        return "needs SiTU beta 4 / linear beta 25"
    if moe.shared_experts is None or moe.routed_expert_norm is None:
        return "needs shared experts and the latent norm"
    if config.num_expert_group != 1 or config.topk_group != 1:
        return "needs ungrouped routing"
    lib = _moe()
    if not lib.available():
        return "needs gfx950 with FlyDSL"
    w13 = (
        config.num_experts,
        2 * (moe.padded_moe_intermediate_size // moe.tp_size),
        moe.moe_hidden_size // 2,
    )
    if w13 not in lib.mxfp4_moe_experts_shapes():
        return f"no mxfp4_moe_experts for w13 {w13}"
    return None


def setup(moe, config) -> bool:
    """Whether `moe` (KimiMoE) may run the chain; call at the end of
    KimiMoE.__init__."""
    if not (akr_switch.on("EXPERTS") or akr_switch.on("LATENT_AR")):
        return False
    return not akr_config.unsupported_reason(
        "moe", static_unsupported_reason(moe, config)
    )


def _expert_weights(runner):
    r = runner.routed_experts
    w13 = getattr(r, "w13_weight", None)
    if w13 is None:
        w13 = r.w13_weight_packed
    w2 = getattr(r, "w2_weight", None)
    if w2 is None:
        w2 = r.w2_weight_packed
    return w13.data, r.w13_weight_scale.data, w2.data, r.w2_weight_scale.data


# (device index, rows, width) -> zeros for the load-time reason checks
_DUMMY: dict = {}


def _zeros(m: int, n: int, device) -> torch.Tensor:
    key = (device.index, m, n)
    t = _DUMMY.get(key)
    if t is None:
        t = _DUMMY[key] = torch.zeros(m, n, dtype=torch.bfloat16, device=device)
    return t


def _chain_reason(
    moe, weights, router_ok: bool, down_ok: bool
) -> tuple[str | None, bool, bool]:
    """The entries' reasons for this layer's chain at 1 and `max_rows()` rows (zeros on
    the layer's weights): (experts reason, router_ok, down_ok)."""
    lib, gl = _moe(), _gemm()
    runner = moe.experts
    dev = weights.w13.device
    hidden = moe.gate.weight.shape[1]
    latent = moe.moe_hidden_size
    topk = _topk(moe)
    for m in (1, max_rows()):
        x = _zeros(m, hidden, dev)
        if router_ok:
            why = lib.router_sigmoid_bias_topk_unsupported_reason(
                x,
                moe.gate.weight.data,
                moe.gate.e_score_correction_bias.data,
                1.0,
                topk=topk,
                shuffled=False,
                aux=_aux(moe, m, dev, topk),
            ) or lib.moe_sort_from_aux_unsupported_reason(
                _aux(moe, m, dev, topk), weights=weights
            )
            router_ok = not akr_config.unsupported_reason("moe", why)
        if down_ok:
            why = gl.bf16_latent_down_mxfp4_quant_unsupported_reason(
                x, runner.routed_input_transform.weight.data, shuffled=False
            )
            down_ok = not akr_config.unsupported_reason("gemm", why)
        y = _zeros(m, latent, dev)
        why = lib.quantize_moe_activations_unsupported_reason(y, weights=weights)
        if why is None:
            codes, scales = lib.quantize_moe_activations(y, weights=weights)
            ids = torch.zeros(m, topk, dtype=torch.int32, device=dev)
            sort = lib.moe_sort(ids, torch.zeros(m, topk, device=dev), weights=weights)
            why = lib.mxfp4_moe_experts_unsupported_reason(codes, scales, weights, sort)
        if why is not None:
            return why, router_ok, down_ok
    return None, router_ok, down_ok


def _tail_reason(moe, norm_w, up_w, eps) -> str | None:
    """The latent tail entry's reasons for this layer's weights at 1 row, the fused and
    the slot-folded form."""
    lib = _moe()
    comm = akr_runtime.latent_comm
    dev = up_w.device
    shared = _zeros(1, up_w.shape[0], dev)
    partial = _zeros(1, up_w.shape[1], dev)
    return lib.allreduce_latent_rmsnorm_up_proj_unsupported_reason(
        comm, partial, norm_w, up_w, shared, None, eps
    )


def prepare(moe) -> bool:
    """After the weights are loaded and processed: build the layer state (views only);
    returns whether the layer runs the chain (`moe._akr_state`) and/or the stock-experts
    latent tail."""
    runner = moe.experts
    moe._akr_state = None
    transform = runner.routed_output_transform
    up = getattr(transform, "up_proj", None)
    tail_ok = (
        akr_switch.on("LATENT_AR")
        and akr_runtime.latent_comm is not None
        and runner._tail_shardable
        and runner._up_proj_preshard
        and up is not None
        and transform.norm is not None
    )
    norm_w: torch.Tensor | None = None
    up_w: torch.Tensor | None = None
    eps = 0.0
    if tail_ok:
        assert up is not None and transform.norm is not None  # checked above
        norm_w = transform.norm.weight.data
        up_w = up.weight.data
        eps = float(transform.norm.variance_epsilon)
        tail_ok = not akr_config.unsupported_reason(
            "moe", _tail_reason(moe, norm_w, up_w, eps)
        )
    if not tail_ok:
        norm_w = up_w = None
        eps = 0.0
    runner._akr_tail = (norm_w, up_w, eps) if tail_ok else None
    if not akr_switch.on("EXPERTS"):
        return False
    lib, gl = _moe(), _gemm()
    if runner.moe_config.is_sequence_parallel or runner.moe_config.ep_size > 1:
        akr_config.unsupported_reason("moe", "needs plain TP (no EP / SP)")
        return False
    try:
        w = _expert_weights(runner)
    except AttributeError as e:
        akr_config.unsupported_reason("moe", f"experts weights not found ({e!r})")
        return False
    if akr_config.unsupported_reason("moe", lib.moe_weights_unsupported_reason(*w)):
        return False
    weights = lib.moe_weights(*w)
    gate_w, bias = moe.gate.weight, moe.gate.e_score_correction_bias
    router_ok = akr_switch.on("ROUTER") and not akr_config.unsupported_reason(
        "moe",
        lib.router_sigmoid_bias_topk_weight_reason(
            gate_w.data, bias.data, topk=_topk(moe)
        ),
    )
    down = runner.routed_input_transform
    down_ok = (
        akr_switch.on("DOWN_QUANT")
        and down is not None
        and not akr_config.unsupported_reason(
            "gemm",
            gl.bf16_latent_down_mxfp4_quant_weight_reason(
                down.weight.data, getattr(down, "bias", None)
            ),
        )
    )
    if down_ok and akr_switch.shared_mode() == "start":
        # shared experts beside the down projection: its XCC-local slab exchange assumes
        # the dispatcher's workgroup -> XCC mapping, which a concurrent kernel can shift
        gl.latent_down_xcd_local(
            torch.accelerator.current_device_index(), False, w=down.weight.data
        )
    why, router_ok, down_ok = _chain_reason(moe, weights, router_ok, down_ok)
    if akr_config.unsupported_reason("moe", why):
        return False
    moe._akr_state = LayerState(
        weights,
        router_ok,
        down_ok,
        tail_ok,
        norm_w,
        up_w,
        eps,
        gate_w.shape[1],
        _topk(moe),
    )
    return True


def max_rows() -> int:
    """Rows up to which the layer runs the chain (default 192, checked against the
    entries at load: `prepare`).

    The own experts tie the stock stages at 192 rows (0.97-1.00x with skewed routing,
    1.0x uniform) and win up to 128 (1.1-1.4x), so `VLLM_KIMI_AKR_MOE_MAX_M=128` hands
    129..192 rows to the stock experts (their latent all-reduce tail still runs the akr
    kernel)."""
    import os

    try:
        return max(1, min(MAX_M, int(os.environ.get("VLLM_KIMI_AKR_MOE_MAX_M", MAX_M))))
    except ValueError:
        return MAX_M


def eligible(moe, x: torch.Tensor) -> bool:
    return (
        getattr(moe, "_akr_state", None) is not None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == moe._akr_state.hidden
        and 1 <= x.shape[0] <= max_rows()
    )


def _shared(moe, x):
    """Start the shared experts like the stock runner: on the aux stream below
    `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (`VLLM_KIMI_AKR_SHARED_AUX=0`: inline).

    Co-residency: the aux kernels never wait on anything, so the spinning router /
    down-projection / experts kernels on the main stream only ever wait for work that is
    eventually scheduled; the 8-rank tail kernels are launched after `sx.wait()` (see
    `forward`)."""
    sx = moe.experts._shared_experts
    if (
        sx is not None
        and akr_switch.shared_mode() != "inline"
        and sx.maybe_forward_async(x)
    ):
        return sx, None
    return sx, moe.shared_experts(x)


def forward(moe, x: torch.Tensor) -> torch.Tensor:
    """The routed + shared output of one layer, [M, hidden] bf16: TP partial when
    `runner._akr_defer_final` (the next layer's fused all-reduce + AttnRes sums it), the
    all-reduced sum otherwise. Caller checked `eligible`."""
    st: LayerState = moe._akr_state
    runner = moe.experts
    defer_reduce = runner._akr_defer_final
    lib, gl = _moe(), _gemm()
    m = x.shape[0]
    x = x.contiguous()
    dev = x.device

    mode = akr_switch.shared_mode()
    sx, shared = runner._shared_experts, None
    if mode == "start":
        sx, shared = _shared(moe, x)

    # router + sort
    if st.router_ok:
        aux = _aux(moe, m, dev, st.topk)
        lib.router_sigmoid_bias_topk(
            x,
            moe.gate.weight.data,
            moe.gate.e_score_correction_bias.data,
            1.0,
            topk=st.topk,
            shuffled=False,
            aux=aux,
        )
        sort = lib.moe_sort_from_aux(aux, weights=st.weights)
    else:
        logits, _ = moe.gate(x)
        topk_w, topk_ids = runner.router.select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
        )
        sort = lib.moe_sort(topk_ids, topk_w, weights=st.weights)

    # latent down projection (+ MXFP4 quant)
    if st.down_ok:
        codes, scales = gl.bf16_latent_down_mxfp4_quant(
            x, runner.routed_input_transform.weight.data, shuffled=False
        )
    else:
        y = runner.routed_input_transform(x)
        y = y[0] if isinstance(y, tuple) else y
        codes, scales = lib.quantize_moe_activations(y.contiguous(), weights=st.weights)

    if mode != "start":
        # the shared experts fork here (aux stream) so that the placement-sensitive
        # down-projection kernel (and the router) never run beside them
        sx, shared = _shared(moe, x)

    # routed experts
    fold = st.tail_ok and akr_switch.on("SLOT_FOLD") and m <= SLOT_FOLD_MAX_M
    exp_out = lib.mxfp4_moe_experts(codes, scales, st.weights, sort, reduce=not fold)

    # join the shared experts BEFORE the persistent 8-rank tail kernel: no aux-stream
    # kernel may be in flight while it spins (co-residency), and it needs shared_output
    # anyway.
    if shared is None:
        sx.wait()
        shared = sx.output

    if st.tail_ok:
        comm = akr_runtime.latent_comm
        cols = runner._up_proj_shard_size
        view = shared.narrow(1, comm.rank * cols, cols)
        if fold:
            lib.allreduce_slots_rmsnorm_up_proj(
                comm,
                exp_out,
                sort.reverse_sorted,
                st.norm_weight,
                st.up_weight,
                view,
                None,
                st.eps,
            )
        else:
            lib.allreduce_latent_rmsnorm_up_proj(
                comm, exp_out, st.norm_weight, st.up_weight, view, None, st.eps
            )
        return shared if defer_reduce else tensor_model_parallel_all_reduce(shared)
    # stock tail (latent AR + norm + up-projection); it honours runner._akr_defer_final
    return runner._shard_up_proj_tail(exp_out, shared, None)


def tail_stock_experts(
    runner, fused_output: torch.Tensor, shared_output: torch.Tensor
) -> bool:
    """The latent AR + RMSNorm + up-projection kernel for the stock-experts path
    (`ROCmLatentMoERunner._shard_up_proj_tail`): True when it ran (shared_output updated
    in place), False to let the stock tail run."""
    tail = getattr(runner, "_akr_tail", None)
    comm = akr_runtime.latent_comm
    if tail is None or comm is None:
        return False
    lib = _moe()
    norm_w, up_w, eps = tail
    cols = runner._up_proj_shard_size
    if shared_output.dim() != 2 or shared_output.shape[1] < (comm.rank + 1) * cols:
        return False
    args = (
        comm,
        fused_output,
        norm_w,
        up_w,
        shared_output.narrow(1, comm.rank * cols, cols),
        None,
        eps,
    )
    if akr_config.call_unsupported(
        "moe", lib.allreduce_latent_rmsnorm_up_proj_unsupported_reason(*args)
    ):
        return False
    lib.allreduce_latent_rmsnorm_up_proj(*args)
    return True


def swap_stream_workspaces(dev: int, a: int, b: int) -> None:
    """Exchange the router / experts / down-projection workspaces of streams `a` and
    `b` on device `dev` (the entries' dict keys only: nothing allocated or copied)."""
    _moe().swap_stream_workspaces(dev, a, b)
    _gemm().latent_down_swap_stream_workspaces(dev, a, b)


def prewarm_capture_stream(moe) -> None:
    """One eager call of the stream-keyed kernels on the CURRENT stream.

    Covers router slabs / aux ids per row count, experts slots and down-projection
    partials for every graph size <= max rows: it allocates their workspaces and
    compiles every variant outside a capture. Run once on the main stream at load and
    once on the graph-capture stream when `graph_capture` is entered
    (akr_runtime.install_capture_hook); the weights are the layer's own, the inputs
    zeros."""
    st = getattr(moe, "_akr_state", None)
    if st is None:
        return
    runner = moe.experts
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    lib, gl = _moe(), _gemm()
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    sizes = [
        m
        for m in (cfg.compilation_config.cudagraph_capture_sizes if cfg else [])
        if 1 <= m <= MAX_M
    ]
    sizes = sorted(set(sizes) | {16, MAX_M})
    n_experts = moe.gate.weight.shape[0]
    for m in sizes:
        x = torch.zeros(m, st.hidden, dtype=torch.bfloat16, device=dev)
        aux = _aux(moe, m, dev, st.topk)
        if st.router_ok:
            lib.router_sigmoid_bias_topk(
                x,
                moe.gate.weight.data,
                moe.gate.e_score_correction_bias.data,
                1.0,
                topk=st.topk,
                shuffled=False,
                aux=aux,
            )
            sort = lib.moe_sort_from_aux(aux, weights=st.weights)
        else:
            ids = (
                torch.arange(m * st.topk, device=dev, dtype=torch.int32) % n_experts
            ).view(m, st.topk)
            sort = lib.moe_sort(
                ids,
                torch.full((m, st.topk), 1.0 / st.topk, device=dev),
                weights=st.weights,
            )
        if st.down_ok:
            codes, scales = gl.bf16_latent_down_mxfp4_quant(
                x, runner.routed_input_transform.weight.data, shuffled=False
            )
        else:
            codes, scales = lib.quantize_moe_activations(
                torch.zeros(m, moe.moe_hidden_size, dtype=torch.bfloat16, device=dev),
                weights=st.weights,
            )
        fold = st.tail_ok and akr_switch.on("SLOT_FOLD") and m <= SLOT_FOLD_MAX_M
        lib.mxfp4_moe_experts(codes, scales, st.weights, sort, reduce=not fold)


def warm_tail(moe) -> None:
    """Dummy calls of the latent tail entries (compile + first launch on zeros,
    collective: every rank at the same point)."""
    st = getattr(moe, "_akr_state", None)
    comm = akr_runtime.latent_comm
    runner = moe.experts
    tail = getattr(runner, "_akr_tail", None)
    if comm is None or tail is None:
        return
    lib = _moe()
    norm_w, up_w, eps = tail
    dev = norm_w.device
    cols = runner._up_proj_shard_size
    hidden, latent = moe.gate.weight.shape[1], moe.moe_hidden_size
    for m in (16, 64):
        shared = torch.zeros(m, hidden, dtype=torch.bfloat16, device=dev)
        view = shared.narrow(1, comm.rank * cols, cols)
        lib.allreduce_latent_rmsnorm_up_proj(
            comm,
            torch.zeros(m, latent, dtype=torch.bfloat16, device=dev),
            norm_w,
            up_w,
            view,
            None,
            eps,
        )
        if st is not None and akr_switch.on("SLOT_FOLD") and m <= SLOT_FOLD_MAX_M:
            cap = lib.moe_sort_capacity(m, weights=st.weights)
            lib.allreduce_slots_rmsnorm_up_proj(
                comm,
                torch.zeros(cap, latent, dtype=torch.bfloat16, device=dev),
                torch.zeros(m * st.topk, dtype=torch.int32, device=dev),
                norm_w,
                up_w,
                view,
                None,
                eps,
            )
    torch.accelerator.synchronize()
