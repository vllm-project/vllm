# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel prefill for MiniMax-M3 under pure TP (ROCm).

From ``SP_MIN_TOKENS`` tokens (``SP_MIN_UBATCH_TOKENS`` per DBO microbatch) each rank
keeps ``rows`` tokens of the residual stream. Per layer the two bf16 all-reduces
become an all-to-all of the MXFP8 attention output (each rank applies the whole
o_proj) and a reduce-scatter of the FFN output, each followed by an all-gather of
the next MXFP8 input; norms, quants and the router run on the shard.
"""

import ctypes
from dataclasses import dataclass, field

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# TP4 isolated prefill: 4096 tokens 133 vs 140 ms.
SP_MIN_TOKENS = 4096
# Every microbatch of a DBO step needs this many tokens (their hand-offs alternate,
# so all take one path); below, the phases overlap too little.
SP_MIN_UBATCH_TOKENS = 8192
# The MXFP8 prefill MoE chain's limit.
SP_MAX_TOKENS = 65536
_ROW_ALIGN = 32


def _run(issue) -> None:
    from vllm.v1.worker.ubatching import dbo_tp_collective

    if dbo_tp_collective(issue):
        return
    comm = get_tp_group().device_communicator.pynccl_comm
    issue(comm, torch.cuda.current_stream())


# ops.sp_coll.Pools after init_collectives; None = RCCL.
_POOLS = None
# The pools fit one microbatch of a 65536-token DBO step; larger phases use RCCL.
SP_COLL_MAX_TOKENS = SP_MAX_TOKENS // 2


def collectives_slot_bytes(hidden: int, q_size: int) -> int:
    """Bytes of one pool slot: the FFN phase's inputs at ``SP_COLL_MAX_TOKENS``
    (bf16 partials, codes + scales + routing, two bf16 gathers, slack)."""
    world = 4
    rows = SP_COLL_MAX_TOKENS // world
    padded = rows * world
    assert padded * (q_size + q_size // 32) <= padded * hidden * 2
    return (
        padded * hidden * 2
        + rows * (hidden + hidden // 32 + 64 * 4)
        + 2 * rows * hidden * 2
        + 8 * 256
    )


def init_collectives(
    hidden: int, q_size: int, device: torch.device, sets: int = 2
) -> None:
    """Create the FlyDSL pools, one slot set per DBO microbatch (TP4 on gfx950).
    Collective: every TP rank calls it once."""
    global _POOLS
    if _POOLS is not None:
        return
    from vllm.platforms.rocm import on_gfx950

    world = get_tensor_model_parallel_world_size()
    if world != 4 or not on_gfx950():
        logger.info_once(
            "MiniMax-M3 sequence-parallel prefill: RCCL collectives (the FlyDSL "
            "ones serve TP4 on gfx950)"
        )
        return
    from vllm.models.minimax_m3.amd.ops import sp_coll

    tp = get_tp_group()
    _POOLS = sp_coll.Pools(
        tp.rank_in_group,
        world,
        tp.cpu_group,
        device,
        collectives_slot_bytes(hidden, q_size),
        sets=max(2, sets),
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


class Collectives:
    """RCCL collectives of one communication phase."""

    def __init__(self, comm, stream, world: int):
        self.comm = comm
        self.stream = stream
        self.world = world

    def alloc(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
        """RCCL reads any tensor: the producer allocates."""
        return None

    def all_gather(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        """Each ``[rows, ...]`` shard into its ``[padded, ...]`` output."""
        self.comm.group_start()
        for i, o in zip(ins, outs):
            self.comm.all_gather(_bytes(o), _bytes(i.contiguous()), stream=self.stream)
        self.comm.group_end()

    def reduce_scatter(self, out: torch.Tensor, x: torch.Tensor) -> None:
        """``[padded, H]`` partial sums -> this rank's ``[rows, H]`` sums."""
        self.comm.reduce_scatter(out, x, stream=self.stream)

    def all_to_all(self, out: torch.Tensor, x: torch.Tensor) -> None:
        """Row block r of ``x`` to rank r, row block r of ``out`` from rank r."""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            cudaStream_t,
            ncclDataTypeEnum,
        )

        fn = self.comm.nccl.lib.ncclAllToAll
        if fn.argtypes is None:
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
        xb, ob = _bytes(x), _bytes(out)
        assert xb.numel() == ob.numel() and xb.numel() % self.world == 0
        res = fn(
            xb.data_ptr(),
            ob.data_ptr(),
            xb.numel() // self.world,
            ncclDataTypeEnum.ncclUint8,
            self.comm.comm,
            cudaStream_t(self.stream.cuda_stream),
        )
        assert res == 0, f"ncclAllToAll failed: {res}"


class _Phase:
    """The FlyDSL collectives ``begin_phase`` reserved for the next phase."""

    coll = None


@dataclass(frozen=True)
class SeqShard:
    num_tokens: int
    rows: int
    world: int
    rank: int
    state: _Phase = field(default_factory=_Phase, compare=False, repr=False)

    @property
    def padded(self) -> int:
        return self.rows * self.world

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the full ``[num_tokens, ...]`` ``x`` (zero padded)."""
        start = self.rank * self.rows
        n = min(self.rows, max(0, self.num_tokens - start))
        if n == self.rows:
            return x[start : start + n].contiguous()
        out = x.new_zeros((self.rows, *x.shape[1:]))
        if n:
            out[:n].copy_(x[start : start + n])
        return out

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """``[num_tokens, ...]`` -> ``[padded, ...]``, the padding rows zero."""
        if x.shape[0] == self.padded:
            return x
        out = x.new_empty((self.padded, *x.shape[1:]))
        out[: x.shape[0]].copy_(x)
        out[x.shape[0] :].zero_()
        return out

    def _fly_collectives(self):
        from vllm.models.minimax_m3.amd.ops import sp_coll
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        return sp_coll.Collectives(_POOLS, dbo_current_ubatch_id())

    def begin_phase(self) -> None:
        """Reserve the next phase's pool slot before the compute that fills it, so
        its inputs land in place. Every rank calls it in the same order."""
        if _POOLS is None:
            return
        assert self.state.coll is None, "the previous phase did not run"
        self.state.coll = self._fly_collectives()

    def alloc(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """A view of the next phase's pool slot when it fits, else a new tensor."""
        coll = self.state.coll
        t = coll.alloc(shape, dtype) if coll is not None else None
        if t is None:
            t = torch.empty(shape, dtype=dtype, device=device)
        return t

    def phase(self, fn) -> None:
        """Run ``fn(collectives)`` as one phase (one DBO hand-off). The caller
        allocates the outputs: a block the comm stream allocated could be reused
        while the other microbatch still reads it."""

        def issue(comm, stream):
            rccl = Collectives(comm, stream, self.world)
            coll = self.state.coll
            self.state.coll = None
            if coll is None and _POOLS is not None:
                coll = self._fly_collectives()
            fn(rccl if coll is None else coll.bind(stream, rccl))

        _run(issue)


def plan(num_tokens: int) -> SeqShard | None:
    world = get_tensor_model_parallel_world_size()
    if world == 1 or num_tokens == 0 or num_tokens > SP_MAX_TOKENS:
        return None
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.ubatching import dbo_enabled

    if dbo_enabled():
        slices = get_forward_context().ubatch_slices
        # without the step's split, this microbatch's size
        smallest = num_tokens if slices is None else min(s.num_tokens for s in slices)
        if smallest < SP_MIN_UBATCH_TOKENS:
            return None
    elif num_tokens < SP_MIN_TOKENS:
        return None
    rows = (num_tokens + world - 1) // world
    rows = (rows + _ROW_ALIGN - 1) // _ROW_ALIGN * _ROW_ALIGN
    return SeqShard(num_tokens, rows, world, get_tensor_model_parallel_rank())


# Model glue, called from amd/model.py.

_FP8 = torch.float8_e4m3fn
# DBO microbatches per step (one pool set each), set at model init.
_NUM_UBATCHES = 2


def _mxfp8_kernel(layer):
    """``layer``'s MXFP8 linear kernel when it takes pre-quantized input."""
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    # ModelOpt keeps the kernel on the quant method, Quark on the layer's scheme.
    for kernel in (
        getattr(layer.quant_method, "kernel", None),
        getattr(getattr(layer, "scheme", None), "ocp_mx_linear", None),
    ):
        if isinstance(kernel, RocmDotScaledMxfp8LinearKernel):
            return kernel
    return None


def _linear(layer, x_q: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    return _mxfp8_kernel(layer).apply_quantized(layer, x_q, x_s, torch.bfloat16)


def make_o_proj_full(layer, quant_config, prefix: str):
    """An unsharded copy of ``layer``'s o_proj (+3.1 GB per rank at TP4), or None
    where sequence parallelism cannot run."""
    global _NUM_UBATCHES
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_pp_group
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.platforms.rocm import on_gfx950

    o_proj = layer.self_attn.o_proj
    if (
        not envs.VLLM_ROCM_MINIMAX_M3_SP_PREFILL
        or get_tensor_model_parallel_world_size() == 1
        or get_pp_group().world_size > 1
        or not on_gfx950()
        or _mxfp8_kernel(o_proj) is None
    ):
        return None
    _NUM_UBATCHES = max(2, get_current_vllm_config().parallel_config.num_ubatches)
    return ReplicatedLinear(
        o_proj.input_size,
        o_proj.output_size,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.self_attn.o_proj",
    )


def _moe_quant(moe) -> str | None:
    """The routed experts' format a FlyDSL prefill chain covers ("mxfp8" or
    "mxfp4"), else None."""
    import math

    from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
        AiterMxfp8Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterMxfp4Experts,
    )
    from vllm.models.minimax_m3.amd.ops.moe_mxfp4 import prefill as prefill4
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import prefill as prefill8

    runner = moe.experts
    kernel = getattr(runner._quant_method, "moe_kernel", None)
    cfg = runner.moe_config
    if (
        kernel is None
        or runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
        or cfg.moe_parallel_config.use_ep
        or not math.isclose(cfg.swiglu_alpha or 0.0, prefill8.GEMM1_SWIGLU_ALPHA)
        or not math.isclose(cfg.swiglu_limit or 0.0, prefill8.GEMM1_SWIGLU_LIMIT)
    ):
        return None
    experts = kernel.fused_experts
    hidden, inter = cfg.hidden_dim, cfg.intermediate_size_per_partition
    if isinstance(experts, AiterMxfp8Experts):
        return "mxfp8" if prefill8.supports_shapes(hidden, inter) else None
    routed = runner.routed_experts
    if (
        isinstance(experts, AiterMxfp4Experts)
        and experts.quant_config.use_mxfp4_w4a4
        and prefill4.supports_shapes(hidden, inter)
        and routed.w13_weight.shape[1:] == (2 * inter, hidden // 2)
        and routed.w2_weight.shape[1:] == (hidden, inter // 2)
    ):
        return "mxfp4"
    return None


def _layer_supports_sp(layer) -> bool:
    attn = layer.self_attn
    linears = [attn.qkv_proj, attn.o_proj_full]
    if layer.is_moe_layer:
        moe = layer.block_sparse_moe
        moe._sp_quant = _moe_quant(moe)
        if moe._sp_quant is None:
            return False
        if moe.shared_experts is not None:
            linears.append(moe.shared_experts.gate_up_proj)
    else:
        linears.append(layer.mlp.gate_up_proj)
    return all(
        lin is not None
        and (kernel := _mxfp8_kernel(lin)) is not None
        and kernel.supports_quantized_input(lin)
        for lin in linears
    )


def plan_model(model, hidden_states: torch.Tensor) -> SeqShard | None:
    """The token sharding of a sequence-parallel forward of ``model``, or None."""
    if not envs.VLLM_ROCM_MINIMAX_M3_SP_PREFILL:
        return None
    if getattr(model, "_sp_supported", True) is False:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    sp = plan(hidden_states.shape[0])
    if sp is None:
        return None
    if not hasattr(model, "_sp_supported"):
        layers = model.layers[model.start_layer : model.end_layer]
        model._sp_supported = all(_layer_supports_sp(layer) for layer in layers)
        logger.info(
            "MiniMax-M3 sequence-parallel prefill (from %d tokens, DBO microbatches "
            "from %d): %s",
            SP_MIN_TOKENS,
            SP_MIN_UBATCH_TOKENS,
            "on" if model._sp_supported else "off",
        )
        if model._sp_supported:
            init_collectives(
                hidden_states.shape[1],
                layers[0].self_attn.q_size,
                hidden_states.device,
                sets=_NUM_UBATCHES,
            )
    return sp if model._sp_supported else None


def _input_phase(sp: SeqShard, layer, embeddings: torch.Tensor):
    """First input norm + MXFP8 quant on this rank's rows -> gathered codes,
    scales, residual."""
    from .ops.sp_norm_quant import gemma_rmsnorm_mx

    norm = layer.input_layernorm
    hidden = embeddings.shape[1]
    codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=embeddings.device)
    scales = torch.empty(
        (sp.padded, hidden // 32), dtype=torch.uint8, device=embeddings.device
    )
    box = []

    def run(coll):
        _, r, x_q, x_s = gemma_rmsnorm_mx(
            sp.shard(embeddings),
            None,
            norm.weight,
            norm.variance_epsilon,
            q_out=coll.alloc((sp.rows, hidden), _FP8),
            scale_out=coll.alloc((sp.rows, hidden // 32), torch.uint8),
        )
        coll.all_gather([codes, scales], [x_q, x_s])
        box.append(r)

    sp.phase(run)
    return codes, scales, box[0]


def _o_proj_all_to_all(attn, attn_output: torch.Tensor, sp: SeqShard, coll):
    """This rank's heads of every token -> the whole o_proj of this rank's rows,
    through an all-to-all of the MXFP8 codes + scales."""
    from .ops.sp_pack import mxfp8_quant_pack, mxfp8_unpack

    t, d = attn_output.shape
    shape = (sp.padded, d + d // 32)
    send = coll.alloc(shape, torch.uint8)
    if send is None:
        send = torch.empty(shape, dtype=torch.uint8, device=attn_output.device)
    mxfp8_quant_pack(attn_output, out=send[:t])
    send[t:].zero_()
    unpacked = None
    if hasattr(coll, "all_to_all_unpack"):
        unpacked = coll.all_to_all_unpack(send, d)
    if unpacked is None:
        recv = torch.empty_like(send)
        coll.all_to_all(recv, send)
        unpacked = mxfp8_unpack(recv, sp.world, sp.rows, d)
    return _linear(attn.o_proj_full, *unpacked)


def _routing_width(moe) -> int:
    """Columns of the gathered routing: the top-k ids, then their fp32 weights."""
    fused = moe.n_shared_experts if moe.is_fused_shared_expert_enabled else 0
    return 2 * (moe.experts.router.top_k + fused)


def _route(moe, x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    runner = moe.experts
    router_logits, _ = moe.gate(x)
    topk_weights, topk_ids = runner.router.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
    )
    return torch.cat(
        [topk_ids.to(torch.int32), topk_weights.float().view(torch.int32)],
        dim=1,
        out=out,
    )


def _post_attention_phase(sp: SeqShard, layer, attn_output, residual):
    """o_proj, post-attention norm, FFN input quant and routing on this rank's
    rows -> gathered MXFP8 codes and scales (None when only MXFP4 experts read
    them), routing (None for a dense MLP), MXFP4 codes and scales (MXFP4 experts
    only), residual."""
    from .ops.moe_mxfp4.prefill import mxfp4_quant
    from .ops.sp_norm_quant import MOE_MIN_SCALE_BYTE, gemma_rmsnorm_mx

    hidden = residual.shape[1]
    device = residual.device
    moe = layer.block_sparse_moe if layer.is_moe_layer else None
    mxfp4 = moe is not None and moe._sp_quant == "mxfp4"
    need_fp8 = not mxfp4 or moe.shared_experts is not None
    codes = scales = fp4 = None
    if need_fp8:
        codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=device)
        scales = torch.empty(
            (sp.padded, hidden // 32), dtype=torch.uint8, device=device
        )
    if mxfp4:
        fp4 = (
            torch.empty((sp.padded, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((sp.padded, hidden // 32), dtype=torch.uint8, device=device),
        )
    routing = None
    if moe is not None:
        width = _routing_width(moe)
        routing = torch.empty((sp.padded, width), dtype=torch.int32, device=device)
    norm = layer.post_attention_layernorm
    box = []

    def run(coll):
        h = _o_proj_all_to_all(layer.self_attn, attn_output, sp, coll)
        x, r, x_q, x_s = gemma_rmsnorm_mx(
            h,
            residual,
            norm.weight,
            norm.variance_epsilon,
            min_scale_byte=MOE_MIN_SCALE_BYTE if moe is not None and not mxfp4 else 0,
            store_out=moe is not None,
            q_out=coll.alloc((sp.rows, hidden), _FP8) if need_fp8 else None,
            scale_out=(
                coll.alloc((sp.rows, hidden // 32), torch.uint8) if need_fp8 else None
            ),
        )
        outs, ins = [], []
        if need_fp8:
            outs += [codes, scales]
            ins += [x_q, x_s]
        if mxfp4:
            outs += list(fp4)
            ins += mxfp4_quant(
                x,
                coll.alloc((sp.rows, hidden // 2), torch.uint8),
                coll.alloc((sp.rows, hidden // 32), torch.uint8),
            )
        if moe is not None:
            rt = _route(moe, x, coll.alloc((sp.rows, width), torch.int32))
            outs.append(routing)
            ins.append(rt)
        coll.all_gather(outs, ins)
        box.append(r)

    sp.phase(run)
    return codes, scales, routing, fp4, box[0]


def max_regular_tokens(num_ubatches: int = 1) -> int:
    """The largest forward or microbatch on the regular path. The SP profile run
    skips the modular kernel, so it reserves that MoE workspace before the lock;
    the last microbatch takes up to ``num_ubatches - 1`` extra tokens."""
    return max(SP_MIN_TOKENS, SP_MIN_UBATCH_TOKENS) - 1 + max(num_ubatches, 1) - 1


def _reserve_moe_workspace(moe, num_tokens: int, topk: int) -> None:
    """Size the workspace for a regular MoE call of ``num_tokens`` rows."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    if (
        not is_workspace_manager_initialized()
        or current_workspace_manager().is_locked()
    ):
        return
    runner = moe.experts
    kernel = runner._quant_method.moe_kernel
    routed = runner.routed_experts
    w1, w2 = routed.w13_weight, routed.w2_weight
    hidden = w2.shape[1]
    a1 = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=w1.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=w1.device)
    _, m, n, k, top_k = kernel.fused_experts.moe_problem_size(a1, w1, w2, topk_ids)
    global_num_experts = routed.global_num_experts
    if global_num_experts == -1:
        global_num_experts = w1.shape[0]
    kernel.impl._allocate_buffers(
        a1.dtype,
        a1.device,
        m,
        m,
        n,
        k,
        top_k,
        global_num_experts,
        w1.shape[0],
        None,
        routed.activation,
    )


# The profile run compiles the MoE configs of these sizes, so no request waits.
_WARM_UP_TOKENS = (SP_MIN_TOKENS, 16384, 32768)
_warmed = False


def _moe_experts(sp: SeqShard, moe, codes, scales, routing, fp4) -> torch.Tensor:
    """The MoE of every token -> its TP partial ``[padded, H]``."""
    global _warmed
    runner = moe.experts
    t = sp.num_tokens
    topk = routing.shape[1] // 2
    _reserve_moe_workspace(moe, max_regular_tokens(_NUM_UBATCHES), topk)
    w13, w2 = runner.routed_experts.w13_weight, runner.routed_experts.w2_weight
    hidden = w2.shape[1]
    experts = runner._quant_method.moe_kernel.fused_experts
    if fp4 is None:
        from .ops.moe_mxfp8.prefill import a8w8_prefill_moe as moe_chain

        x_q, x_s = codes, scales
        w13_s, w2_s = experts.w1_scale_val, experts.w2_scale_val
    else:
        from .ops.moe_mxfp4.prefill import a4w4_prefill_moe as moe_chain

        x_q, x_s = fp4
        w13_s, w2_s = experts.quant_config.w1_scale, experts.quant_config.w2_scale

    def chain(m: int, out: torch.Tensor | None) -> torch.Tensor:
        return moe_chain(
            (x_q[:m], x_s[:m]),
            w13,
            w13_s,
            w2,
            w2_s,
            routing[:m, topk:].view(torch.float32),
            routing[:m, :topk],
            hidden_size=hidden,
            intermediate_size=runner.moe_config.intermediate_size_per_partition,
            num_experts=w13.shape[0],
            out=out,
        )

    if not _warmed:
        _warmed = True
        for m in _WARM_UP_TOKENS:
            if m < t:
                chain(m, None)
    out = sp.alloc((sp.padded, hidden), torch.bfloat16, routing.device)
    out[t:].zero_()
    chain(t, out[:t])
    if runner.routed_scaling_factor != 1.0:
        out[:t] *= runner.routed_scaling_factor
    if moe.shared_experts is not None:
        out[:t] += _mlp(moe.shared_experts, codes[:t], scales[:t])
    return out


def _mlp(mlp, x_q: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    """A dense MLP on pre-quantized rows -> its TP partial (no reduce)."""
    from .ops import swiglu_oai_split

    x = swiglu_oai_split(
        _linear(mlp.gate_up_proj, x_q, x_s),
        alpha=mlp.swiglu_alpha,
        beta=mlp.swiglu_beta,
        limit=mlp.swiglu_limit,
    )
    return mlp.down_proj.quant_method.apply(mlp.down_proj, x)


def _ffn(sp: SeqShard, layer, codes, scales, routing, fp4) -> torch.Tensor:
    """The MoE / dense MLP of every token -> its TP partial ``[padded, H]``."""
    if layer.is_moe_layer:
        return _moe_experts(sp, layer.block_sparse_moe, codes, scales, routing, fp4)
    t = sp.num_tokens
    return sp.pad(_mlp(layer.mlp, codes[:t], scales[:t]))


def _ffn_out_phase(sp: SeqShard, model, partial, residual, nxt, want_aux, aux_out):
    """Reduce-scatter of the FFN partial, then ``nxt``'s input norm + quant (the
    final norm after the last layer) on this rank's rows and the gather; an aux
    hidden state is gathered into ``aux_out``."""
    from .ops.sp_norm_quant import gemma_rmsnorm_mx

    hidden = partial.shape[1]
    device = partial.device
    aux = None
    if want_aux:
        aux = torch.empty((sp.padded, hidden), dtype=partial.dtype, device=device)
        aux_out.append(aux)
    scales = None
    if nxt is not None:
        codes = torch.empty((sp.padded, hidden), dtype=_FP8, device=device)
        scales = torch.empty(
            (sp.padded, hidden // 32), dtype=torch.uint8, device=device
        )
    else:
        codes = torch.empty((sp.padded, hidden), dtype=partial.dtype, device=device)
    box = []

    def run(coll):
        h = torch.empty((sp.rows, hidden), dtype=partial.dtype, device=device)
        coll.reduce_scatter(h, partial)
        if nxt is not None:
            norm = nxt.input_layernorm
            _, r, x_q, x_s = gemma_rmsnorm_mx(
                h,
                residual,
                norm.weight,
                norm.variance_epsilon,
                q_out=coll.alloc((sp.rows, hidden), _FP8),
                scale_out=coll.alloc((sp.rows, hidden // 32), torch.uint8),
            )
            outs, ins = [codes, scales], [x_q, x_s]
        else:
            normed, r = model.norm(h, residual)
            outs, ins = [codes], [normed]
        if aux is not None:
            outs.append(aux)
            ins.append(r)
        coll.all_gather(outs, ins)
        box.append(r)

    sp.phase(run)
    return codes, scales, box[0]


def forward(
    model,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor],
    sp: SeqShard,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Sequence-parallel forward of ``model`` from the embeddings to the final
    norm."""
    layers = model.layers[model.start_layer : model.end_layer]
    t = sp.num_tokens
    codes, scales, residual = _input_phase(sp, layers[0], hidden_states)
    aux_gathered: list[torch.Tensor] = []
    for idx, layer in enumerate(layers):
        sp.begin_phase()
        qkv = _linear(layer.self_attn.qkv_proj, codes[:t], scales[:t])
        attn_output = layer.self_attn.attend(positions, qkv)
        codes, scales, routing, fp4, residual = _post_attention_phase(
            sp, layer, attn_output, residual
        )
        sp.begin_phase()
        partial = _ffn(sp, layer, codes, scales, routing, fp4)
        nxt = layers[idx + 1] if idx + 1 < len(layers) else None
        want_aux = model.start_layer + idx + 1 in model.aux_hidden_state_layers
        codes, scales, residual = _ffn_out_phase(
            sp, model, partial, residual, nxt, want_aux, aux_gathered
        )
    aux_hidden_states += [a[:t] for a in aux_gathered]
    if aux_hidden_states:
        return codes[:t], aux_hidden_states
    return codes[:t]
