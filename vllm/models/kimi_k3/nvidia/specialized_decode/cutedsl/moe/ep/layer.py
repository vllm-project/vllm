# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEP (TP1 attention, EP over all ranks) latent-MoE layer.

`EpMoeLayer` is the per-batch schedule of one MoE layer on a DEP rank (every op
memory-bound at decode, so the schedule's job is to keep HBM streaming):

  main  [shared gate_up | latent down] in ONE split-K GEMM (both read x; one weight
        stream instead of two cuBLAS launches) -> routed experts (`routed`)
  side  FP32 router GEMM + sigmoid top-16 beside the front GEMM; the shared SiTU and
        down-projection beside the routed experts
  main  latent RMSNorm + up-projection accumulated into the shared output (cuBLAS
        addmm: one rounding, as stock)

The routed callable is the deployment's EP kernel: the stock DeepGEMM MegaMoE (one
fused dispatch / FC1 / FC2 / combine kernel; it requires every SM, so the side stream
fills its gaps) or `EpRouted`. Precision is stock: BF16 GEMMs with FP32
accumulation (split-K reduces in FP32), FP32 router logits.

`EpRouted` is the split EP path: NVLink dispatch of the MXFP8 latent (bit-identical
to `mxfp8_quantize`) -> trtllm-gen MXFP4 x MXFP8 experts over the received rows ->
NVLink combine fused with the latent RMSNorm (comm.py)."""

import functools

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...common.host import fake, stream, to_cute
from .comm import LATENT, TOPK, EpBuffers, EpCombine, EpDispatch, EpLayout, stride

SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0


def combine_grid(tokens):
    """CTAs of the combine: enough to push the received rows (~N T / 2 at DEP16) and
    reduce the own tokens in one or two rounds."""
    return 148 if tokens > 64 else 128


@functools.cache
def compile_dispatch(layout_key, tokens, rank, device):
    layout = EpLayout(*layout_key)
    kernel = EpDispatch(layout, tokens, rank)
    args = (
        fake((tokens, LATENT)),
        fake((tokens, TOPK), dtype=torch.int32),
        fake((tokens, TOPK), dtype=torch.float32),
        fake((layout.ranks + 1,), dtype=torch.int64),
    )
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_combine(layout_key, tokens, rank, grid, device):
    layout = EpLayout(*layout_key)
    kernel = EpCombine(layout, tokens, rank, grid)
    args = (
        fake((layout.rows, LATENT)),
        fake((tokens, TOPK), dtype=torch.int32),
        fake((LATENT,)),
        fake((tokens, LATENT)),
        fake((layout.ranks + 1,), dtype=torch.int64),
    )
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


class EpRouted:
    """The routed experts of one DEP rank for every MoE layer (buffers are shared by
    the layers: each layer's combine completes before the next dispatch)."""

    def __init__(self, group, max_tokens, device, *, experts=896):
        self.group, self.device = group, torch.device(device)
        self.ranks, self.rank = group.size(), group.rank()
        self.local = experts // self.ranks
        self.experts = experts
        self.buffers = EpBuffers(group, self.ranks, self.rank, max_tokens, self.device)
        lay = self.buffers.layout
        self.key = (lay.ranks, lay.tokens)
        self.out = torch.empty(lay.rows, LATENT, device=device, dtype=torch.bfloat16)
        self.alpha = torch.full((self.local,), SITU_BETA, device=device)
        self.beta = torch.full((self.local,), SITU_LINEAR_BETA, device=device)
        self.prepared = {}

    def prepare(self, tokens):
        """Every rank of the group must run the same `tokens` in a step (vLLM pads
        DP ranks to one batch): the receive rows are addressed by stride(tokens)."""
        if not 0 < tokens <= self.buffers.layout.tokens:
            raise ValueError(
                f"EpRouted: {tokens} tokens outside the buffer's "
                f"1..{self.buffers.layout.tokens}"
            )
        if tokens not in self.prepared:
            grid = combine_grid(tokens)
            index = self.device.index
            self.prepared[tokens] = (
                compile_dispatch(self.key, tokens, self.rank, index),
                compile_combine(self.key, tokens, self.rank, grid, index),
            )
        return self.prepared[tokens]

    def dispatch(self, latent, ids, weights):
        code, _ = self.prepare(latent.shape[0])
        args = (latent, ids, weights, self.buffers.peers)
        code(*map(to_cute, args), stream(self.device))

    def combine(self, ids, norm_weight, normed):
        _, code = self.prepare(ids.shape[0])
        args = (self.out, ids, norm_weight, normed, self.buffers.peers)
        code(*map(to_cute, args), stream(self.device))
        return normed

    def experts_trtllm(self, tokens, w13, s13, w2, s2):
        """trtllm-gen over the N stride(T) receive rows (empty rows are non-local)."""
        from flashinfer import trtllm_fp4_block_scale_routed_moe
        from flashinfer.tllm_enums import ActivationType, RoutingMethodType

        rows, b = self.ranks * stride(tokens), self.buffers
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(b.ids(rows), b.weights(rows)),
            routing_bias=None,
            hidden_states=b.x(rows),
            hidden_states_scale=b.sf(rows).view(torch.float8_e4m3fn).reshape(-1),
            gemm1_weights=w13,
            gemm1_weights_scale=s13,
            gemm1_bias=None,
            gemm1_alpha=self.alpha,
            gemm1_beta=self.beta,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self.experts,
            top_k=TOPK,
            n_group=None,
            topk_group=None,
            intermediate_size=w2.shape[-1] * 2,
            local_expert_offset=self.rank * self.local,
            local_num_experts=self.local,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize,
            do_finalize=True,
            enable_pdl=True,
            activation_type=ActivationType.Situ.value,
            output=self.out[:rows],
            tune_max_num_tokens=self.buffers.layout.rows,
        )

    def routed(self, latent, ids, weights, norm_weight, normed, experts):
        """Latent [T, 3584] BF16 -> RMSNorm(routed latent) [T, 3584] BF16."""
        self.dispatch(latent, ids, weights)
        self.experts_trtllm(latent.shape[0], *experts)
        return self.combine(ids, norm_weight, normed)


def front_tactic(tokens, rows=64, split=1):
    """Split-K tactic (weight rows, token tile, split, stages) of a DEP projection at
    `tokens` rows (GB300, TP1 MoE shapes, tactic sweep: 64-row
    weight tiles, one K split, the deepest pipeline that fits)."""
    n = next(b for b in (8, 16, 32, 64, 128) if b >= tokens)
    return (rows, n, split, min(8, (200 * 1024) // ((rows + n) * 256)))


# Up to this many tokens the split-K front beats cuBLAS on every DEP projection
# (GB300, [gate_up | down] 39.1 / 40.1 us at T5 / T20 vs 47.7 /
# 47.4 cuBLAS; at T80 43.4 vs 47.5; at T320 114.8 vs 59.4).
FRONT_MAX = 80


def pack_front(moe):
    """One [shared gate_up | latent down] BF16 weight [12288 + 3584, 7168]; the stock
    parameters become views of it (no extra memory)."""
    packed = getattr(moe, "ep_front_weight", None)
    if packed is not None:
        return packed
    gate_up = moe.shared_experts.gate_up_proj.weight
    down = moe.routed_expert_down_proj.weight
    packed = torch.cat([gate_up.data, down.data])
    gate_up.data = packed[: gate_up.shape[0]]
    down.data = packed[gate_up.shape[0] :]
    moe.ep_front_weight = packed
    return packed


class EpMoeLayer:
    """One DEP MoE layer's specialized schedule over the stock module's weights.

    `weights`: gate [896, 7168] BF16 + e_score_correction_bias FP32, front
    (pack_front), shared down [7168, 6144], latent norm [3584], up [7168, 3584]; `act`
    the stock SiTU-and-mul; `routed(latent, topk_weights, topk_ids) -> BF16 latent`
    the EP experts. Batches above FRONT_MAX tokens (or unprepared) run the stock
    order."""

    def __init__(self, weights, act, routed, batches, *, side_stream=None):
        from ...gemm.projection import PreparedProjection

        self.w, self.act, self.routed = weights, act, routed
        self.side = side_stream or torch.cuda.Stream()
        self.events = [torch.cuda.Event() for _ in range(4)]
        fronts = [b for b in batches if b <= FRONT_MAX]
        pol = {b: front_tactic(b) for b in fronts}
        router = {b: front_tactic(b, split=4 if b <= 32 else 2) for b in fronts}
        self.front = PreparedProjection(weights["front"], pol).prepare()
        self.router = PreparedProjection(
            weights["gate"], router, output_dtype=torch.float32
        ).prepare()
        self.sdown = PreparedProjection(weights["shared_down"], pol).prepare()
        self.batches = set(fronts)
        self.inter2 = weights["front"].shape[0] - weights["up"].shape[1]

    def __call__(self, x):
        from vllm import _custom_ops as ops

        w, t = self.w, x.shape[0]
        if t not in self.batches:
            return self.stock_order(x)
        main, ev = torch.cuda.current_stream(), self.events
        # Tensors the main stream consumes are allocated on the main stream (or
        # recorded on it), so the caching allocator never hands their blocks to the
        # side stream while main-stream work still reads them.
        shared = torch.empty_like(x)
        ev[0].record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev[0])
            logits = torch.empty(
                t, w["gate"].shape[0], device=x.device, dtype=torch.float32
            )
            self.router.run(x, logits)
            topk_w, topk_ids = self.topk(x, logits)
            topk_w.record_stream(main)
            topk_ids.record_stream(main)
            ev[1].record(self.side)
        front = torch.empty(t, w["front"].shape[0], device=x.device, dtype=x.dtype)
        self.front.run(x, front)
        ev[2].record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev[2])
            act = self.act(self.gate_up(front))
            self.sdown.run(act, shared)
            ev[3].record(self.side)
        main.wait_event(ev[1])
        routed = self.routed(front[:, self.inter2 :].contiguous(), topk_w, topk_ids)
        normed = torch.empty_like(routed)
        ops.rms_norm(normed, routed, w["norm"], w.get("eps", 1e-5))
        main.wait_event(ev[3])
        return shared.addmm_(normed, w["up"].t())

    def gate_up(self, front):
        """The shared gate_up columns of the packed front as contiguous rows: the
        stock SiTU-and-mul CUDA kernel reads row r at r * 2I, while the packed
        front's rows are 2I + 3584 apart."""
        return front[:, : self.inter2].contiguous()

    def topk(self, x, logits):
        """The stock router's sigmoid top-16 (one group) of the FP32 logits."""
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            fused_grouped_topk,
        )

        w = self.w
        renormalize, scale = w.get("renormalize", True), w.get("scale", 1.0)
        return fused_grouped_topk(
            x, logits, 16, renormalize, w["bias"], 1, 1, "sigmoid", scale
        )

    def stock_order(self, x):
        """Batches the split-K front does not serve: the stock op sequence on one
        stream (cuBLAS GEMMs, FP32 router logits)."""
        from vllm import _custom_ops as ops

        w = self.w
        logits = torch.mm(x, w["gate"].t(), out_dtype=torch.float32)
        topk_w, topk_ids = self.topk(x, logits)
        front = torch.mm(x, w["front"].t())
        shared = torch.mm(self.act(self.gate_up(front)), w["shared_down"].t())
        routed = self.routed(front[:, self.inter2 :].contiguous(), topk_w, topk_ids)
        normed = torch.empty_like(routed)
        ops.rms_norm(normed, routed, w["norm"], w.get("eps", 1e-5))
        return shared.addmm_(normed, w["up"].t())
