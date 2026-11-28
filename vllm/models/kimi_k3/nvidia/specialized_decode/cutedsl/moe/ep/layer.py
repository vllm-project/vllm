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

The routed callable is the deployment's EP kernel (stock DeepGEMM MegaMoE: one
fused dispatch / FC1 / FC2 / combine kernel; it requires every SM, so the side
stream fills its gaps). Precision is stock: BF16 GEMMs with FP32 accumulation
(split-K reduces in FP32), FP32 router logits."""

import torch


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
