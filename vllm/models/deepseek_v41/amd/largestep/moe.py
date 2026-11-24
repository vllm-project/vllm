# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One MoE block of a large decode step on gfx942, from vLLM's routing to
this rank's unreduced output. The router and the TP all-reduce stay vLLM's.

Launches, all graph-capturable: pick sort (`moe_sort`), x to MXFP8
(`moe_xq`), routed stage 1 (`moe_ug`), h to MXFP8, routed stage 2
(`moe_down`), combine (`moe_small`). The routed weights are the mono decode
layer's gfx942 copies (`mono942`). The shared expert is the layer's own
(`shared_experts`).
"""

import functools
import weakref

import torch

from vllm.models.deepseek_v41.amd.mono.stages.dims import MOE_INTER
from vllm.models.deepseek_v41.amd.mono_decode import MAX_ROWS_GFX942, MonoDecodeLayer

from .moe_down import build_moe_down
from .moe_small import build_combine, build_quant_h
from .moe_sort import MAX_ROWS, build_moe_sort
from .moe_ug import build_moe_ug
from .moe_xq import build_quant_x
from .shapes import EXPERTS, GROUP, HIDDEN, TOPK

# The steps the block takes: smaller ones run the mono decode layer, larger
# ones vLLM's MoE.
MIN_ROWS = MAX_ROWS_GFX942 + 1


class _Scratch:
    """Scratch of one block call for the largest step (MAX_ROWS rows),
    allocated once so that a captured graph keeps its addresses. All layers of
    a rank share one set (`_scratch`): the layers run one after another on one
    stream. A call returns a view of `routed`, which the next layer's call
    overwrites, so the caller must consume it before then: `LargeStepHook`
    adds the shared expert into a new tensor at once. One set is 22.4 MB at
    TP4 and 21.1 MB at TP8 (the y rows dominate), where one set a layer would
    be about 0.9 GB a rank over 40 layers."""

    def __init__(self, tp: int, device: torch.device):
        m, p, i = MAX_ROWS, MAX_ROWS * TOPK, MOE_INTER // tp
        groups = EXPERTS + p // GROUP + 1

        def z(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.table = z(groups, 4 + GROUP, dtype=torch.int32)
        self.count = z(1, dtype=torch.int32)
        self.h = z(p, i, dtype=torch.bfloat16)
        self.hq = z(p, i, dtype=torch.uint8)
        self.hqs = z(p, i // 32, dtype=torch.uint8)
        self.y = z(p, HIDDEN, dtype=torch.bfloat16)
        self.routed = z(m, HIDDEN, dtype=torch.bfloat16)
        self.xb = z(m, HIDDEN, dtype=torch.uint8)
        self.xs = z(m, HIDDEN // 32, dtype=torch.uint8)


@functools.cache
def _scratch(tp: int, device: torch.device) -> _Scratch:
    return _Scratch(tp, device)


class LargeStepMoE:
    def __init__(self, experts, tp: int, device: torch.device):
        self.tp = tp
        self.inter = MOE_INTER // tp
        self.w13, self.w13_s, self.w2, self.w2_s = experts.mono942
        self.ug = build_moe_ug(tp)
        self.down = build_moe_down(tp)
        self.qh = build_quant_h(self.inter)
        self.combine = build_combine()
        self.sort = build_moe_sort()
        self.qx = build_quant_x()
        self.scratch = _scratch(tp, torch.device(device))

    def __call__(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        s = x.shape[0]
        # A raise, not an assert, so python -O keeps it: moe_sort keeps its
        # pick ids in an LDS array of MAX_ROWS * TOPK, and more picks would
        # overwrite its scan and counts and give a wrong table without an error.
        if not 1 <= s <= MAX_ROWS:
            raise ValueError(f"large-step MoE takes 1 to {MAX_ROWS} rows, got {s}")
        p = s * TOPK
        # The current stream, so a CUDA graph captures the launches.
        st = torch.cuda.current_stream()
        ids = topk_ids.to(torch.int32).contiguous()
        wts = topk_weights.float().reshape(-1).contiguous()
        b = self.scratch
        tab, cnt = b.table.data_ptr(), b.count.data_ptr()
        self.sort(ids.data_ptr(), p, tab, cnt, st)
        xb, xs = b.xb, b.xs
        x = x.contiguous()
        self.qx(x.data_ptr(), xb.data_ptr(), xs.data_ptr(), s, st)
        self.ug(
            self.w13.data_ptr(), self.w13_s.data_ptr(), xb.data_ptr(),
            xs.data_ptr(), tab, cnt, b.h.data_ptr(), st,
        )  # fmt: skip
        self.qh(
            b.h.data_ptr(), b.hq.data_ptr(), b.hqs.data_ptr(),
            p * (self.inter // 32), st,
        )  # fmt: skip
        self.down(
            self.w2.data_ptr(), self.w2_s.data_ptr(), b.hq.data_ptr(),
            b.hqs.data_ptr(), wts.data_ptr(), tab, cnt, b.y.data_ptr(), st,
        )  # fmt: skip
        # A CUDA graph padding row's picks are expert -1, which the sort drops,
        # so the down kernel never writes their y rows: the combine sums
        # whatever an earlier call left there. Those outputs are finite (the
        # scratch starts zeroed and is only ever written with finite values)
        # and the model discards padding rows, so they are not cleared.
        self.combine(b.y.data_ptr(), b.routed.data_ptr(), s, st)
        return b.routed[:s]


class LargeStepHook:
    """`DeepseekV4MoE.largestep`: decides per call and runs the block.

    It takes decode-only steps of MIN_ROWS to MAX_ROWS rows, the same check
    as the mono decode layer (`MonoDecodeLayer.decode_step`). Prefill chunks
    and mixed batches of that size take the original path.

    The router is the layer's own (`experts.router.select_experts`), so the
    top-6 and weights are those of the original path, including the image
    token bias (`bias_vl`). The block's output is this rank's unreduced
    routed + shared sum; the TP all-reduce is vLLM's, as the original path
    does it inside the fused MoE."""

    def __init__(self, layer, moe_layer, tp: int, device: torch.device):
        # The decoder layer, whose attention metadata says whether a step is
        # decode-only. A weak reference: the layer owns this hook.
        self._layer = weakref.ref(layer)
        self.block = LargeStepMoE(moe_layer.experts.routed_experts, tp, device)

    def takes(self, x: torch.Tensor) -> bool:
        # The quantization kernels read x as bf16 bits.
        if x.dtype != torch.bfloat16:
            return False
        if x.dim() != 2 or not MIN_ROWS <= x.shape[0] <= MAX_ROWS:
            return False
        layer = self._layer()
        if layer is None:
            return False
        why = MonoDecodeLayer.decode_step(layer, x.shape[0], MAX_ROWS)
        return why is None

    def forward(self, moe_layer, x: torch.Tensor, input_ids) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        logits, _ = moe_layer.gate(x)
        topk_w, topk_ids = moe_layer.experts.router.select_experts(
            x, logits, input_ids=input_ids
        )
        # The routed sum in bf16 plus the shared expert in bf16, as the
        # original path adds them before its all-reduce.
        out = self.block(x, topk_ids, topk_w) + moe_layer.shared_experts(x)
        return tensor_model_parallel_all_reduce(out)
