# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One MoE block of a large decode step on gfx942, from vLLM's routing to
this rank's unreduced output. The router and the TP all-reduce stay vLLM's.

Launches, all graph-capturable: pick sort (`moe_sort`), x to MXFP8
(`moe_xq`), routed stage 1 (`moe_ug`), h to MXFP8, routed stage 2
(`moe_down`), combine (`moe_small`), shared expert (`shared`). The weights
are the mono decode layer's gfx942 copies (`mono942`).
"""

import torch

from vllm.models.deepseek_v41.amd.mono.stages.dims import MOE_INTER
from vllm.models.deepseek_v41.amd.mono_decode import MAX_ROWS_GFX942

from . import moe_host as H
from .moe_down import build_moe_down
from .moe_host import EXPERTS, TOPK
from .moe_small import build_combine, build_quant_h
from .moe_sort import MAX_ROWS, build_moe_sort
from .moe_ug import build_moe_ug
from .moe_xq import build_quant_x
from .shared import build_shared_down, build_shared_gate_up


class LargeStepMoE:
    def __init__(self, experts, shared, tp: int, device: torch.device):
        self.tp = tp
        self.inter = MOE_INTER // tp
        self.w13, self.w13_s, self.w2, self.w2_s = experts.mono942
        self.sgu, self.sgu_s = shared.gate_up_proj.mono942
        self.sw2, self.sw2_s = shared.down_proj.mono942
        self.ug = build_moe_ug(tp)
        self.down = build_moe_down(tp)
        self.qh = build_quant_h(self.inter)
        self.qh_nat = build_quant_h(self.inter, even_odd=False)
        self.combine = build_combine()
        self.sort = build_moe_sort()
        self.qx = build_quant_x()
        self.sgu_k = build_shared_gate_up(tp)
        self.sdown_k = build_shared_down(tp)
        # Scratch for the largest step, allocated once so that a captured
        # graph keeps its addresses.
        m, p, i = MAX_ROWS, MAX_ROWS * TOPK, self.inter
        groups = EXPERTS + p // H.GROUP + 1

        def z(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.table = z(groups, 4 + H.GROUP, dtype=torch.int32)
        self.count = z(1, dtype=torch.int32)
        self.h = z(p, i, dtype=torch.bfloat16)
        self.hq = z(p, i, dtype=torch.uint8)
        self.hqs = z(p, i // 32, dtype=torch.uint8)
        self.y = z(p, H.HIDDEN, dtype=torch.bfloat16)
        self.routed = z(m, H.HIDDEN, dtype=torch.bfloat16)
        self.sh = z(m, i, dtype=torch.bfloat16)
        self.shq = z(m, i, dtype=torch.uint8)
        self.shqs = z(m, i // 32, dtype=torch.uint8)
        self.shared = z(m, H.HIDDEN, dtype=torch.bfloat16)
        self.xb = z(m, H.HIDDEN, dtype=torch.uint8)
        self.xs = z(m, H.HIDDEN // 32, dtype=torch.uint8)
        self.xbn = z(m, H.HIDDEN, dtype=torch.uint8)
        self.xsn = z(m, H.HIDDEN // 32, dtype=torch.uint8)

    def __call__(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        s = x.shape[0]
        assert 1 <= s <= MAX_ROWS, s
        p = s * TOPK
        # The current stream, so a CUDA graph captures the launches.
        st = torch.cuda.current_stream()
        ids = topk_ids.to(torch.int32).contiguous()
        wts = topk_weights.float().reshape(-1).contiguous()
        tab, cnt = self.table.data_ptr(), self.count.data_ptr()
        self.sort(ids.data_ptr(), p, tab, cnt, st)
        xb, xs, xbn, xsn = self.xb, self.xs, self.xbn, self.xsn
        x = x.contiguous()
        self.qx(
            x.data_ptr(), xb.data_ptr(), xs.data_ptr(), xbn.data_ptr(),
            xsn.data_ptr(), s, st,
        )  # fmt: skip
        self.ug(
            self.w13.data_ptr(), self.w13_s.data_ptr(), xb.data_ptr(),
            xs.data_ptr(), tab, cnt, self.h.data_ptr(), st,
        )  # fmt: skip
        self.qh(
            self.h.data_ptr(), self.hq.data_ptr(), self.hqs.data_ptr(),
            p * (self.inter // 32), st,
        )  # fmt: skip
        self.down(
            self.w2.data_ptr(), self.w2_s.data_ptr(), self.hq.data_ptr(),
            self.hqs.data_ptr(), wts.data_ptr(), tab, cnt, self.y.data_ptr(), st,
        )  # fmt: skip
        self.combine(self.y.data_ptr(), self.routed.data_ptr(), s, st)
        self.sgu_k(
            self.sgu.data_ptr(), self.sgu_s.data_ptr(), xbn.data_ptr(),
            xsn.data_ptr(), s, self.sh.data_ptr(), st,
        )  # fmt: skip
        self.qh_nat(
            self.sh.data_ptr(), self.shq.data_ptr(), self.shqs.data_ptr(),
            s * (self.inter // 32), st,
        )  # fmt: skip
        self.sdown_k(
            self.sw2.data_ptr(), self.sw2_s.data_ptr(), self.shq.data_ptr(),
            self.shqs.data_ptr(), s, self.shared.data_ptr(), st,
        )  # fmt: skip
        return (self.routed[:s].float() + self.shared[:s].float()).to(torch.bfloat16)


# Smaller steps run the mono decode layer.
MIN_ROWS = MAX_ROWS_GFX942 + 1


class LargeStepHook:
    """`DeepseekV4MoE.largestep`: decides per call and runs the block.

    The router stays vLLM's (`gate` + `fused_topk_bias`, the same top-6 and
    weights the server's path computes). The block's output is this rank's
    unreduced routed + shared sum; the TP all-reduce is vLLM's, as the
    original path does it inside the fused MoE."""

    def __init__(self, moe_layer, tp: int, device: torch.device):
        # Calls that took the block, for tests. A CUDA graph replay does not
        # count, so it counts eager and capture calls only.
        self.steps = 0
        self.block = LargeStepMoE(
            moe_layer.experts.routed_experts, moe_layer.shared_experts, tp, device
        )

    @staticmethod
    def takes(x: torch.Tensor) -> bool:
        return x.dim() == 2 and MIN_ROWS <= x.shape[0] <= MAX_ROWS

    def forward(self, moe_layer, x: torch.Tensor, input_ids) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )

        logits, _ = moe_layer.gate(x)
        topk_w, topk_ids = fused_topk_bias(
            hidden_states=x,
            gating_output=logits,
            scoring_func=moe_layer.scoring_func,
            e_score_correction_bias=moe_layer.gate.e_score_correction_bias,
            topk=moe_layer.n_activated_experts,
            renormalize=moe_layer.renormalize,
            input_tokens=input_ids,
            hash_indices_table=moe_layer.gate.tid2eid,
            routed_scaling_factor=moe_layer.routed_scaling_factor,
        )
        self.steps += 1
        return tensor_model_parallel_all_reduce(self.block(x, topk_ids, topk_w))
