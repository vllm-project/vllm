# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned EP buffers and rank-local graph preparation."""

from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4Mxfp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
    empty_quantized_layer_input,
)
from vllm.utils import flashinfer as fi_utils

from .owner import EPReduceScatterMHC
from .transport import FinalizeOwnerPublish, PeerFrontDispatch

if TYPE_CHECKING:
    from .moe import DEPMoEStep


def gemm_args(moe, data):
    xq, scales, ids, weights = data
    layer = moe.runner.routed_experts
    local_experts = layer.w13_weight.shape[0]
    return (
        xq,
        scales.view(torch.float8_e4m3fn),
        ids,
        weights,
        layer.w13_weight,
        layer.w2_weight,
        layer.activation,
        layer.global_num_experts,
        local_experts,
        get_ep_group().rank_in_group * local_experts,
        6,
    )


class DEPCommunication:
    def __init__(self, plans, config, target=None):
        self.plans, self.config = plans, config
        self.group = get_ep_group().cpu_group
        self.device = plans[0].layer.hc_attn_fn.device
        self.owner = (
            EPReduceScatterMHC(self.group, self.device, config.max_rows)
            if target is None
            else target.owner
        )
        self.channels, self.publishers, self.placeholders = {}, {}, {}
        self.storage = None
        self.warmed = False
        self.prepared = set()
        for index, plan in enumerate(plans):
            # A post-only boundary cannot consume a fused norm/quant result.
            eligible = index + 1 < len(plans) and plans[index + 1].engram is None
            plan.moe.prepare.owner_eligible = eligible
            plan.moe.owner = self.owner
            if target is None and eligible and plan.moe.front is not None:
                if plan.moe.runner.routed_experts.apply_router_weight_on_input:
                    raise ValueError("DEP transport requires output router weights")
                plan.moe.communication = self
        if any(plan.moe.communication is self for plan in plans):
            self.storage = PeerFrontDispatch(
                self.device,
                self.group,
                config.max_rows,
            )

    def _warm_deferred(self):
        if self.warmed:
            return
        representatives: dict[tuple, DEPMoEStep] = {}
        for plan in self.plans:
            moe = plan.moe
            if moe.communication is not self:
                continue
            layer = moe.runner.routed_experts
            key = (
                type(moe.kernel.fused_experts),
                layer.w13_weight.shape,
                layer.w2_weight.shape,
                layer.w13_weight.dtype,
                layer.w2_weight.dtype,
                layer.activation,
                layer.global_num_experts,
            )
            representatives.setdefault(key, moe)
        total = self.config.max_rows * self.owner.width
        with (
            torch.inference_mode(),
            fi_utils.autotune(
                tune_mode=self.config.autotune,
                tuning_buckets=fi_utils.flashinfer_get_hybrid_num_tokens_buckets(total),
                round_up=True,
            ),
        ):
            xq = torch.zeros(
                (total, 5120), dtype=torch.float8_e4m3fn, device=self.device
            )
            scales = torch.full(
                (total, 160), 127, dtype=torch.uint8, device=self.device
            )
            indices = torch.arange(
                total * 6, dtype=torch.int32, device=self.device
            ).view(total, 6)
            weights = torch.full(
                (total, 6), 1 / 6, dtype=torch.float32, device=self.device
            )
            for moe in representatives.values():
                ids = indices % moe.runner.routed_experts.global_num_experts
                # Only local expert GEMMs: shape preparation can differ in order
                # across ranks and must not enter an EP collective.
                experts = moe.kernel.fused_experts
                assert isinstance(experts, TrtLlmMxfp4Mxfp8ExpertsModular)
                experts._invoke_kernel(
                    None,
                    *gemm_args(moe, (xq, scales, ids, weights)),
                )
        self.warmed = True

    def prepare_rows(self, rows):
        if rows in self.prepared:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DEP communication must be prepared before capture")
        total = self.owner.width * rows
        partials = torch.empty((total, 5120), dtype=torch.bfloat16, device=self.device)
        residual = torch.empty(
            (rows, 4, 5120), dtype=torch.bfloat16, device=self.device
        )
        post = torch.empty((rows, 4, 1), dtype=torch.float32, device=self.device)
        comb = torch.empty((rows, 4, 4), dtype=torch.float32, device=self.device)
        pre = torch.empty((rows, 4), dtype=torch.float32, device=self.device)
        shared = torch.empty((rows, 5120), dtype=torch.bfloat16, device=self.device)
        q = empty_quantized_layer_input(rows, 5120, self.device)
        for quant in (None, (q.data.view(torch.uint8), q.scale)):
            for aux in (False, True):
                for shared_input in (None, shared):
                    self.owner.prepare(
                        partials,
                        residual,
                        post,
                        comb,
                        pre,
                        self.plans[0].layer.attn_norm,
                        quant,
                        aux,
                        shared=shared_input,
                    )
        if self.storage is not None and rows % 2 == 0:
            self._warm_deferred()
            self.channels[rows] = PeerFrontDispatch(
                self.device,
                self.group,
                rows,
                storage=self.storage,
            )
            template = UnfinalizedMoEOutput(
                torch.empty((1, 5120), dtype=torch.bfloat16, device=self.device),
                torch.empty((total, 6), dtype=torch.float32, device=self.device),
                torch.empty((total, 6), dtype=torch.int32, device=self.device),
            )
            self.publishers[rows] = FinalizeOwnerPublish(self.owner, template, rows)
            self.placeholders[rows] = partials
        self.prepared.add(rows)
