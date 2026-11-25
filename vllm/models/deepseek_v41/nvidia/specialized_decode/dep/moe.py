# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 local compute with owner-local communication for uniform decode."""

from contextlib import contextmanager
from functools import partial

import torch

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4Mxfp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm import (
    FrontConfig,
    _front_config,
    front_applies,
    padding_mask,
)

from .front import DEPFront
from .owner import OwnerOutput, uniform_owner_layout
from .shared_boundary import LocalMoEOutput


class PreparedDispatch(MoEPrepareAndFinalizeNaiveDPEPModular):
    """Reuse stock dispatch; defer eligible uniform graph combines to mHC."""

    def __init__(self, original):
        if type(original) is not MoEPrepareAndFinalizeNaiveDPEPModular:
            raise ValueError("DEP requires the stock modular AG/RS prepare/finalize")
        if original.is_sequence_parallel or original._lora_context is not None:
            raise ValueError("DEP does not support sequence parallelism or LoRA")
        super().__init__(False, original.num_dispatchers())
        self.pending = None
        self.owner_eligible = False
        self.owner_partial = None

    @contextmanager
    def use_front(self, x, front):
        if self.pending is not None:
            raise RuntimeError("overlapping use of DEP front buffers")
        self.pending = (x, front)
        try:
            yield
        finally:
            self.pending = None

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if self.pending is None:
            return super().prepare(
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )
        x, front = self.pending
        if (
            a1 is not x
            or topk_ids is not front.topk_ids
            or topk_weights is not front.topk_weights
            or apply_router_weight_on_input
            or defer_input_quant
            or quant_config.quant_dtype != "mxfp8"
            or quant_config.is_scale_swizzled
        ):
            raise ValueError("prepared front does not match this modular call")
        # Same tensor order and scale layout as naive_dp_ep.prepare. Never skip
        # this collective on an empty/all-padding rank.
        with self._comm_region():
            dispatched = get_ep_group().dispatch(
                front.x_quant,
                topk_weights,
                topk_ids,
                is_sequence_parallel=False,
                extra_tensors=[front.x_scale],
            )
        assert len(dispatched) == 4
        xq, weights, ids, scales = dispatched
        if len(scales) != 1:
            raise RuntimeError("unexpected EP scale result")
        return xq, scales[0], None, ids, weights

    def finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        dp = get_forward_context().dp_metadata
        sizes = None if dp is None else dp.local_sizes
        if not (
            self.owner_eligible
            and torch.cuda.is_current_stream_capturing()
            and uniform_owner_layout(sizes, output.shape[0], get_ep_group().world_size)
        ):
            return super().finalize(
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            )
        if self.owner_partial is not None:
            raise RuntimeError("unconsumed DEP owner partial")
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        combine_input = (
            fused_expert_output
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP)
            else self._allocate_combine_input(
                (topk_ids.shape[0], fused_expert_output.shape[-1]),
                fused_expert_output.dtype,
                fused_expert_output.device,
            )
        )
        partials = weight_and_reduce_impl.apply(
            output=combine_input,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if partials.shape != (get_ep_group().world_size * output.shape[0], 5120):
            raise RuntimeError("unexpected DEP owner layout")
        self.owner_partial = partials


class DEPMoEStep:
    def __init__(self, ffn, workspace, max_rows=128):
        self.ffn = ffn
        self.runner = runner = ffn.experts
        self.deferrable = False
        routed = runner.routed_experts
        routed._ensure_moe_quant_config_init()
        old = routed.quant_method.moe_kernel
        if (
            old is None
            or old.is_monolithic
            or not isinstance(old.fused_experts, TrtLlmMxfp4Mxfp8ExpertsModular)
            or ffn.use_native_mega_moe
            or runner.do_naive_dispatch_combine
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner.routed_scaling_factor != 1.0
            or runner._fse_fuse_gate
            or runner.gate is None
            or type(runner.router) is not FusedTopKBiasRouter
            or runner.moe_config.hidden_dim != 5120
            or runner.moe_config.hidden_dim_unpadded not in (None, 5120)
        ):
            raise ValueError(
                "unsupported DEP MoE runner; not bypassing its collectives"
            )
        quant = old.fused_experts.quant_config
        if quant.quant_dtype != "mxfp8" or quant.is_scale_swizzled:
            raise ValueError("DEP requires MXFP8 activations with linear scales")
        placement = runner.expert_placement_strategy
        if getattr(placement, "value", placement) != "linear":
            raise ValueError("DEP requires contiguous expert ownership")
        # TRTLLM computes its first global expert as ep_rank * local_num_experts.
        local_experts = routed.w13_weight.shape[0]
        if (
            routed.w2_weight.shape[0] != local_experts
            or local_experts * get_ep_group().world_size != routed.global_num_experts
        ):
            raise ValueError("DEP requires equally sized contiguous expert shards")
        self.prepare = PreparedDispatch(old.prepare_finalize)
        # Keep main's complete local routing/GEMM/finalize implementation.
        self.kernel = FusedMoEKernel(self.prepare, old.fused_experts)
        self.front = None
        front = _front_config(runner)
        if isinstance(front, FrontConfig):
            self.front = DEPFront(front, workspace)
            for rows in (8, 16, 32, 64, 128, 192, 384):
                if rows <= max_rows:
                    self.front.warmup(rows)
        self.front_config = front if isinstance(front, FrontConfig) else None
        self.max_rows = max_rows
        self.owner = None
        self.communication = None

    def _run_direct(self, x, input_ids):
        from .communication import gemm_args

        communication = self.communication
        assert communication is not None and self.owner is not None
        assert self.front is not None
        runner, rows = self.runner, x.shape[0]
        channel = communication.channels[rows]
        shared = runner._shared_experts
        overlapping = shared is not None and shared.maybe_forward_async(x)
        try:
            logits, _ = runner.gate(x)
            runner._maybe_apply_shared_experts(x, SharedExpertsOrder.NO_OVERLAP)
            mask = padding_mask(rows)
            if not (
                front_applies(self.front_config, runner)
                and self.front.accepts(x, logits, input_ids, mask)
            ):
                raise RuntimeError("DEP front changed after common admission")
            self.front(x, logits, input_ids, mask, output=channel.local_output)
            data = channel.gather_before_owner()
            experts = self.kernel.fused_experts
            assert isinstance(experts, TrtLlmMxfp4Mxfp8ExpertsModular)
            if data.x_quant.shape[0] > experts._max_supported_tokens(
                6, runner.routed_experts.global_num_experts
            ):
                raise RuntimeError("deferred DEP experts cannot split the call")
            routed = experts._invoke_kernel(None, *gemm_args(self, data))
            if not isinstance(routed, UnfinalizedMoEOutput):
                raise TypeError("TRTLLM did not return deferred finalize inputs")
        finally:
            if overlapping:
                shared.wait()
        return OwnerOutput(
            communication.placeholders[rows],
            None if shared is None else shared.output,
            self.owner,
            partial(communication.publishers[rows], routed),
        )

    def run(self, x, input_ids, step):
        runner = self.runner
        if self.prepare.owner_partial is not None:
            raise RuntimeError("previous finalize did not hand off its partial")
        # This fallback still runs the original modular AG/RS sequence. It must
        # not be replaced with a local no-op, including on an idle DP rank.
        if not 0 < x.shape[0] <= self.max_rows:
            return self.ffn(x, input_ids, None)
        if self.communication is not None and torch.cuda.is_current_stream_capturing():
            with runner._sequence_parallel_context():
                dp = get_forward_context().dp_metadata
                sizes = None if dp is None else dp.local_sizes
                shared = runner._shared_experts
                if (
                    x.shape[0] % 2 == 0
                    and uniform_owner_layout(
                        sizes, x.shape[0], get_ep_group().world_size
                    )
                    and (
                        shared is None
                        or shared._determine_shared_experts_order(x)
                        != SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
                    )
                ):
                    return self._run_direct(x, input_ids)
        shared = runner._shared_experts
        overlapping = shared is not None and shared.maybe_forward_async(x)
        try:
            logits, _ = runner.gate(x)
            # Even at SP1 this populates DPMetadata.local_sizes for AG/RS.
            with runner._sequence_parallel_context():
                runner._maybe_apply_shared_experts(x, SharedExpertsOrder.NO_OVERLAP)
                mask = padding_mask(x.shape[0])
                front = None
                if (
                    self.front is not None
                    and front_applies(self.front_config, runner)
                    and self.front.accepts(x, logits, input_ids, mask)
                ):
                    front = self.front(x, logits, input_ids, mask)
                    weights, ids = front.topk_weights, front.topk_ids
                else:
                    weights, ids = runner.router.select_experts(
                        hidden_states=x,
                        router_logits=logits,
                        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
                        input_ids=input_ids,
                    )
                routed = runner.routed_experts
                args = dict(
                    hidden_states=x,
                    w1=routed.w13_weight,
                    w2=routed.w2_weight,
                    topk_weights=weights,
                    topk_ids=ids,
                    activation=routed.activation,
                    global_num_experts=routed.global_num_experts,
                    expert_map=routed.expert_map,
                    apply_router_weight_on_input=routed.apply_router_weight_on_input,
                    shared_experts=shared,
                    shared_experts_input=x,
                )
                if front is None:
                    result = self.kernel.apply(**args)
                else:
                    with self.prepare.use_front(x, front):
                        result = self.kernel.apply(**args)
        except BaseException:
            self.prepare.owner_partial = None
            raise
        finally:
            if overlapping:
                shared.wait()
        partials = self.prepare.owner_partial
        self.prepare.owner_partial = None
        if partials is not None:
            return OwnerOutput(
                partials,
                None if shared is None else shared.output,
                self.owner,
            )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("EP combine must finish before the next mHC boundary")
        # TP1: the shared expert is local; EP combine already returned this
        # rank's token rows. No TP partial is sent to the next boundary.
        return result if shared is None else LocalMoEOutput(result, shared.output)
