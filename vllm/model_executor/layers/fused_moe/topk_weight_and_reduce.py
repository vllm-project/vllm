# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk


class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):
    """
    Useful in the case when some FusedMoEExpertsModular
    implementation does not perform weight application and reduction
    but cannot address the needs of all the compatible PrepareAndFinalize
    implementations.
    For example, BatchedTritonExperts is compatible with both
    PplxPrepareAndFinalize and BatchedPrepareAndFinalize. PplxPrepareAndFinalize
    does the weight-application + reduction as part of the pplx combine kernel.
    But the BatchedPrepareAndFinalize needs an implementation. To facilitate
    this case, the BatchedTritonExperts could use TopKWeightAndReduceDelegate
    so the PrepareAndFinalize implementations could choose how to
    weight + reduce.
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceDelegate)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError(
            "The caller is expected to choose an appropriate "
            "TopKWeightAndReduce implementation."
        )


class TopKWeightAndReduceNoOP(mk.TopKWeightAndReduce):
    """
    The fused_experts outputs have already been weight applied and reduced.
    This implementation is a no-op.
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceNoOP)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # Weight application and reduction operations are already done.
        if output is None:
            return fused_expert_output

        # MoEPrepareAndFinalizeNoEP needs the output to be in the `output`
        # tensor.
        assert output.size() == fused_expert_output.size(), (
            "output shape is expected to match the fused_expert_output shape. "
            f"But got output={output.size()}, "
            f"used_expert_output={fused_expert_output.size()}"
        )
        output.copy_(fused_expert_output, non_blocking=True)
        return output


class TopKWeightAndReduceContiguous(mk.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (m, topk, K)
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceContiguous)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        m, num_topk = topk_ids.size()
        k = fused_expert_output.size(-1)
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(m, num_topk, k)

        assert fused_expert_output.size() == (m, num_topk, k), (
            f"Expected fused_expert_output size {(m, num_topk, k)}. But got "
            f"{fused_expert_output.size()}"
        )

        if not apply_router_weight_on_input:
            fused_expert_output.mul_(topk_weights.view(m, -1, 1))

        if output is None:
            output = torch.empty(
                (m, k),
                device=fused_expert_output.device,
                dtype=fused_expert_output.dtype,
            )
        assert output.size() == (m, k), (
            f"Expected output size {(m, k)}. But got {output.size()}"
        )

        ops.moe_sum(fused_expert_output, output)
        return output


class TopKWeightAndReduceNaiveBatched(mk.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (num_experts, batch_size, K)
    """

    def __init__(self, rank: int):
        self.rank = rank

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceNaiveBatched) and (
            other.rank == self.rank
        )

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        assert fused_expert_output.ndim == 3
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        K = fused_expert_output.size(-1)

        if output is None:
            output = torch.zeros(
                (num_tokens, K),
                device=fused_expert_output.device,
                dtype=fused_expert_output.dtype,
            )
        else:
            output.fill_(0)

        assert output.size() == (num_tokens, K), (
            f"Expected output size {(num_tokens, K)}, but got {output.size()}"
        )

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            matching_tokens = topk_ids == expert_id
            topks = torch.any(matching_tokens, dim=1).flatten()
            rows = torch.count_nonzero(topks)
            rhs = fused_expert_output[expert_id - first_expert, :rows, :]
            if not apply_router_weight_on_input:
                rhs.mul_(topk_weights[matching_tokens].view(rhs.size(0), 1))
            output[topks] = output[topks] + rhs

        return output
