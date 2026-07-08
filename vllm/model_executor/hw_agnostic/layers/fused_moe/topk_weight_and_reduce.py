# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel as mk


class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):
    """Sentinel: experts class defers weight-application + reduction to
    the PrepareAndFinalize side."""

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

        # Skip self-copy when caller aliased fused_out to output earlier.
        if output is fused_expert_output:
            return output

        # MoEPrepareAndFinalizeNoDPEPModular needs the output to be in the `output`
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

        torch.sum(fused_expert_output, dim=1, out=output)
        return output
