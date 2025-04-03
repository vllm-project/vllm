# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


def moe_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> Tuple[int, int, int, int]:
    """
    Extract the MoE problem size from the given tensor arguments:
    - a: The hidden states, input to the MoE layer.
    - w1: The first set of expert weights.
    - w2: The second set of expert weights.
    - topk_ids: The topk ids.

    Note: extracting the problem shape from the weight and activation tensors is
    not obvious.  It needs to be done this way specifically due to subtle issues
    with particular kernels, e.g. the int4 kernels divide the trailing dimension
    by two, so it's not "correct" to extract N or K from the trailing dimension
    of w1 or w2.  Similarly, some kernels transpose the weights, so this needs
    to be kept in mind.
    """
    # Make sure we are using the correct a1 (pre-permute)
    assert topk_ids.shape[0] == a1.shape[0]
    M, _ = a1.shape
    E, N, _ = w1.shape
    K = w2.shape[1]
    topk = topk_ids.shape[1]
    return E, M, N, K, topk


#
# A set of base classes used to make MoE kernels more modular.
#
# Architecture:
# [Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
#
# [Quantize-Dispatch] and [Combine] functionality are bundled into a single
# class `FusedMoEQuantizeDispatchCombine` since they could use collective
# communication mechanisms that need to be consistent.
#
# Ideal architecture:
# [Router] → [Quantize-Dispatch-Permute] → [Experts] → [Unpermute-Combine]
#


class FusedMoEQuantizeDispatchCombine(ABC):
    """
    """

    @abstractmethod
    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform any quantization (and/or) dispatching needed
        for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk_ids.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.

        Returns a tuple of:
        - quantized + dispatched a.
        - quantized + dispatched a1_scales.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        """
        raise NotImplementedError


class FusedMoEPermuteExpertsUnpermute(ABC):

    @abstractmethod
    def workspace_shapes(self, a_dtype: torch.dtype, M: int, N: int, K: int,
                         topk: int,
                         num_experts: int) -> Tuple[int, int, torch.dtype]:
        """
        Compute the number of elements for the temporary outputs of the two
        gemms and activation in the fused expert function.  Since the
        gemms are independent, the workspace for the first gemm can be shared
        with the workspace for the last gemm.

        Returns a tuple of:
        - Number of workspace13 elements: must be large enough to hold the
          result of either expert gemm.
        - Number of workspace2 elements: must be large enough to hold the
          result of the activation function.
        - Workspace type: The dtype to use for the workspace tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> torch.Tensor:
        """
        This function computes the intermediate result of a Mixture of Experts
        (MoE) layer using two sets of weights, w1 and w2.

        Parameters:
        - a1q: (torch.Tensor): The (quantized) input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - w1_scale (Optional[torch.Tensor]): Optional scale to be used for w1.
        - w2_scale (Optional[torch.Tensor]): Optional scale to be used for w2.
        - w1_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w1.
        - w2_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w2.
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1.
        - a2_scale (Optional[torch.Tensor]): Optional scale to be used for a2.
        - workspace13 (torch.Tensor): A scratch tensor used for gemm outputs
          must be large enough to hold output of either MoE gemm.
        - workspace2 (torch.Tensor): A scratch tensor used for the activation
          function.

        Returns:
        - torch.Tensor: The unweighted, unreduced output tensor
        """
        raise NotImplementedError


# Note: only intended for use with a single model layer (due to temp buffers,
# constants, etc.)
class FusedMoEModularKernel(torch.nn.Module):  # should this be a module?

    def __init__(
        self,
        dispatch_combine: FusedMoEQuantizeDispatchCombine,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.dispatch_combine = dispatch_combine
        self.fused_experts = fused_experts

    def forward(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - a1: (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights (torch.Tensor): The topk weights applied at the end of
          the layer.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - inplace (bool): If True, perform the operation in-place.
          Defaults to False.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - w1_scale (Optional[torch.Tensor]): Optional scale to be used for w1.
        - w2_scale (Optional[torch.Tensor]): Optional scale to be used for w2.
        - w1_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w1.
        - w2_zp (Optional[torch.Tensor]): Optional zero points to be used for
          w2.
        - a1_scale (Optional[torch.Tensor]): Optional scale to be used for a1.
        - a2_scale (Optional[torch.Tensor]): Optional scale to be used for a2.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """
        E, M, N, K, top_k = moe_problem_size(a1, w1, w2, topk_ids)

        if global_num_experts == -1:
            global_num_experts = E

        output = a1 if inplace else torch.empty_like(a1)

        workspace13_shape, workspace2_shape, workspace_dtype = (
            self.fused_experts.workspace_shapes(a1.dtype, M, N, K, top_k,
                                                global_num_experts))

        # We can reuse the memory between cache1 and cache3 because by the time
        # we need cache3, we're done with cache1
        workspace13 = torch.empty(workspace13_shape,
                                  device=a1.device,
                                  dtype=workspace_dtype)
        workspace2 = torch.empty(workspace2_shape,
                                 device=a1.device,
                                 dtype=workspace_dtype)

        a1q, a1q_scale = self.dispatch_combine.dispatch(
            a1,
            a1_scale,
            a2_scale,
            topk_ids,
            global_num_experts,
            expert_map,
        )

        fused_out = self.fused_experts.apply(
            a1q,
            w1,
            w2,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1q_scale,
            a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
        )

        self.dispatch_combine.combine(output, fused_out, topk_weights,
                                      topk_ids)

        return output
