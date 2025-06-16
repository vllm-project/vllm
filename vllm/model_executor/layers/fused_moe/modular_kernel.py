# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from math import prod
from typing import Optional

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.utils import cdiv

#
# This file defines a set of base classes used to make MoE kernels more modular.
# The goal is to be able to utilize different communication mechanisms with
# any fused MoE kernel without needing to have combinatoric implementations.
#
# The fused moe kernels are broken down into the following components:
#
# [Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
#
# Each component will be independent of the others except for
# [Quantize-Dispatch] and `[Combine] (see below). The components can then be
# mixed and matched with so that DP+EP can be supported easily for multiple
# MoE kernel implementations.
#
# The following main classes are defined:
# * FusedMoEPrepareAndFinalize - an abstract base class for preparation of MoE
#   inputs (e.g. quantization, distribution) and finalization of Moe outputs.
#   The prepare method must take care of any needed quantization and the
#   finalize method must apply weights and do the final reduction of the output.
# * FusedMoEPermuteExpertsUnpermute - an abstract base class for the main fused
#   MoE operation. One important feature to note is that this class does not
#   apply topk weights or reduce the final output.
# * FusedMoEModularKernel - an interface class that combines a
#   FusedMoEPrepareAndFinalize and a FusedMoEPermuteExpertsUnpermute to
#   provide the standard fused MoE kernel interface.
#
# [Quantize-Prepare] and [Finalize] functionality are bundled into a single
# class `FusedMoEPrepareAndFinalize` since they could use collective
# communication mechanisms that need to be consistent.
#


def _moe_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
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
    assert w1.dim() == 3 and w2.dim() == 3
    E, N, _ = w1.size()
    K = w2.size(1)

    if a1.dim() == 2:
        # Make sure we are using the correct a1 (pre-permute).
        assert topk_ids.size(0) == a1.size(0), \
            f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)
    else:
        assert a1.dim() == 3
        assert a1.size(0) == E, f"{a1.size(0)} == {E}"
        M = a1.size(1)  # This is max_num_tokens

    assert topk_ids.dim() == 2
    topk = topk_ids.size(1)

    return E, M, N, K, topk


class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform any quantization (and/or) dispatching needed
        for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.

        Returns a tuple of:
        - quantized + dispatched a.
        - quantized + dispatched a1_scales.
        - Optional tensor as big as number of local experts that contains the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        """
        raise NotImplementedError

    @abstractmethod
    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        """
        The PrepareFinalize All2All implementations generally constrain the
        dtype of the topk_ids they support. This function returns the
        required topk indices dtype so it can be respected.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> Optional[int]:
        """
        Some PrepareFinalize All2All implementations are batched. Meaning,
        they can processes only as set of tokens at a time. This
        function returns the batch size i.e the maximum number of tokens
        the implementation can process at a time.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError


class FusedMoEPermuteExpertsUnpermute(ABC):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
    above.
    """

    # TODO (bnell): make this return a CHUNK_SIZE or None instead?
    @abstractmethod
    def supports_chunking(self) -> bool:
        """
        A flag indicating whether or not this class supports activation
        chunking.
        """
        raise NotImplementedError

    @abstractmethod
    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        raise NotImplementedError

    def activation(self, activation: str, output: torch.Tensor,
                   input: torch.Tensor) -> None:
        assert output.size(-1) * 2 == input.size(-1)
        if activation == "silu":
            torch.ops._C.silu_and_mul(output, input)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(output, input)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    def enable_chunking(self):
        return envs.VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING and \
          self.supports_chunking()

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
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
        expert_num_tokens: Optional[torch.Tensor],
    ):
        """
        This function computes the intermediate result of a Mixture of Experts
        (MoE) layer using two sets of weights, w1 and w2.

        Parameters:
        - output: (torch.Tensor): The unweighted, unreduced output tensor.
        - hidden_states: (torch.Tensor): The (quantized) input tensor to the MoE
          layer.
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
        - expert_num_tokens: An optional tensor containing the number of tokens
          assigned to each expert when using batched experts format input.
        """
        raise NotImplementedError


def _chunk_scales(scales: Optional[torch.Tensor], start: int,
                  end: int) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            return scales
        else:
            return scales[start:end]
    return None


class FusedMoEModularKernel(torch.nn.Module):
    """
    This class combines a FusedMoEPrepareAndFinalize instance and
    a FusedMoEPermuteExpertsUnpermute to provide an interface that
    is compatible with the `fused_experts` function in fused_moe.py.

    It takes care of managing any required scratch space.

    Note: Instances of this class should only be used for a single model
    layer due to any layer specific state that may be used by the component
    objects.
    """

    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
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
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states: (torch.Tensor): The input tensor to the MoE layer.
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
        - apply_router_weight_on_input (bool): When true, the topk weights are
          applied directly on the inputs. This is only applicable when topk is
          1.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """

        a1 = hidden_states
        output = a1 if inplace else torch.zeros_like(a1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        (a1q, a1q_scale, expert_num_tokens, _expert_topk_ids,
         _expert_topk_weights) = self.prepare_finalize.prepare(
             a1, a1_scale, a2_scale, topk_weights, topk_ids,
             global_num_experts, expert_map, apply_router_weight_on_input)

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (topk_weights if _expert_topk_weights is None else
                        _expert_topk_weights)

        fused_out = None

        if a1q.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            fused_out = torch.empty_like(a1q).to(dtype=a1.dtype)
        else:
            _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

            if self.fused_experts.enable_chunking():
                CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
                num_chunks = cdiv(M, CHUNK_SIZE)
            else:
                CHUNK_SIZE = M
                num_chunks = 1

            if num_chunks == 1:
                (workspace13_shape, workspace2_shape, fused_out_shape,
                 workspace_dtype) = self.fused_experts.workspace_shapes(
                     a1, a1q, M, N, K, top_k, global_num_experts,
                     local_num_experts)
            else:
                # Use the full M to get the final output shape.
                _, _, fused_out_shape, _ = (
                    self.fused_experts.workspace_shapes(
                        a1, a1q, M, N, K, top_k, global_num_experts,
                        local_num_experts))
                # Use the CHUNK_SIZE to get the workspace shapes.
                workspace13_shape, workspace2_shape, _, workspace_dtype = (
                    self.fused_experts.workspace_shapes(
                        a1, a1q, CHUNK_SIZE, N, K, top_k, global_num_experts,
                        local_num_experts))

            # We can reuse the memory between cache1 and cache3 because by the
            # time we need cache3, we're done with cache1.
            workspace13 = torch.empty(prod(workspace13_shape),
                                      device=a1.device,
                                      dtype=workspace_dtype)
            workspace2 = torch.empty(prod(workspace2_shape),
                                     device=a1.device,
                                     dtype=workspace_dtype)

            if num_chunks == 1:
                fused_out = _resize_cache(workspace13, fused_out_shape)

                self.fused_experts.apply(
                    fused_out,
                    a1q,
                    w1,
                    w2,
                    topk_ids,
                    activation=activation,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=w1_zp,
                    w2_zp=w2_zp,
                    a1q_scale=a1q_scale,
                    a2_scale=a2_scale,
                    workspace13=workspace13,
                    workspace2=workspace2,
                    expert_num_tokens=expert_num_tokens,
                )
            else:
                # The leading output dimension may not be equal to M, so
                # we compute output indices separately.
                M_out = fused_out_shape[0]
                assert M_out >= M
                factor = M_out // M
                assert factor > 0
                OUT_CHUNK_SIZE = CHUNK_SIZE * factor

                fused_out = torch.empty(fused_out_shape,
                                        device=a1q.device,
                                        dtype=workspace_dtype)

                assert cdiv(M_out, OUT_CHUNK_SIZE) == num_chunks, (
                    f"{cdiv(M_out, OUT_CHUNK_SIZE)} == {num_chunks}")

                for chunk in range(num_chunks):
                    begin_chunk_idx = chunk * CHUNK_SIZE
                    end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, M)
                    begin_out_idx = chunk * OUT_CHUNK_SIZE
                    end_out_idx = min((chunk + 1) * OUT_CHUNK_SIZE, M_out)
                    curr_a1q = a1q[begin_chunk_idx:end_chunk_idx]
                    curr_a1q_scale = _chunk_scales(a1q_scale, begin_chunk_idx,
                                                   end_chunk_idx)
                    curr_a2_scale = _chunk_scales(a2_scale, begin_chunk_idx,
                                                  end_chunk_idx)
                    curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]

                    self.fused_experts.apply(
                        fused_out[begin_out_idx:end_out_idx],
                        curr_a1q,
                        w1,
                        w2,
                        curr_topk_ids,
                        activation=activation,
                        global_num_experts=global_num_experts,
                        expert_map=expert_map,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        w1_zp=w1_zp,
                        w2_zp=w2_zp,
                        a1q_scale=curr_a1q_scale,
                        a2_scale=curr_a2_scale,
                        workspace13=workspace13,
                        workspace2=workspace2,
                        expert_num_tokens=expert_num_tokens,
                    )

        self.prepare_finalize.finalize(output, fused_out, topk_weights,
                                       topk_ids, apply_router_weight_on_input)

        return output
