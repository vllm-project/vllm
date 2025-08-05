# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Optional, final

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (  # yapf: disable
    _resize_cache, count_expert_num_tokens)
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
# Each component will be independent of (but may inform) the others except for
# [Quantize-Dispatch] and `[Combine] (see below). The components can then be
# mixed and matched with so that DP+EP can be supported easily for multiple
# MoE kernel implementations.
#
# The following main classes are defined:
# * FusedMoEPrepareAndFinalize - an abstract base class for preparation of MoE
#   inputs (e.g. quantization, distribution) and finalization of Moe outputs.
#   The prepare method must take care of any needed quantization and the
#   finalize method, informed by the FusedMoEPermuteExpertsUnpermute method,
#   may apply weights and/or do the final reduction of the output.
# * FusedMoEPermuteExpertsUnpermute - an abstract base class for the main fused
#   MoE operation, i.e matmul + act_mul + optionally quant + matmul.
#   Some FusedMoEPermuteExpertsUnpermute implementations may choose to do
#   the weight application and/or reduction. The class communicates this
#   to [Finalize] via a TopKWeightAndReduce object.
# * FusedMoEModularKernel - an interface class that combines a
#   FusedMoEPrepareAndFinalize and a FusedMoEPermuteExpertsUnpermute to
#   provide the standard fused MoE kernel interface.
# * TopKWeightAndReduce - A TopKWeightAndReduce implementation chosen
#   by the FusedMoEPermuteExpertsUnpermute implementation that is passed
#   on to [Finalize].
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


class FusedMoEActivationFormat(Enum):
    """
    The standard activation format (num_tokens, hidden dim).
    """
    Standard = "standard",
    """
    The batched experts format (num experts, max tokens per expert, hidden dim)
    """
    BatchedExperts = "batched_experts",


@dataclass
class ExpertTokensMetadata:
    """
  Metadata regarding expert-token routing.
  """
    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: Optional[torch.Tensor]

    @staticmethod
    def make_from_list(expert_num_tokens_list: list[int],
                       device: str) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(expert_num_tokens_list,
                                             device="cpu",
                                             dtype=torch.int32)
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device,
                                                       non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu)


class TopKWeightAndReduce(ABC):
    """
    An abstract base class for weight application and reduction implementations.
    """

    @abstractmethod
    def apply(self, output: Optional[torch.Tensor],
              fused_expert_output: torch.Tensor, topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              apply_router_weight_on_input: bool) -> torch.Tensor:
        """
        Apply topk_weights to the fused_experts_outputs and/or reduce.
        If an output tensor is not passed, it will be created in the
        function.
        """
        raise NotImplementedError


# TODO: pass FusedMoEParallelConfig in as ctor parameter?
class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.
    """

    @abstractmethod
    def prepare(
        self, a1: torch.Tensor, a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor], topk_weights: torch.Tensor,
        topk_ids: torch.Tensor, num_experts: int,
        expert_map: Optional[torch.Tensor], apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        extra_prepare_args: Optional[dict[str, Any]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[ExpertTokensMetadata], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
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
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: TopKWeightAndReduce,
                 extra_finalize_args: Optional[dict[str, Any]]) -> None:
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
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def activation_format(self) -> FusedMoEActivationFormat:
        """
        A property indicating the output format of the activations for the
        'prepare' method.
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

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError


class FusedMoEPermuteExpertsUnpermute(ABC):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
    above.
    """

    def __init__(
        self,
        quant_config: Optional[FusedMoEQuantConfig],
    ):
        if quant_config is not None:
            self.quant_config = quant_config
        else:
            self.quant_config = FusedMoEQuantConfig()

    @property
    @abstractmethod
    def activation_formats(
            self) -> tuple[FusedMoEActivationFormat, FusedMoEActivationFormat]:
        """
        A property which is a tuple of the input and output activation formats
        for the 'apply' method.
        """
        raise NotImplementedError

    @property
    def quant_dtype(self) -> Optional[torch.dtype]:
        return self.quant_config.quant_dtype

    @property
    def block_shape(self) -> Optional[list[int]]:
        return self.quant_config.block_shape

    @property
    def per_act_token_quant(self) -> bool:
        return self.quant_config.per_act_token_quant

    @property
    def per_out_ch_quant(self) -> bool:
        return self.quant_config.per_out_ch_quant

    # TODO (bnell): make this return a CHUNK_SIZE or None instead?
    @abstractmethod
    def supports_chunking(self) -> bool:
        """
        A flag indicating whether or not this class supports activation
        chunking.
        """
        raise NotImplementedError

    @abstractmethod
    def supports_expert_map(self) -> bool:
        """
        A flag indicating whether or not this class supports expert maps
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
        expert_tokens_meta: Optional[ExpertTokensMetadata],
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

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
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
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
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
        - topk_weights: A map of row to expert weights. Some implementations
          choose to do weight application. 
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
        - expert_tokens_meta (Optional[ExpertTokensMetadata]) - An optional
          ExpertTokensMetadata object containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - apply_router_weight_on_input: True if router weights are already
          applied on the input. This is relevant if the implementation
          chooses to do weight application.
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


@final
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
        assert prepare_finalize.activation_format == \
            fused_experts.activation_formats[0], (
                f"{prepare_finalize.__class__.__name__}."
                f"{prepare_finalize.activation_format} == "
                f"{fused_experts.__class__.__name__}."
                f"{fused_experts.activation_formats[0]}")

    def _do_fused_experts(
            self, fused_out: Optional[torch.Tensor], a1: torch.Tensor,
            a1q: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
            activation: str, global_num_experts: int, local_num_experts: int,
            expert_map: Optional[torch.Tensor],
            w1_scale: Optional[torch.Tensor], w2_scale: Optional[torch.Tensor],
            w1_zp: Optional[torch.Tensor], w2_zp: Optional[torch.Tensor],
            a1q_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            expert_tokens_meta: Optional[ExpertTokensMetadata],
            apply_router_weight_on_input: bool,
            extra_expert_args: Optional[dict[str, Any]]) -> torch.Tensor:

        _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

        (workspace13_shape, workspace2_shape, fused_out_shape,
         workspace_dtype) = self.fused_experts.workspace_shapes(
             a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
             expert_tokens_meta)

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        workspace13 = torch.empty(prod(workspace13_shape),
                                  device=a1.device,
                                  dtype=workspace_dtype)
        workspace2 = torch.empty(prod(workspace2_shape),
                                 device=a1.device,
                                 dtype=workspace_dtype)

        assert fused_out is None or fused_out.shape == fused_out_shape, (
            f"fused_out {fused_out.shape} but expected {fused_out_shape}")
        if fused_out is None:
            # reuse workspace13 for the output
            fused_out = _resize_cache(workspace13, fused_out_shape)

        self.fused_experts.apply(
            fused_out,
            a1q,
            w1,
            w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
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
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args)

        return fused_out

    def _maybe_chunk_fused_experts(
        self,
        a1: torch.Tensor,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:

        _, M, N, K, top_k = _moe_problem_size(a1q, w1, w2, topk_ids)

        CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
        num_chunks = cdiv(M, CHUNK_SIZE)

        if not self.fused_experts.supports_chunking() or num_chunks == 1:
            return self._do_fused_experts(
                fused_out=None,
                a1=a1,
                a1q=a1q,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args)

        # Chunking required case
        assert num_chunks > 1

        # Construct the entire output that can then be processed in chunks.
        (_, _, fused_out_shape, _) = self.fused_experts.workspace_shapes(
            a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
            expert_tokens_meta)
        fused_out = torch.empty(fused_out_shape,
                                device=a1q.device,
                                dtype=a1.dtype)

        def slice_input_tensors(
            chunk_idx: int
        ) -> tuple[torch.Tensor, Optional[torch.Tensor],
                   Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
            s = chunk_idx * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, M)
            return (a1q[s:e], _chunk_scales(a1q_scale, s, e),
                    _chunk_scales(a2_scale, s,
                                  e), topk_ids[s:e], topk_weights[s:e])

        def slice_output_tensor(chunk_idx: int) -> torch.Tensor:
            assert fused_out.size(0) % M == 0, (
                f"fused_out shape {fused_out.shape} vs M {M}")
            factor = fused_out.size(0) // M
            out_chunk_size = CHUNK_SIZE * factor
            s = chunk_idx * out_chunk_size
            e = min(s + out_chunk_size, fused_out.size(0))
            return fused_out[s:e]

        def slice_expert_tokens_metadata(
                full_expert_tokens_meta: ExpertTokensMetadata,
                chunk_topk_ids: torch.Tensor, local_num_experts: int,
                expert_map: Optional[torch.Tensor]) -> ExpertTokensMetadata:
            # The existing expert_num_tokens is for the entire a1q
            # input. Chunking forces recomputation of the number
            # of tokens assigned to each expert.
            c_expert_num_tokens = count_expert_num_tokens(
                chunk_topk_ids, local_num_experts, expert_map)

            c_expert_num_tokens_cpu = None
            need_expert_num_tokens_cpu = (
                full_expert_tokens_meta.expert_num_tokens_cpu is not None)
            if need_expert_num_tokens_cpu:
                # This is blocking as some implementations need the count
                # on the CPU to determine appropriate input/out fused-moe
                # buffers
                c_expert_num_tokens_cpu = c_expert_num_tokens.to(
                    "cpu", non_blocking=False)

            return ExpertTokensMetadata(
                expert_num_tokens=c_expert_num_tokens,
                expert_num_tokens_cpu=c_expert_num_tokens_cpu)

        m = None
        if extra_expert_args is not None and 'm' in extra_expert_args:
            m = extra_expert_args.get('m')

        if extra_expert_args is not None:
            chunked_extra_expert_args = extra_expert_args
        else:
            chunked_extra_expert_args = {}

        for chunk_idx in range(num_chunks):
            c_a1q, c_a1q_scale, c_a2_scale, c_topk_ids, c_topk_weights = (
                slice_input_tensors(chunk_idx))

            c_expert_tokens_meta = None
            if expert_tokens_meta is not None:
                c_expert_tokens_meta = slice_expert_tokens_metadata(
                    expert_tokens_meta, c_topk_ids, local_num_experts,
                    expert_map)

            s = chunk_idx * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, M)

            if m is not None:
                chunked_extra_expert_args['m'] = e - s
            self._do_fused_experts(
                fused_out=slice_output_tensor(chunk_idx),
                a1=a1,
                a1q=c_a1q,
                w1=w1,
                w2=w2,
                topk_weights=c_topk_weights,
                topk_ids=c_topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1q_scale=c_a1q_scale,
                a2_scale=c_a2_scale,
                expert_tokens_meta=c_expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=chunked_extra_expert_args)

        return fused_out

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
        extra_expert_args: Optional[dict] = None,
        extra_prepare_args: Optional[dict] = None,
        extra_finalize_args: Optional[dict] = None,
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
        - extra_expert_args (Optional[dict]): Extra keyword arguments to pass to
          fused_experts.apply.
        - extra_prepare_args (Optional[dict]): Extra keyword arguments to pass
          to prepare.
        - extra_finalize_args (Optional[dict]): Extra keyword arguments to pass 
          to finalize.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """

        a1 = hidden_states
        output = a1 if inplace else torch.zeros_like(a1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
         _expert_topk_weights) = self.prepare_finalize.prepare(
             a1,
             a1_scale,
             a2_scale,
             topk_weights,
             topk_ids,
             global_num_experts,
             expert_map,
             apply_router_weight_on_input,
             self.fused_experts.quant_config,
             extra_prepare_args,
         )

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
            fused_out = self._maybe_chunk_fused_experts(
                a1=a1,
                a1q=a1q,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args)

        self.prepare_finalize.finalize(
            output, fused_out, topk_weights, topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
            extra_finalize_args)

        return output
