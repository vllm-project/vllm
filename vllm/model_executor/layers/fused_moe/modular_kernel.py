# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Callable, Generic, Optional, TypeVar, Union, final

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (  # yapf: disable
    _resize_cache, count_expert_num_tokens)
from vllm.utils import cdiv
from vllm.v1.worker.ubatching import (
    Schedule, dbo_current_ubatch_id, dbo_maybe_run_recv_hook,
    dbo_register_recv_hook, dbo_switch_to_comm, dbo_switch_to_compute,
    dbo_switch_to_compute_sync, dbo_yield,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm)

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


#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - Optional ExpertTokensMetadata containing gpu/cpu tensors
#   as big as the number of local experts with the information about the
#   number of tokens assigned to each local expert.
# - Optional dispatched expert topk IDs
# - Optional dispatched expert topk weight
#
# See `prepare` method below.
#
PrepareResultType = tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[ExpertTokensMetadata],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]

ReceiverType = Callable[[], PrepareResultType]
_R = TypeVar('_R')


#
# Prepare and Finalize Op Chains
#
# The prepare and finalize functions are broken down into a chain of sequential
# operations/steps.
#
class _PhasedGen(Generic[_R]):
    """
    Enforce an exact number of yields (phases), then a final return.

    Contract:
      - The generator must yield exactly `expected_yields` times.
      - The next advance must StopIteration with a return value (may be None).
      - Early StopIteration or extra yields raise RuntimeError.
      - Duplicate step/finish after completion raises RuntimeError.
    """
    __slots__ = ("_gen", "_expected", "_steps", "_done", "_ret")

    def __init__(self, gen: Generator[None, None, _R], expected_yields: int):
        self._gen = gen
        self._expected = expected_yields
        self._steps = 0
        self._done = False
        self._ret: Optional[_R] = None

    def step(self, label: str) -> None:
        if self._done:
            raise RuntimeError(
                f"Generator already finished; unexpected '{label}'.")
        if self._steps >= self._expected:
            raise RuntimeError(
                f"Too many steps: called '{label}' after {self._expected} "
                "phases; expected to finish instead.")
        try:
            next(self._gen)
        except StopIteration as exc:
            raise RuntimeError(
                f"Generator ended early during '{label}' "
                f"(completed {self._steps}/{self._expected} phases).") from exc
        self._steps += 1

    def finish(self, label: str) -> _R:
        if self._done:
            raise RuntimeError(
                f"Generator already finished; duplicate '{label}'.")
        if self._steps != self._expected:
            raise RuntimeError(
                f"Cannot finish at '{label}': only {self._steps}/"
                f"{self._expected} phases completed.")
        try:
            next(self._gen)
        except StopIteration as e:
            self._done = True
            self._ret = e.value  # may be None
            return self._ret  # type: ignore[return-value]
        else:
            raise RuntimeError(
                f"Generator yielded more than expected ({self._expected}); "
                f"should have finished at '{label}'.")


@dataclass
class AsyncOps(Generic[_R]):
    """
    3-phase async:
      1) prepare()
      2) send()
      3) recv()
      4) finish() -> R
    """
    prepare: Callable[[], None]
    send: Callable[[], None]
    recv: Callable[[], None]
    finish: Callable[[], _R]

    @classmethod
    def from_generator(cls, gen: Generator[None, None, _R]) -> 'AsyncOps[_R]':
        ph = _PhasedGen[_R](gen, expected_yields=3)
        return cls(
            prepare=lambda: ph.step("prepare"),
            send=lambda: ph.step("send"),
            recv=lambda: ph.step("recv"),
            finish=lambda: ph.finish("finish"),
        )


@dataclass
class SyncOps(Generic[_R]):
    """
    2-phase sync:
      1) prepare()
      2) send_recv()
      3) finish() -> R
    """
    prepare: Callable[[], None]
    send_recv: Callable[[], None]
    finish: Callable[[], _R]

    @classmethod
    def from_generator(cls, gen: Generator[None, None, _R]) -> 'SyncOps[_R]':
        ph = _PhasedGen[_R](gen, expected_yields=2)
        return cls(
            prepare=lambda: ph.step("prepare"),
            send_recv=lambda: ph.step("send_recv"),
            finish=lambda: ph.finish("finish"),
        )


AsyncPrepareOps = AsyncOps[PrepareResultType]
SyncPrepareOps = SyncOps[PrepareResultType]
AsyncFinalizeOps = AsyncOps[None]
SyncFinalizeOps = SyncOps[None]


# TODO: pass FusedMoEParallelConfig in as ctor parameter?
class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.
    """

    @abstractmethod
    def create_prepare_ops(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> Union[SyncPrepareOps, AsyncPrepareOps]:
        """
        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - quant_config: Quantization info provided by the fused experts.

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    def supports_async(self) -> bool:
        """
        Indicates whether or not this class implements prepare_async and
        finalize_async.
        """
        return False

    @abstractmethod
    def create_finalize_ops(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> Union[SyncFinalizeOps, AsyncFinalizeOps]:
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
        they can process only as set of tokens at a time. This
        function returns the batch size i.e the maximum number of tokens
        the implementation can process at a time.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError


# TODO: add supported activations method (return string)
class FusedMoEPermuteExpertsUnpermute(ABC):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
    above.
    """

    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
    ):
        """
        quant_config: Quantization parameters for this experts instance.
        """
        self.quant_config = quant_config

    @property
    @abstractmethod
    def activation_formats(
            self) -> tuple[FusedMoEActivationFormat, FusedMoEActivationFormat]:
        """
        A property which is a tuple of the input and output activation formats
        for the 'apply' method.
        """
        raise NotImplementedError

    def moe_problem_size(
        self,
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

        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.

        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        assert w1.dim() == 3 and w2.dim() == 3
        E, N, _ = w1.size()
        K = a1.size(-1)

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

    #
    # Various helpers for accessing quantization parameters from the
    # quant_config.
    #

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

    @property
    def a1_scale(self) -> Optional[torch.Tensor]:
        return self.quant_config.a1_scale

    @property
    def a2_scale(self) -> Optional[torch.Tensor]:
        return self.quant_config.a2_scale

    @property
    def a1_gscale(self) -> Optional[torch.Tensor]:
        return self.quant_config.a1_gscale

    @property
    def a2_gscale(self) -> Optional[torch.Tensor]:
        return self.quant_config.a2_gscale

    @property
    def w1_scale(self) -> Optional[torch.Tensor]:
        return self.quant_config.w1_scale

    @property
    def w2_scale(self) -> Optional[torch.Tensor]:
        return self.quant_config.w2_scale

    @property
    def w1_zp(self) -> Optional[torch.Tensor]:
        return self.quant_config.w1_zp

    @property
    def w2_zp(self) -> Optional[torch.Tensor]:
        return self.quant_config.w2_zp

    @property
    def w1_bias(self) -> Optional[torch.Tensor]:
        return self.quant_config.w1_bias

    @property
    def w2_bias(self) -> Optional[torch.Tensor]:
        return self.quant_config.w2_bias

    @property
    def g1_alphas(self) -> Optional[torch.Tensor]:
        return self.quant_config.g1_alphas

    @property
    def g2_alphas(self) -> Optional[torch.Tensor]:
        return self.quant_config.g2_alphas

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
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
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
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1.  Result of quantization from prepare/finalize and not
          from the FusedMoEQuantConfig.
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


class SharedResizableBuffer:

    def __init__(self):
        self.buffer = None

    def get(self, shape: tuple[int, ...], device: torch.device,
            dtype: torch.dtype):
        if shape == () or shape is None:
            return None
        shape_numel = prod(shape)
        if (self.buffer is None or self.buffer.numel() < shape_numel
                or self.buffer.device != device or self.buffer.dtype != dtype):
            self.buffer = torch.empty(shape_numel, device=device, dtype=dtype)
        return self.buffer[:shape_numel].view(*shape)


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
    fused_out_buffer = SharedResizableBuffer()
    workspace13_buffer = SharedResizableBuffer()
    workspace2_buffer = SharedResizableBuffer()

    class SharedBuffers:

        def __init__(self) -> None:
            self.fused_out = SharedResizableBuffer()
            self.workspace13 = SharedResizableBuffer()
            self.workspace2 = SharedResizableBuffer()

    # Persistent buffers that are shared across `FusedMoEModularKernel`
    # instances (layers), to save memory and allocattions.
    #
    # We have two sets of buffers to support dual batch overlap (DBO) where each
    # microbatch (ubatch) should use its own set of buffers to avoid
    # cross-ubatch contimination.
    # NOTE that memory is lazily allocated for these buffers, meaning that if
    # DBO isn't being used, the second SharedBuffers will be empty.
    shared_buffers: list[SharedBuffers] = [SharedBuffers(), SharedBuffers()]

    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
        shared_experts: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts = shared_experts
        assert prepare_finalize.activation_format == \
            fused_experts.activation_formats[0], (
                f"{prepare_finalize.__class__.__name__}."
                f"{prepare_finalize.activation_format} == "
                f"{fused_experts.__class__.__name__}."
                f"{fused_experts.activation_formats[0]}")

    def _do_fused_experts(
        self,
        fused_out: Optional[torch.Tensor],
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
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:

        _, M, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids)

        (workspace13_shape, workspace2_shape, fused_out_shape,
         workspace_dtype) = self.fused_experts.workspace_shapes(
             a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
             expert_tokens_meta)

        # select per-ubatch buffers to avoid cross-ubatch reuse under DBO
        ubatch_idx = dbo_current_ubatch_id()
        buffers = self.shared_buffers[ubatch_idx]

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        workspace13 = buffers.workspace13.get(workspace13_shape,
                                              device=a1.device,
                                              dtype=workspace_dtype)
        workspace2 = buffers.workspace2.get(workspace2_shape,
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
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

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
        a1q_scale: Optional[torch.Tensor],
        expert_tokens_meta: Optional[ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:

        _, M, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids)

        CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
        num_chunks = cdiv(M, CHUNK_SIZE)

        # TODO(bnell): get rid of one level here, update slice functions
        # to nops on num_chunks==1

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
                a1q_scale=a1q_scale,
                a2_scale=self.fused_experts.a2_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Chunking required case
        assert num_chunks > 1

        # Construct the entire output that can then be processed in chunks.
        (_, _, fused_out_shape, _) = self.fused_experts.workspace_shapes(
            a1, a1q, M, N, K, top_k, global_num_experts, local_num_experts,
            expert_tokens_meta)

        ubatch_idx = dbo_current_ubatch_id()
        buffers = self.shared_buffers[ubatch_idx]
        fused_out = buffers.fused_out.get(fused_out_shape,
                                          device=a1q.device,
                                          dtype=a1.dtype)

        def slice_input_tensors(
            chunk_idx: int
        ) -> tuple[torch.Tensor, Optional[torch.Tensor],
                   Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
            s = chunk_idx * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, M)
            return (
                a1q[s:e],
                _chunk_scales(a1q_scale, s, e),
                _chunk_scales(self.fused_experts.a2_scale, s, e),
                topk_ids[s:e],
                topk_weights[s:e],
            )

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

        for chunk_idx in range(num_chunks):
            c_a1q, c_a1q_scale, c_a2_scale, c_topk_ids, c_topk_weights = (
                slice_input_tensors(chunk_idx))

            c_expert_tokens_meta = None
            if expert_tokens_meta is not None:
                c_expert_tokens_meta = slice_expert_tokens_metadata(
                    expert_tokens_meta, c_topk_ids, local_num_experts,
                    expert_map)

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
                a1q_scale=c_a1q_scale,
                a2_scale=c_a2_scale,
                expert_tokens_meta=c_expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

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
        apply_router_weight_on_input: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        - apply_router_weight_on_input (bool): When true, the topk weights are
          applied directly on the inputs. This is only applicable when topk is
          1.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """

        a1 = hidden_states
        if inplace and self.shared_experts is None:
            output = a1
        else:
            output = torch.zeros_like(a1)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        shared_output: Optional[torch.Tensor] = None

        prepare_ops = self.prepare_finalize.create_prepare_ops(
            a1,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            self.fused_experts.quant_config,
        )

        prepare_ops.prepare()

        if isinstance(prepare_ops, SyncOps):
            # We yield before launching the dispatch kernel since the dispatch
            # kernel will block the CPU so we want to queue up all the compute
            # for the other ubatch before the dispatch kernel starts.
            dbo_yield_and_switch_from_compute_to_comm()
            prepare_ops.send_recv()
            dbo_switch_to_compute_sync()
        else:
            assert isinstance(prepare_ops, AsyncOps)

            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ops.send()

            if dbo_register_recv_hook(
                    lambda: prepare_ops.recv(),
                    schedules=(Schedule.MLP_SHARED_OVERLAP, )):
                dbo_yield(all_schedules=True)
            else:
                prepare_ops.recv()

        (a1q, a1q_scale, expert_tokens_meta, _expert_topk_ids,
         _expert_topk_weights) = prepare_ops.finish()

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
                a1q_scale=a1q_scale,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        finalize_ops = self.prepare_finalize.create_finalize_ops(
            output,
            fused_out,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
        )

        finalize_ops.prepare()

        if isinstance(finalize_ops, SyncOps):
            dbo_yield_and_switch_from_compute_to_comm()
            finalize_ops.send_recv()

            dbo_switch_to_compute()
            if self.shared_experts is not None and shared_output is None:
                shared_output = self.shared_experts(a1)
            dbo_switch_to_comm()

            dbo_yield_and_switch_from_comm_to_compute()
            finalize_ops.finish()
        else:
            assert isinstance(finalize_ops, AsyncOps)
            dbo_maybe_run_recv_hook()
            finalize_ops.send()

            if self.shared_experts is not None and shared_output is None:
                shared_output = self.shared_experts(a1)

            if dbo_register_recv_hook(lambda: finalize_ops.recv(),
                                      all_schedules=True):
                dbo_yield(all_schedules=True)
            else:
                finalize_ops.recv()
            finalize_ops.finish()

        if self.shared_experts is None:
            return output
        else:
            assert shared_output is not None
            return shared_output, output
