# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import final

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    apply_moe_activation,
    count_expert_num_tokens,
    disable_inplace,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

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

    Standard = ("standard",)
    """
    The batched experts format (num experts, max tokens per expert, hidden dim)
    """
    BatchedExperts = ("batched_experts",)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(
            expert_num_tokens_list, device="cpu", dtype=torch.int32
        )
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device, non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu,
        )


class TopKWeightAndReduce(ABC):
    """
    An abstract base class for weight application and reduction implementations.
    """

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
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
    torch.Tensor | None,
    ExpertTokensMetadata | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

ReceiverType = Callable[[], PrepareResultType]


# TODO: pass FusedMoEParallelConfig in as ctor parameter?
class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.
    """

    def post_init_setup(self, fused_experts: "FusedMoEPermuteExpertsUnpermute"):
        """
        Initialize FusedMoEPrepareAndFinalize settings that depend on
        FusedMoEPermuteExpertsUnpermute experts object.
        The FusedMoEPrepareAndFinalize implementations that have such
        dependencies may choose to override this function.
        """
        return

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> PrepareResultType:
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
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEPermuteExpertsUnpermute
          in cases where the compute kernel expects unquantized inputs

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

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[Callable, ReceiverType] | ReceiverType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel
        but do not wait for results from other workers.
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
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEPermuteExpertsUnpermute
          in cases where the compute kernel expects unquantized inputs

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `prepare`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        e.g.

        ret = obj.prepare_async(...)

        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()

        if hook is not None:
        a, a_scales, expert_meta, topk_ids, topk_weights = receiver()

        is equivalent to:

        a, a_scales, expert_meta, topk_ids, topk_weights = obj.prepare(...)
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
        weight_and_reduce_impl: TopKWeightAndReduce,
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
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.
        """
        raise NotImplementedError

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output but do not wait for results from other workers.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `finalize`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        ret = obj.finalize_async(output, ...)
        ... output not valid yet ...
        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()
        receiver()
        ... output valid here ...

        is equivalent to:

        obj.finalize(output, ...)
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
    def topk_indices_dtype(self) -> torch.dtype | None:
        """
        The PrepareFinalize All2All implementations generally constrain the
        dtype of the topk_ids they support. This function returns the
        required topk indices dtype so it can be respected.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
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

    @abstractmethod
    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of finalize is reduced across all
        ranks.
        """
        raise NotImplementedError


# TODO: add supported activations method (return string)
class FusedMoEPermuteExpertsUnpermute(ABC):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """
        moe_config: MoE layer configuration.
        quant_config: Quantization parameters for this experts instance.
        """
        if self.activation_format() == FusedMoEActivationFormat.Standard and (
            max_num_tokens is not None or num_dispatchers is not None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers should only be set for "
                "BatchedExperts activation format."
            )
        elif self.activation_format() == FusedMoEActivationFormat.BatchedExperts and (
            max_num_tokens is None or num_dispatchers is None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers must be set for "
                "BatchedExperts activation format."
            )

        self.moe_config = moe_config
        self.quant_config = quant_config
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Whether or not the PrepareFinalize should defer input quantization
        in the prepare step. If True, then the Experts kernel will
        execute the input quantization itself.

        Sample subclasses that override are AITER and FlashInfer CUTLASS.
        """
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
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
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{a1.size(0)} == {E}"
            M = a1.size(1)  # This is max_num_tokens

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    #
    # Various helpers for registering support for various features.
    # Used by the oracle to select a particular kernel for a deployment.
    #

    @staticmethod
    def is_supported_config(
        cls: type["FusedMoEPermuteExpertsUnpermute"],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        def _make_reason(reason: str) -> str:
            return f"kernel does not support {reason}"

        if not cls._supports_current_device():
            return False, _make_reason("current device")
        elif not (moe_config.is_act_and_mul or cls._supports_no_act_and_mul()):
            return False, _make_reason("no act_and_mul MLP layer")
        elif not cls._supports_activation(moe_config.activation):
            return False, _make_reason(f"{moe_config.activation} activation")
        elif not cls._supports_quant_scheme(weight_key, activation_key):
            return False, _make_reason("quantization scheme")
        elif not cls._supports_parallel_config(moe_config.moe_parallel_config):
            return False, _make_reason("parallel config")
        elif activation_format != cls.activation_format():
            return False, _make_reason(f"{activation_format.value} activation format")
        return True, None

    @staticmethod
    @abstractmethod
    def _supports_current_device() -> bool:
        """
        Whether the kernel supports the current device type
        (compute cability and current platform).
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_no_act_and_mul() -> bool:
        """
        Whether the kernel supports act_and_mul=False, i.e.
        non-gated MoE models like Nemotron-Nano.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_activation(activation: str) -> bool:
        """
        Whether the kernel supports a particular act function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """
        Whether the kernel supports deployment in expert parallel.
        """
        raise NotImplementedError

    #
    # Various helpers for accessing quantization parameters from the
    # quant_config.
    #

    @property
    def quant_dtype(self) -> torch.dtype | None:
        return self.quant_config.quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def per_act_token_quant(self) -> bool:
        return self.quant_config.per_act_token_quant

    @property
    def per_out_ch_quant(self) -> bool:
        return self.quant_config.per_out_ch_quant

    @property
    def a1_scale(self) -> torch.Tensor | None:
        return self.quant_config.a1_scale

    @property
    def a2_scale(self) -> torch.Tensor | None:
        return self.quant_config.a2_scale

    @property
    def a1_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a1_gscale

    @property
    def a2_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a2_gscale

    @property
    def w1_scale(self) -> torch.Tensor | None:
        return self.quant_config.w1_scale

    @property
    def w2_scale(self) -> torch.Tensor | None:
        return self.quant_config.w2_scale

    @property
    def w1_zp(self) -> torch.Tensor | None:
        return self.quant_config.w1_zp

    @property
    def w2_zp(self) -> torch.Tensor | None:
        return self.quant_config.w2_zp

    @property
    def w1_bias(self) -> torch.Tensor | None:
        return self.quant_config.w1_bias

    @property
    def w2_bias(self) -> torch.Tensor | None:
        return self.quant_config.w2_bias

    @property
    def g1_alphas(self) -> torch.Tensor | None:
        return self.quant_config.g1_alphas

    @property
    def g2_alphas(self) -> torch.Tensor | None:
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

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        A flag indicating whether or not this class can process packed ue8m0
        activation scales.
        """
        return False

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        """
        Workspace type: The dtype to use for the workspace tensors.
        """
        return act_dtype

    @abstractmethod
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Inputs:
        - M: number of tokens.
        - N: Row (or column) dimension of expert weights.
        - K: hidden dimension
        - topk: The number of top-k experts to select.
        - global_num_experts: global number of experts.
        - local_num_experts: local number of experts due to DP/EP.
        - expert_tokens_meta: number of tokens per expert metadata for batched
                              format.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Note: workspace shapes can be 0 if the workspace is not needed.
          But in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens when the shape is
          not 0.
        """
        raise NotImplementedError

    @staticmethod
    def adjust_N_for_activation(N: int, activation: str) -> int:
        """
        Calculate the output dimension for the activation function.

        For *_no_mul activations (e.g. relu2_no_mul),
        there's no gate/up split, so output size equals input size (N).

        For regular gated activations (e.g., silu, gelu, swigluoai),
        output size is N // 2 due to gate × activation(up) multiplication.

        Args:
            N: The intermediate size (width of w1/w3 weights).
            activation: The activation function name.

        Returns:
            The output dimension after activation.
        """
        is_no_mul = activation.endswith("_no_mul")
        return N if is_no_mul else N // 2

    def activation(
        self, activation: str, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        apply_moe_activation(activation, output, input)

    def enable_chunking(self):
        return (
            envs.VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING and self.supports_chunking()
        )

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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
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


def _slice_scales(
    scales: torch.Tensor | None, start: int, end: int
) -> torch.Tensor | None:
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
        shared_experts: torch.nn.Module | None = None,
        moe_parallel_config: FusedMoEParallelConfig | None = None,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts = shared_experts
        # for EPLB
        self.local_to_global_physical_experts = None
        self.expert_map = None

        # prefer an explicit FusedMoEParallelConfig when available (from
        # FusedMoE layers / tests).
        # if not provided, assume this kernel is
        # running in a non-DP+EP context
        self.moe_parallel_config: FusedMoEParallelConfig | None = moe_parallel_config
        self.is_dp_ep = (
            moe_parallel_config is not None
            and moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
        )

        self._post_init_setup()
        assert (
            prepare_finalize.activation_format == fused_experts.activation_format()
        ), (
            f"{prepare_finalize.__class__.__name__}."
            f"{prepare_finalize.activation_format} == "
            f"{fused_experts.__class__.__name__}."
            f"{fused_experts.activation_format()}"
        )

    def _post_init_setup(self):
        """
        Resolve any leftover setup dependencies between self.prepare_finalize
        and self.fused_experts here.
        """
        self.prepare_finalize.post_init_setup(self.fused_experts)

    def supports_expert_map(self) -> bool:
        """
        A flag indicating whether or not this class supports expert maps.
        """
        return self.fused_experts.supports_expert_map()

    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of fused MoE kernel
        is reduced across all ranks.
        """
        return self.prepare_finalize.output_is_reduced()

    def _chunk_info(self, M: int) -> tuple[int, int]:
        """
        Compute number of chunks and chunk size for given M.
        If chunking is not supported, set the CHUNK_SIZE to M so we
        get num_chunks == 1. Take max(M, 1) to avoid divide by zero.
        If there are no tokens to process, the number of chunks will be zero.
        """
        CHUNK_SIZE = max(
            1,
            (
                M
                if not self.fused_experts.enable_chunking()
                else min(M, envs.VLLM_FUSED_MOE_CHUNK_SIZE)
            ),
        )
        num_chunks = cdiv(M, CHUNK_SIZE)
        # If there are no tokens, then there should be no loop iterations.
        assert M > 0 or num_chunks == 0
        return num_chunks, CHUNK_SIZE

    def _allocate_buffers(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        M_chunk: int,
        M_full: int,
        N: int,
        K: int,
        top_k: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Allocate temporary and output buffers for the fused experts op.
        Inputs:
        - out_dtype: output type of workspace and output tensors.
        - device: the device of the workspace and output tensors.
        See `workspace_shapes` for a description of the remainder of arguments.
        Returns a tuple of (workspace13, workspace2, output) tensors.
        """
        assert M_full > 0 and M_chunk > 0

        num_chunks, _ = self._chunk_info(M_full)
        workspace_dtype = self.fused_experts.workspace_dtype(out_dtype)

        # Force worst-case allocation in profiling run for
        # "mk.FusedMoEModularKernel.Standard" formats where this is only bounded
        # by `VLLM_FUSED_MOE_CHUNK_SIZE` and may not be seen during profiling with
        # DP+EP due to the random token routing.
        is_profile_run = (
            is_forward_context_available()
            and get_forward_context().attn_metadata is None
        )
        if is_profile_run and self.fused_experts.enable_chunking() and self.is_dp_ep:
            max_workspace_13, max_workspace_2, max_fused_out_shape = (
                self.fused_experts.workspace_shapes(
                    envs.VLLM_FUSED_MOE_CHUNK_SIZE,
                    N,
                    K,
                    top_k,
                    global_num_experts,
                    local_num_experts,
                    # expert_tokens_meta help in allocating optimal/minimal
                    # amount of workspace. Mark it None, so we allocate for
                    # the worst-case scenario.
                    expert_tokens_meta=None,
                    activation=activation,
                )
            )

            current_workspace_manager().get_simultaneous(
                (max_workspace_13, workspace_dtype),
                (max_workspace_2, workspace_dtype),
                (max_fused_out_shape, out_dtype),
            )

        # Get intermediate workspace shapes based off the chunked M size.
        workspace13_shape, workspace2_shape, _ = self.fused_experts.workspace_shapes(
            M_chunk,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # Get final output shape based on the full M size.
        _, _, fused_out_shape = self.fused_experts.workspace_shapes(
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # We can reuse the memory between cache1 and cache3 because by the
        # time we need cache3, we're done with cache1.
        # Construct the entire output that can then be processed in chunks.
        # Reuse workspace13 for the output in the non-chunked case.
        # This will not always be the case for standard
        # format experts and with experts that have empty workspaces.
        if num_chunks == 1:
            max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))
            common_workspace, workspace2 = current_workspace_manager().get_simultaneous(
                ((max_shape_size,), workspace_dtype),
                (workspace2_shape, workspace_dtype),
            )
            workspace13 = _resize_cache(common_workspace, workspace13_shape)
            fused_out = _resize_cache(common_workspace, fused_out_shape)
        else:
            workspace13, workspace2, fused_out = (
                current_workspace_manager().get_simultaneous(
                    (workspace13_shape, workspace_dtype),
                    (workspace2_shape, workspace_dtype),
                    (fused_out_shape, out_dtype),
                )
            )

        return workspace13, workspace2, fused_out

    @staticmethod
    def _slice_output_tensor(
        fused_out: torch.Tensor,
        chunk_idx: int,
        num_chunks: int,
        CHUNK_SIZE: int,
        M: int,
    ) -> torch.Tensor:
        if num_chunks == 1:
            return fused_out

        assert fused_out.size(0) % M == 0, f"fused_out shape {fused_out.shape} vs M {M}"
        factor = fused_out.size(0) // M
        out_chunk_size = CHUNK_SIZE * factor
        s = chunk_idx * out_chunk_size
        e = min(s + out_chunk_size, fused_out.size(0))
        return fused_out[s:e]

    @staticmethod
    def _slice_expert_tokens_metadata(
        num_chunks: int,
        full_expert_tokens_meta: ExpertTokensMetadata | None,
        chunk_topk_ids: torch.Tensor,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> ExpertTokensMetadata | None:
        if num_chunks == 1 or full_expert_tokens_meta is None:
            return full_expert_tokens_meta

        # The existing expert_num_tokens is for the entire a1q
        # input. Chunking forces recomputation of the number
        # of tokens assigned to each expert.
        c_expert_num_tokens = count_expert_num_tokens(
            chunk_topk_ids, local_num_experts, expert_map
        )

        c_expert_num_tokens_cpu = None
        need_expert_num_tokens_cpu = (
            full_expert_tokens_meta.expert_num_tokens_cpu is not None
        )
        if need_expert_num_tokens_cpu:
            # This is blocking as some implementations need the count
            # on the CPU to determine appropriate input/out fused-moe
            # buffers
            c_expert_num_tokens_cpu = c_expert_num_tokens.to("cpu", non_blocking=False)

        return ExpertTokensMetadata(
            expert_num_tokens=c_expert_num_tokens,
            expert_num_tokens_cpu=c_expert_num_tokens_cpu,
        )

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_load_view: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles DBO and async.
        """
        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = receiver()

        # In EPLB, update expert load from expert_num_tokens.
        if (
            expert_tokens_meta is not None
            and expert_load_view is not None
            and expert_tokens_meta.expert_num_tokens is not None
            and expert_map is not None
        ):
            # Initialize the mapping of the local physical experts
            # to global physical experts, after which it will not change.
            # expert_load_view: (num_physical_experts,)
            # expert_num_tokens: (local_num_physical_experts,)
            local_num_experts = expert_tokens_meta.expert_num_tokens.shape[0]
            if self.expert_map is None or not torch.equal(self.expert_map, expert_map):
                self.expert_map = expert_map.clone()

            start_idx = int(torch.distributed.get_rank()) * local_num_experts
            expert_load_view[start_idx : start_idx + local_num_experts] += (
                expert_tokens_meta.expert_num_tokens
            )

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (
            topk_weights if _expert_topk_weights is None else _expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def _fused_experts(
        self,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: ExpertTokensMetadata | None,
    ) -> torch.Tensor:
        _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids
        )

        num_chunks, CHUNK_SIZE = self._chunk_info(M_full)

        def input_chunk_range(chunk_idx: int) -> tuple[int, int]:
            if num_chunks == 1:
                # Use a1q.size(0) here since batched format does not
                # keep M in the first dimension.
                return 0, a1q.size(0)
            else:
                s = chunk_idx * CHUNK_SIZE
                e = min(s + CHUNK_SIZE, M_full)
                return s, e

        # This happens when none of the tokens from the all2all reach this
        # EP rank. Also, note that this is only relevant for CUDAGraph
        # incompatible all2all kernels like the DeepEP high-throughput
        # kernels. CUDAGraph compatible all2all kernels like the pplx
        # kernels and the DeepEP low-latency kernels are always batched
        # and can never run into the tensor.numel() == 0 case.
        if M_full == 0:
            assert num_chunks == 0
            workspace13 = None
            workspace2 = None
            fused_out = torch.empty_like(a1q, dtype=in_dtype)
        else:
            assert num_chunks > 0
            workspace13, workspace2, fused_out = self._allocate_buffers(
                in_dtype,
                a1q.device,
                CHUNK_SIZE,
                M_full,
                N,
                K,
                top_k,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )

        for chunk_idx in range(num_chunks):
            s, e = input_chunk_range(chunk_idx)

            c_expert_tokens_meta = self._slice_expert_tokens_metadata(
                num_chunks,
                expert_tokens_meta,
                topk_ids[s:e],
                local_num_experts,
                expert_map,
            )

            c_fused_out = self._slice_output_tensor(
                fused_out, chunk_idx, num_chunks, CHUNK_SIZE, M_full
            )

            self.fused_experts.apply(
                output=c_fused_out,
                hidden_states=a1q[s:e],
                w1=w1,
                w2=w2,
                topk_weights=topk_weights[s:e],
                topk_ids=topk_ids[s:e],
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=_slice_scales(a1q_scale, s, e),
                a2_scale=_slice_scales(self.fused_experts.a2_scale, s, e),
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=c_expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        return fused_out

    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        The _finalize method is a wrapper around self.prepare_finalize.finalize
        that handles DBO, async and shared expert overlap.
        """
        shared_output: torch.Tensor | None = None

        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                finalize_ret
                if isinstance(finalize_ret, tuple)
                else (None, finalize_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()

        if self.shared_experts is None:
            return output
        else:
            assert shared_output is not None
            return shared_output, output

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
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        expert_load_view: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        - expert_load_view (Optional[torch.Tensor]): Optional tensor for
          tracking expert load statistics. If provided, the kernel will
          update it using ExpertTokensMetadata.expert_num_tokens for
          better performance.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """

        if inplace and self.shared_experts is None and not disable_inplace():
            output = hidden_states
        else:
            output = torch.zeros_like(hidden_states)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            expert_load_view=expert_load_view,
        )

        fused_out = self._fused_experts(
            in_dtype=hidden_states.dtype,
            a1q=a1q,
            a1q_scale=a1q_scale,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_tokens_meta=expert_tokens_meta,
        )

        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )
