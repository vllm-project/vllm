# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experts-kernel base classes for the fused MoE.

``FusedMoEExpertsModular`` is the [Permute-Experts-Unpermute] compute step:
its ``apply`` runs the two expert GEMMs, applies the router weights, and
reduces the top-k outputs. The surrounding quantize / dispatch / combine
pipeline lives in ``fused_moe_forward``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

logger = init_logger(__name__)


class FusedMoEActivationFormat(Enum):
    """Activation tensor layout, (num_tokens, hidden dim)."""

    Standard = ("standard",)


@dataclass
class ExpertTokensMetadata:
    """Metadata regarding expert-token routing."""

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None


class FusedMoEExperts(ABC):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        """
        moe_config: MoE layer configuration.
        quant_config: Quantization parameters for this experts instance.
        """
        self.moe_config = moe_config
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # noqa: B027
        pass

    @property
    def expects_unquantized_inputs(self) -> bool:
        """Whether ``fused_moe_forward`` should defer activation quantization
        to the experts kernel."""
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
        """Activation format produced by ``prepare`` and consumed by ``apply``."""
        raise NotImplementedError

    #
    # Various helpers for accessing quantization parameters from the
    # quant_config.
    #

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.weight_quant_dtype

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

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        A flag indicating whether or not this class can process packed ue8m0
        activation scales.
        """
        return False


class FusedMoEExpertsModular(FusedMoEExperts):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above.
    """

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
        assert len(w1.shape) == 3 and len(w2.shape) == 3
        E, N, _ = w1.shape
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
        activation: MoEActivation,
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
    def adjust_N_for_activation(N: int, activation: MoEActivation) -> int:
        """
        Calculate the output dimension for the activation function.

        For *_no_mul activations (e.g. relu2_no_mul),
        there's no gate/up split, so output size equals input size (N).

        For regular gated activations (e.g., silu, gelu),
        output size is N // 2 due to gate × activation(up) multiplication.

        Args:
            N: The intermediate size (width of w1/w3 weights).
            activation: The activation function enum.

        Returns:
            The output dimension after activation.
        """
        return N if not activation.is_gated else N // 2

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        clamp_limit: float | None = None,
    ) -> None:
        apply_moe_activation(activation, output, input, clamp_limit=clamp_limit)

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
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
        - output: (torch.Tensor): The weighted, reduced output tensor.
        - hidden_states: (torch.Tensor): The (quantized) input tensor to the MoE
          layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights: A map of row to expert weights, applied by this kernel.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1, produced by ``fused_moe_forward``'s input quantization.
        - workspace13 (torch.Tensor): A scratch tensor used for gemm outputs
          must be large enough to hold output of either MoE gemm.
        - workspace2 (torch.Tensor): A scratch tensor used for the activation
          function.
        - expert_tokens_meta (Optional[ExpertTokensMetadata]) - An optional
          ExpertTokensMetadata object containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - apply_router_weight_on_input: True if router weights are already
          applied on the input (topk == 1).
        """
        raise NotImplementedError
