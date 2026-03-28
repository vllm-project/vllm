# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EPLB (Expert Parallelism Load Balancing) Manager.

This module provides the EplbManager class which encapsulates all EPLB-related
functionality for MoE layers, including state management, expert weight
collection, and expert parameter mapping.
"""

from collections.abc import Iterable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState


class EplbManager:
    """
    Manages Expert Parallelism Load Balancing (EPLB) state and operations
    for a MoE layer.

    This class encapsulates all EPLB-related functionality including:
    - Runtime state (expert load view, logical-to-physical mapping)
    - Expert weight collection for load balancing
    - Expert parameter mapping for weight loading with redundant experts
    """

    def __init__(
        self,
        num_redundant_experts: int = 0,
    ):
        self.num_redundant_experts = num_redundant_experts

        # Runtime EPLB state
        self.state = EplbLayerState()

    def set_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state for this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.

        Args:
            moe_layer_idx: Index of this MoE layer
            expert_load_view: View into global expert load tracking tensor
            logical_to_physical_map: Mapping from logical to physical expert IDs
            logical_replica_count: Number of replicas for each logical expert
        """
        self.state.expert_load_view = expert_load_view[moe_layer_idx]
        self.state.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.state.logical_replica_count = logical_replica_count[moe_layer_idx]

    def get_expert_weights(
        self,
        layer: torch.nn.Module,  # FusedMoE
    ) -> Iterable[torch.Tensor]:
        """
        Collect expert weights from the MoE layer for EPLB.

        Returns weights reshaped as (local_num_experts, -1) for efficient
        expert weight swapping during load balancing.

        Args:
            layer: The FusedMoE layer to collect weights from

        Returns:
            Iterable of expert weight tensors
        """

        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(layer.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        # `w13_input_scale` and `w2_input_scale` are global per-tensor
        # activation scales shared across all experts (e.g. NVFP4).
        # They are broadcast views (stride 0) from .expand() and are
        # not actual expert weights, so exclude them from EPLB.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
            "w13_input_scale",
            "w2_input_scale",
        }

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not (
                name.startswith("_runner._shared_experts._layer")
                or name.startswith("routed_experts.shared_experts._layer")
                or name.startswith("_runner.gate.")
                or name.startswith("_runner.routed_input_transform.")
                or name.startswith("_runner.routed_output_transform.")
            )
            and name not in NON_EXPERT_WEIGHTS
        )

        return [
            weight.data.view(layer.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_runner._shared_experts._layer")
            and not name.startswith("routed_experts.shared_experts._layer")
            # exclude parameters from non-expert submodules,
            # e.g. gate/shared/transforms.
            and not name.startswith("_runner.gate.")
            and not name.startswith("_runner.routed_input_transform.")
            and not name.startswith("_runner.routed_output_transform.")
        ]

    @staticmethod
    def validate_configuration(
        global_num_experts: int,
        ep_size: int,
    ) -> None:
        """
        Validate EPLB configuration.

        Args:
            global_num_experts: Total number of experts (including redundant)
            ep_size: Expert parallelism size

        Raises:
            AssertionError: If configuration is invalid
        """

        # EPLB currently only supports even distribution of experts across ranks
        assert global_num_experts % ep_size == 0, (
            f"EPLB currently only supports even distribution of "
            f"experts across ranks. Got {global_num_experts} experts "
            f"and {ep_size} EP ranks."
        )
