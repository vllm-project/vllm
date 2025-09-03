# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass
class EplbData:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP, pplx-kernels).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    expert_load_window_step: int = 0
    """
    Current step in the sliding window.

    Different from `expert_rearrangement_step`, each EP rank may have its own
    `expert_load_window_step`.
    """
    expert_load_window_size: int = 0
    """
    Size of the expert load sliding window.
    This is a constant and is taken from the config.
    """

    expert_rearrangement_step: int = 0
    """
    Steps after last rearrangement.
    Will trigger a rearrangement if it exceeds the threshold.

    NOTE: Keep in mind that all EP ranks need to have the same
    `expert_rearrangement_step` value to ensure synchronization.
    Otherwise, the rearrangement will hang at collective
    communication calls.
    """
    expert_rearrangement_step_interval: int = 0
    """
    Interval for expert rearrangement steps.
    This is a constant and is taken from the config.
    """

    num_dense_layers: int = 0
    """
    The total number of dense layers in the model.
    """

    global_expert_num: int = 0
    """
    The total number of global experts across all MoE layers.
    """

    num_moe_layers: int = 0
    """
    The total number of Mixture-of-Experts (MoE) layers in the model.
    """

    num_expert_layers: int = 0
    """
    The total number of expert layers (could be different from num_moe_layers
    if experts are grouped or handled differently).
    """

    num_wait_worker_iterations: int = 0
    """
    The number of iterations to wait for workers to synchronize or catch up.
    """

    num_iterations_eplb_update: int = 0
    """
    The number of iterations after which EPLB (Expert Parallel Load Balancing)
    should perform an update.
    """

    gate_eplb: bool = False
    """
    A boolean flag indicating whether EPLB is enabled or active.
    If True, EPLB logic will be applied.
    """

