# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NVFP4 routed experts on Hopper as one DeepGEMM MegaMoE launch."""

import functools

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_world_group
from vllm.distributed.parallel_state import GroupCoordinator, init_model_parallel_group
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_NVFP4_SM90,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.utils.deep_gemm import NVFP4Weights
from vllm.utils.math_utils import cdiv


@functools.cache
def mega_rank_group() -> GroupCoordinator:
    """This rank alone, from one collective call on every rank."""
    world = get_world_group()
    return init_model_parallel_group(
        [[rank] for rank in range(world.world_size)],
        world.local_rank,
        torch.distributed.get_backend(world.device_group),
        group_name="inkling2_mega_moe",
        use_device_communicator=False,
    )


class Inkling2MegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts whose parameters are the `w13` and `w2` NVFP4 weights
    already in the SM90 kernel layout
    (`transform_nvfp4_weights_for_sm90_mega_moe`), this rank's contiguous
    slice of the experts of the launch group (`_group`, the EP group).

    Args:
        l1: `w13` codes, scales and global scales; rows interleave gate/up.
        l2: `w2` codes, scales and global scales.
        top_k: Experts per token.
        gate_cap: Softcap of the SiTU gate.
        up_cap: Softcap of the SiTU up projection.

    """

    # `Inkling2MoE` captures the routed experts of each layer visit itself.
    capture_fn = property(lambda self: None, lambda self, fn: None)

    def __init__(
        self,
        vllm_config: VllmConfig,
        l1: NVFP4Weights,
        l2: NVFP4Weights,
        *,
        top_k: int,
        gate_cap: float,
        up_cap: float,
        prefix: str,
    ):
        self._kernel_layout_weights = {"w13": l1, "w2": l2}
        num_local_experts, gate_up_size, packed_hidden_size = l1[0].shape
        group = self._group()
        super().__init__(
            vllm_config,
            num_experts=num_local_experts * group.world_size,
            num_local_experts=num_local_experts,
            experts_start_idx=num_local_experts * group.rank_in_group,
            top_k=top_k,
            hidden_size=2 * packed_hidden_size,
            intermediate_size=gate_up_size // 2,
            activation=MegaMoEActivation("situ", alpha=gate_cap, beta=up_cap),
            weight_format=MEGA_MOE_NVFP4_SM90,
            prefix=prefix,
        )

    def _create_weights(self) -> None:
        for name, tensors in self._kernel_layout_weights.items():
            suffixes = ("weight", "weight_scale", "weight_scale_2")
            for suffix, tensor in zip(suffixes, tensors):
                param = nn.Parameter(tensor, requires_grad=False)
                self.register_parameter(f"{name}_{suffix}", param)
        del self._kernel_layout_weights


class Inkling2LocalMegaMoEExperts(Inkling2MegaMoEExperts):
    """Each rank launches alone over its own experts (`Inkling2Experts`
    shards them over the TP group) and returns its partial sum."""

    _group = staticmethod(mega_rank_group)

    def _mean_num_tokens(self, num_tokens: int) -> int:
        # About 1 / tp of the tokens' slots land on this rank's experts.
        return cdiv(num_tokens, get_tensor_model_parallel_world_size())
