# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import torch


class MkFusedExpertsSupportsLoRA:
    """
    Inherit MkFusedExpertsSupportsLoRA in FusedMoEPermuteExpertsUnpermute
    implementations that support LoRA.
    """

    """
    All the prologue and epilogue functions are overriden by
    FusedMoEPermuteExpertsUnpermuteWithLoRA class when run with LoRA.
    """

    def activation_prologue(self, gateup_proj_output: torch.Tensor):
        pass

    def activation_epilogue(self, activation_output: torch.Tensor):
        pass

    @contextmanager
    def maybe_activation_with_lora_hook(
        self, gateup_proj_output: torch.Tensor, activation_output: torch.Tensor
    ):
        self.activation_prologue(gateup_proj_output=gateup_proj_output)
        yield
        self.activation_epilogue(activation_output=activation_output)

    def set_lora_token_mapping_offset(self, lora_token_mapping_offset: int):
        self.lora_token_mapping_offset = lora_token_mapping_offset

    def get_lora_token_mapping_offset(self) -> int:
        assert hasattr(self, "lora_token_mapping_offset"), (
            "FusedMoEModularKernel must set_lora_token_mapping_offset before it"
            "can be accessed"
        )
        return self.lora_token_mapping_offset


def mk_fused_experts_supports_lora(mk_fused_experts: object) -> bool:
    return isinstance(mk_fused_experts, MkFusedExpertsSupportsLoRA)
