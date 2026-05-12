# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch ZayaForCausalLM to add BitsAndBytes MoE support."""
import sys
import re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

patch = '''
    packed_modules_mapping: dict = {}

    def get_expert_mapping(self):
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

'''

src = src.replace(
    "class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):\n",
    "class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):\n" + patch,
)

with open(path, "w") as f:
    f.write(src)

print("Patch applied successfully.")
