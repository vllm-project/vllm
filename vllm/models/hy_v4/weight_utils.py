# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.utils import WeightsMapper

# Parameter aliases belong to checkpoint loading, not quantization module names.
HYV4_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".gate.e_score_correction_bias": ".expert_bias",
        ".router.gate.": ".gate.",
    },
    orig_to_new_suffix={
        ".hc_fn": ".hc_fn.weight",
        ".hc_head_fn": ".hc_head_fn.weight",
    },
)

HYV4_MLP_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
        ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
        ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
    }
)
