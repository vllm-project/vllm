# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""iHC sequence-parallel gate shared by the HY4 NVIDIA and ROCm models."""

import vllm.envs as envs
from vllm.config import VllmConfig

# Smallest step that shards the iHC rows; shorter ones keep them replicated.
IHC_SP_MIN_TOKENS = 1024


def ihc_sequence_parallel_enabled(vllm_config: VllmConfig, supported: bool) -> bool:
    """Whether the iHC rows may be sharded across TP ranks."""
    parallel_config = vllm_config.parallel_config
    speculative_config = vllm_config.speculative_config
    return (
        getattr(vllm_config.model_config.hf_text_config, "enable_ihc", False)
        and supported
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.data_parallel_size == 1
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.all2all_backend != "passthrough"
        and vllm_config.lora_config is None
        and not envs.VLLM_BATCH_INVARIANT
        and (
            speculative_config is None or not speculative_config.uses_aux_hidden_states
        )
    )
