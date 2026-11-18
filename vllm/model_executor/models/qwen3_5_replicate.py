# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_QWEN3_5_REPLICATE_FFN: every TP rank holds each dense MLP whole.

Shared by the CUDA model (`vllm.models.qwen3_5.nvidia.model`) and the generic one
(`vllm.model_executor.models.qwen3_5`).
"""

from typing import Any


def setup_replicated_ffn(config: Any, parallel_config: Any) -> None:
    """Checks and cross-rank determinism for VLLM_QWEN3_5_REPLICATE_FFN."""
    if config.model_type != "qwen3_5_text":
        raise NotImplementedError(
            "VLLM_QWEN3_5_REPLICATE_FFN applies to dense Qwen3.5 models only."
        )
    # The layouts where tp_synced_autotune.configure() turns syncing off: ranks
    # would tune the replicated MLP each on their own.
    unsynced = [
        name
        for name, active in (
            ("expert parallelism", parallel_config.enable_expert_parallel),
            ("sequence-parallel MoE", parallel_config.use_sequence_parallel_moe),
            (
                "context parallelism",
                parallel_config.decode_context_parallel_size > 1
                or parallel_config.prefill_context_parallel_size > 1,
            ),
        )
        if active
    ]
    if unsynced:
        raise NotImplementedError(
            "VLLM_QWEN3_5_REPLICATE_FFN is not supported with "
            + ", ".join(unsynced)
            + "."
        )
    # Nothing re-syncs ranks after a replicated MLP: autotune must agree.
    from vllm.triton_utils.tp_synced_autotune import install

    install()
    # Same for kernels torch.compile generates: Inductor benchmarks configs that
    # change numerics per process unless told not to.
    import torch._inductor.config as inductor_config

    inductor_config.deterministic = True
