# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup for DeepSeek V4.1 NVIDIA kernels."""

import torch


def autotune_replay_wkv(model: torch.nn.Module, num_tokens: int) -> None:
    """Tune replay-only projections that startup dummy forwards do not execute."""
    from vllm.utils.flashinfer import autotune, flashinfer_get_hybrid_num_tokens_buckets

    for module in model.modules():
        projection = getattr(module, "replay_wkv", None)
        if projection is None:
            continue
        hidden_states = torch.randn(
            num_tokens,
            projection.input_size,
            dtype=projection.params_dtype,
            device=projection.weight.device,
        )
        with autotune(
            tuning_buckets=flashinfer_get_hybrid_num_tokens_buckets(num_tokens),
            round_up=True,
        ):
            projection(hidden_states)
