# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading for DeepSeek attention sinks."""

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def load_padded_attn_sink(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    head_start: int,
    head_end: int,
) -> None:
    """Load a TP-sliced checkpoint sink into a padded runtime sink parameter.

    DeepSeek pads the runtime sink to the platform's padded head count so padded
    heads can be disabled with ``-inf``. The checkpoint holds only real heads, so
    the padded runtime representation is rebuilt here and then handed to the
    parameter's current loader. Rebuilding on every load keeps the padding
    correct when the loader is replayed onto a freshly materialized parameter
    during a layerwise reload.

    Args:
        param: Padded runtime sink parameter of shape ``(padded_heads,)``.
        loaded_weight: Global checkpoint sink tensor.
        head_start: First global head index owned by this rank.
        head_end: One past the last global head index owned by this rank.

    """
    local_weight = loaded_weight[head_start:head_end]

    runtime_weight = torch.full(
        param.shape,
        -float("inf"),
        dtype=param.dtype,
        device=local_weight.device,
    )
    if local_weight.numel():
        runtime_weight[: local_weight.shape[0]].copy_(local_weight)

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, runtime_weight)
