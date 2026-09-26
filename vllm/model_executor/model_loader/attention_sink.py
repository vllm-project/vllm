# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading for attention sinks whose runtime shape is padded.

DeepSeek pads the runtime sink to the platform's padded head count so that
padded heads can be disabled with ``-inf``. The checkpoint holds only real
heads, so the padded runtime representation has to be rebuilt at load time and
then handed to the parameter's current ``weight_loader``. A model that writes
the slice directly with ``copy_`` bypasses that loader, and a live (layerwise)
weight update then never reaches the storage the attention kernel reads.
"""

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def load_padded_attn_sink(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    head_start: int,
    head_end: int,
) -> None:
    """Load a TP-sliced checkpoint sink into a padded runtime sink parameter.

    The rank's head range is taken exactly once, here, and the result is
    expanded to the full padded shape with a ``-inf`` tail before being passed
    to the parameter's current loader. Rebuilding on every load keeps the
    padding correct when the loader is replayed onto a freshly materialized
    parameter during a layerwise reload, and keeps the element count equal to
    the parameter's, which the online loader requires.

    Args:
        param: Padded runtime sink parameter of shape ``(padded_heads,)``.
        loaded_weight: Global checkpoint sink tensor.
        head_start: First global head index owned by this rank.
        head_end: One past the last global head index owned by this rank.

    """
    local_weight = loaded_weight[head_start:head_end]
    if local_weight.numel() > param.numel():
        raise ValueError(
            f"Attention sink head range [{head_start}, {head_end}) selects "
            f"{local_weight.numel()} values, which does not fit the runtime "
            f"sink parameter of shape {tuple(param.shape)}"
        )

    runtime_weight = torch.empty(
        param.shape, dtype=param.dtype, device=local_weight.device
    )
    runtime_weight[: local_weight.numel()].copy_(local_weight.reshape(-1))
    if runtime_weight.numel() > local_weight.numel():
        runtime_weight[local_weight.numel() :].fill_(-float("inf"))

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, runtime_weight)
