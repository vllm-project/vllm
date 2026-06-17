# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for MLA DCP query replication."""

import torch

from vllm import envs


def dcp_q_replicate_enabled() -> bool:
    return bool(envs.VLLM_DCP_Q_REPLICATE)


def dcp_q_group_index(tp_rank: int, dcp_world_size: int) -> int:
    assert dcp_world_size > 0
    return tp_rank // dcp_world_size


def dcp_q_replicated_heads(local_heads: int, dcp_world_size: int) -> int:
    assert local_heads > 0
    assert dcp_world_size > 0
    return local_heads * dcp_world_size


def load_dcp_replicated_column_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    group_idx: int,
) -> None:
    """Load this DCP group's contiguous output-row slice into a replicated param."""
    output_dim = param.output_dim
    shard_size = param.data.shape[output_dim]
    loaded_shard = loaded_weight.narrow(output_dim, group_idx * shard_size, shard_size)
    param.weight_loader(param, loaded_shard)


def dcp_qrep_replica_map(
    params_dict: dict[str, torch.nn.Parameter],
) -> dict[str, str]:
    """Map each qrep source-weight name -> its ``dcp_`` replica param name."""
    prefix = ".self_attn.dcp_"
    return {
        name.replace(prefix, ".self_attn.", 1): name
        for name in params_dict
        if prefix in name
    }
