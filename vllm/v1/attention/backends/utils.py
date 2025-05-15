# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""
    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""


def validate_kv_target_layer(layer_name, kv_target_layer_name, layers):
    if kv_target_layer_name not in layers:
        raise ValueError(
            f"KV sharing target layer for {layer_name} not valid. "
            f"{kv_target_layer_name} is not a Attention layer in the model.")

    if (layers[kv_target_layer_name].attn_type
            != layers[layer_name].attn_type):
        raise ValueError(
            f"Expected KV sharing target layer for {layer_name} to "
            f"have attn_type {layers[layer_name].attn_type}, but got "
            f"{layers[kv_target_layer_name].attn_type} instead.")
