# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import MiniMaxM3VLTextConfig


class MiniMaxM3MTPConfig(MiniMaxM3VLTextConfig):
    """Config for a standalone MiniMax-M3 MTP checkpoint.

    Its per-layer fields are copied from the main model, so only the leading
    entries, which the MTP layers read, are kept.
    """

    model_type = "minimax_m3_mtp"
    num_hidden_layers: int = 1

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        n = self.num_hidden_layers
        self.layer_types = self.layer_types[:n]  # type: ignore[has-type]
        self.mlp_layer_types = self.mlp_layer_types[:n]  # type: ignore[has-type]
