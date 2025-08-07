# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/configuration_kimi_vl.py
from transformers.configuration_utils import PretrainedConfig


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(
            self,
            patch_size: int = 14,
            init_pos_emb_height: int = 64,
            init_pos_emb_width: int = 64,
            num_attention_heads: int = 16,
            num_hidden_layers: int = 27,
            hidden_size: int = 1152,
            intermediate_size: int = 4304,
            merge_kernel_size: tuple[int, int] = (2, 2),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Positional embedding config
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        # Transformer config
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Patch merger config
        self.merge_kernel_size = merge_kernel_size
