# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.positional_encoding import get_fused_rope

# Attempt to import vLLM configuration utilities
try:
    # Third Party
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
except ImportError:
    # Fallback for older versions or different paths
    set_current_vllm_config = None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non CUDA implementations for CUDA enhanced functions",
)
def test_rope():
    head_dim = 128
    max_position_embeddings = 8192
    rope_scaling = None
    rope_theta = 500000.0
    is_neox_style = True
    dtype = torch.bfloat16

    # ---  manually set vLLM config context ---
    if set_current_vllm_config is not None:
        c_config = CompilationConfig()
        c_config.custom_ops = ["all"]

        mock_config = MagicMock(spec=VllmConfig)
        mock_config.compilation_config = c_config

        with set_current_vllm_config(mock_config):
            fused_rotary_emb = get_fused_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                dtype=dtype,
            )
    else:
        # For older vLLM versions, no context is needed
        fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
    # ------------------------------------------

    assert fused_rotary_emb is not None, "Failed to get fused rotary embedding"
