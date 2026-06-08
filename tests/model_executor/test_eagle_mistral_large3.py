# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for EagleMistralLarge3Model.__init__.

EagleMistralLarge3Model bypasses DeepseekV2Model.__init__ and must
manually set every attribute that DeepseekV2Model.forward reads.
This file guards against future drift.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_config(model_type="mistral", qk_nope_head_dim=128, qk_rope_head_dim=64):
    cfg = MagicMock()
    cfg.model_type = model_type
    cfg.qk_nope_head_dim = qk_nope_head_dim
    cfg.qk_rope_head_dim = qk_rope_head_dim
    cfg.vocab_size = 32000
    cfg.hidden_size = 256
    cfg.num_hidden_layers = 2
    cfg.rms_norm_eps = 1e-5
    cfg.first_k_dense_replace = 0
    return cfg


def _make_vllm_config(hf_config):
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = hf_config
    vllm_config.quant_config = None
    vllm_config.parallel_config.pipeline_parallel_size = 1
    return vllm_config


@pytest.mark.parametrize(
    "model_type,qk_nope_head_dim,qk_rope_head_dim,expected_use_mha",
    [
        # Mistral Small 4 eagle config: MLA dims set → use_mha=False
        ("mistral", 128, 64, False),
        # Both dims zero → treated as MHA
        ("mistral", 0, 0, True),
        # deepseek model_type → always MHA regardless of dims
        ("deepseek", 128, 64, True),
    ],
)
def test_eagle_mistral_large3_use_mha_is_set(
    model_type, qk_nope_head_dim, qk_rope_head_dim, expected_use_mha
):
    """EagleMistralLarge3Model.__init__ must set self.use_mha.

    Regression for: AttributeError: 'EagleMistralLarge3Model' object
    has no attribute 'use_mha' when serving Mistral Small 4 with EAGLE.
    """
    from vllm.model_executor.models.mistral_large_3_eagle import (
        EagleMistralLarge3Model,
    )

    hf_config = _make_config(model_type, qk_nope_head_dim, qk_rope_head_dim)
    vllm_config = _make_vllm_config(hf_config)

    with (
        patch(
            "vllm.model_executor.models.mistral_large_3_eagle"
            ".VocabParallelEmbedding"
        ),
        patch(
            "vllm.model_executor.models.mistral_large_3_eagle"
            ".DeepseekV2DecoderLayer"
        ),
        patch(
            "vllm.model_executor.models.mistral_large_3_eagle.RowParallelLinear"
        ),
        patch("vllm.model_executor.models.mistral_large_3_eagle.RMSNorm"),
        patch(
            "vllm.model_executor.models.mistral_large_3_eagle"
            ".make_empty_intermediate_tensors_factory"
        ),
        patch(
            "vllm.model_executor.models.mistral_large_3_eagle.get_pp_group"
        ) as mock_pp,
    ):
        mock_pp.return_value.world_size = 1
        model = EagleMistralLarge3Model(vllm_config=vllm_config, prefix="model")

    assert hasattr(model, "use_mha"), (
        "EagleMistralLarge3Model.__init__ must set self.use_mha; "
        "DeepseekV2DecoderLayer.forward reads it and crashes if missing."
    )
    assert model.use_mha is expected_use_mha, (
        f"Expected use_mha={expected_use_mha} for model_type={model_type!r}, "
        f"qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}"
    )
