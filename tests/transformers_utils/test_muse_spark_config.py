# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MuseSpark1x text config normalization and the DSpark draft config."""

import json
import subprocess
import sys
from typing import Any

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.models.muse_spark.common.dspark import MuseSparkDSparkForCausalLM
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.muse_spark import (
    GatedDeltaNetArgs,
    MetapArgs,
    MoEArgs,
    MuseSpark1xTextConfig,
    MuseSparkDSparkConfig,
)

TEXT: dict[str, Any] = {
    "hidden_size": 5120,
    "ffn_dim_multiplier": 1.5,
    "moe": {
        "topk": 8,
        "override_routed_hidden_dim": 1408,
        "future_unknown_key": 1,
    },
    "metap": {"use_metap": True, "metap_mode": "mup", "base_width": 256},
    "gated_delta_net": {"conv_size": 4, "future_unknown_key": 2},
}


def test_muse_import_does_not_load_foreign_platform():
    # Use a fresh interpreter: importing ROCm on CPU can reload the _C
    # extension and abort due to duplicate torch operator registrations.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from vllm.platforms import current_platform
from vllm.models.muse_spark.common.dspark import MuseSparkDSparkForCausalLM

assert current_platform.is_rocm() or "vllm.platforms.rocm" not in sys.modules
""",
        ],
        check=True,
    )


def test_text_config_eos_default():
    assert MuseSpark1xTextConfig().eos_token_id == [200001, 200008]
    assert MuseSpark1xTextConfig(eos_token_id=200001).eos_token_id == 200001


def test_nested_subconfig_dicts_promoted():
    c = MuseSpark1xTextConfig(**TEXT)
    assert isinstance(c.moe, MoEArgs) and c.moe.topk == 8
    assert isinstance(c.metap, MetapArgs) and c.metap.use_metap
    assert isinstance(c.gated_delta_net, GatedDeltaNetArgs)
    assert c.gated_delta_net.conv_size == 4
    assert not hasattr(c.moe, "future_unknown_key")
    assert not hasattr(c.gated_delta_net, "future_unknown_key")


def test_derived_intermediate_sizes():
    c = MuseSpark1xTextConfig(**TEXT)
    assert c.intermediate_size_mlp == int(1.5 * 5120)
    assert c.intermediate_size_shared_experts == int(1.5 * 5120)
    assert c.intermediate_size_routed_experts == 1408


def test_metap_rejects_unknown_mode():
    with pytest.raises(AssertionError):
        MuseSpark1xTextConfig(metap={"use_metap": True, "metap_mode": "bogus"})


def _dspark_config() -> dict:
    return {
        "model_type": "muse_spark_dspark",
        "num_hidden_layers": 2,
        "moe": {"is_enabled": False},
        "global_attn_cfg": [2048],
        "layer_types": ["sliding_attention"] * 2,
        "target_layer_ids": [1, 3],
        "num_target_layers": 2,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "dflash_config": {"causal": True, "target_layer_ids": [1, 3]},
    }


@pytest.mark.parametrize("mode", ["rope", "nope"])
def test_dspark_config_loads(tmp_path, mode):
    raw = _dspark_config() | {"draft_position_mode": mode}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, MuseSparkDSparkConfig)
    assert config.architectures == ["MuseSparkDSparkModel"]
    assert config.use_rope == (mode == "rope")
    assert config.draft_vocab_size == config.vocab_size


def test_dspark_speculative_config_preserves_architecture(tmp_path):
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target_path.mkdir()
    draft_path.mkdir()
    # DSpark reads only sizes from the target, so any registered tiny model works.
    target = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
    }
    (target_path / "config.json").write_text(json.dumps(target))
    (draft_path / "config.json").write_text(json.dumps(_dspark_config()))

    config = SpeculativeConfig(
        model=str(draft_path),
        method="dspark",
        num_speculative_tokens=7,
        target_model_config=ModelConfig(model=str(target_path), tokenizer_mode="skip"),
        target_parallel_config=ParallelConfig(),
    )
    assert config.parallel_drafting
    assert config.draft_model_config.architectures == ["MuseSparkDSparkModel"]
    assert isinstance(config.draft_model_config.hf_config, MuseSparkDSparkConfig)


def test_dspark_output_logit_transform():
    model = object.__new__(MuseSparkDSparkForCausalLM)  # type: ignore[type-abstract]
    nn.Module.__init__(model)
    model.output_multiplier = 0.5
    model.output_soft_cap_temp = 2.0
    logits = torch.tensor([[-4.0, 0.0, 4.0]], dtype=torch.bfloat16)
    expected = 2.0 * torch.tanh(logits.float() * 0.5 / 2.0)
    torch.testing.assert_close(model.apply_output_logit_transform(logits), expected)
