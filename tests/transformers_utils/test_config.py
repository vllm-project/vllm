# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

import json
import math
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from vllm.config.model import ModelConfig
from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.config import (
    get_safetensors_params_metadata,
    mrope_num_dims,
    patch_legacy_rope_type,
    patch_rope_parameters,
    try_get_generation_config,
    uses_mrope,
)
from vllm.transformers_utils.configs.mistral import adapt_config_dict


@pytest.mark.parametrize("factor", [None, 2.0])
@pytest.mark.parametrize("rope_type", [None, "default", "dynamic", "linear"])
def test_nomic_legacy_rope_scaling(factor, rope_type):
    config = PreTrainedConfig()
    config.model_type = "nomic_bert"
    config.rotary_emb_base = 1000.0
    config.rotary_emb_fraction = 1.0
    config.rotary_scaling_factor = factor
    config.max_position_embeddings = 8192
    if rope_type is not None:
        config.rope_parameters = {"rope_type": rope_type}
        if rope_type != "default":
            config.rope_parameters["factor"] = 4.0

    patch_rope_parameters(config)
    expected_type = rope_type or ("dynamic" if factor else "default")
    expected = {
        "rope_type": expected_type,
        "rope_theta": 1000.0,
        "partial_rotary_factor": 1.0,
    }
    if expected_type != "default":
        expected["factor"] = 4.0 if rope_type else factor
    assert config.rope_parameters == expected

    # get_config also patches get_text_config(), which can be the same object.
    patch_rope_parameters(config)
    assert config.rope_parameters == expected


@pytest.mark.parametrize("layout", ["mixed", "flat"])
def test_gemma4_dspark_rope_config_preserves_parameters(tmp_path, layout):
    """Remove redundant shared entries while preserving per-layer and flat RoPE."""
    from transformers import Gemma4TextConfig

    per_layer = {
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    }
    rope_parameters: dict[str, object] = dict(per_layer)
    if layout == "mixed":
        rope_parameters.update(rope_type="default", rope_theta=None)
    else:
        rope_parameters = {"rope_type": "default", "rope_theta": 12345.0}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "architectures": ["Gemma4DSparkModel"],
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "rope_parameters": rope_parameters,
            }
        )
    )
    _, config = config_module.HFConfigParser().parse(
        tmp_path, trust_remote_code=False, max_position_embeddings=8192
    )
    assert isinstance(config, Gemma4TextConfig)
    assert config.name_or_path == str(tmp_path)
    assert config.max_position_embeddings == 8192
    if layout == "flat":
        assert config.rope_parameters["rope_theta"] == 12345.0
    else:
        for layer_type, expected in per_layer.items():
            for key, value in expected.items():
                assert config.rope_parameters[layer_type][key] == value
        assert set(config.rope_parameters) == set(per_layer)


def test_patch_legacy_rope_type_preserves_nope_layers():
    """NoPE layers stay disabled while later RoPE layers are normalized."""
    rope_parameters = {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "mrope_section": [24, 20, 20],
        },
    }

    patch_legacy_rope_type(rope_parameters)

    assert rope_parameters == {
        "full_attention": None,
        "sliding_attention": {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
        },
    }


def test_patch_legacy_rope_type_normalizes_telechat3_yarn():
    """TeleChat3's RoPE is YaRN with 0.07 in place of the usual 0.1.

    Encoding that as a precomputed attention_factor keeps the config
    plain YaRN, which Transformers and every "yarn" guard understand.
    `mscale` cannot express it: Transformers only applies mscale when
    mscale_all_dim is also truthy.
    """
    rope_parameters = {
        "type": "telechat3-yarn",
        "rope_type": "telechat3-yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
    }

    patch_legacy_rope_type(rope_parameters)

    assert rope_parameters == {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
        "attention_factor": pytest.approx(0.07 * math.log(4.0) + 1.0),
    }


def test_mistral_yarn_apply_scale_false_disables_yarn_magnitude_scaling():
    """`yarn.apply_scale: false` must reach the DeepSeek-style attentions.

    Transformers spells it `attention_factor = 1.0`, which DeepseekV2Attention
    and its siblings read to select `deepseek_llama_scaling` over
    `deepseek_yarn`; without it Mistral-Large-3 runs with a spurious
    yarn_get_mscale(factor)^2 attention scaling.
    """
    params = {
        "dim": 7168,
        "n_layers": 61,
        "head_dim": 192,
        "hidden_dim": 16384,
        "n_heads": 128,
        "n_kv_heads": 128,
        "norm_eps": 1e-5,
        "vocab_size": 131072,
        "rope_theta": 10000.0,
        "max_position_embeddings": 294912,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "moe": {
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "expert_hidden_dim": 4096,
            "first_k_dense_replace": 3,
            "route_every_n": 1,
            "routed_scale": 1.0,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        },
        "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 8192},
        "yarn": {
            "alpha": 1,
            "apply_scale": False,
            "beta": 32,
            "factor": 36,
            "original_max_position_embeddings": 8192,
        },
    }

    config = adapt_config_dict(params, defaults={})

    assert config.architectures == ["MistralLarge3ForCausalLM"]
    assert config.rope_parameters["attention_factor"] == 1.0


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_model_config_generation_fallback_forwards_code_revision():
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            generation_config="auto",
            hf_config_path=None,
            model="org/model",
            trust_remote_code=True,
            revision="model-pin",
            _hf_config_revision=None,
            code_revision="code-pin",
            config_format="auto",
            hf_token=None,
        ),
    )

    with (
        patch.object(
            config_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError,
        ),
        patch.object(
            config_module,
            "get_config",
            return_value=PreTrainedConfig(),
        ) as get_config,
    ):
        ModelConfig.try_get_generation_config(model_config)

    get_config.assert_called_once_with(
        "org/model",
        trust_remote_code=True,
        revision="model-pin",
        code_revision="code-pin",
        config_format="auto",
        token=None,
    )


def test_safetensors_metadata_of_repo_without_safetensors():
    """A repo storing its weights in another format is an answer, not a failure,
    so it must not be retried."""
    from huggingface_hub.errors import LocalEntryNotFoundError, NotASafetensorsRepoError

    get_safetensors_metadata = MagicMock(
        side_effect=NotASafetensorsRepoError("not a safetensors repo")
    )
    api = SimpleNamespace(
        get_safetensors_metadata=get_safetensors_metadata,
        list_repo_files=MagicMock(return_value=["pytorch_model.bin"]),
        snapshot_download=MagicMock(side_effect=LocalEntryNotFoundError("no cache")),
    )

    with patch.object(config_module, "hf_api", lambda: api):
        assert get_safetensors_params_metadata("some/pytorch-only-model") == {}

    get_safetensors_metadata.assert_called_once()


def test_safetensors_metadata_of_repo_with_a_nonstandard_file_name():
    """`get_safetensors_metadata` only knows `model.safetensors` and its index,
    so older checkpoints are read through the file listing instead."""
    from huggingface_hub.errors import NotASafetensorsRepoError
    from huggingface_hub.utils import TensorInfo

    weights = "gptq_model-4bit-128g.safetensors"
    tensor = TensorInfo(dtype="I32", shape=[5632], data_offsets=(0, 22528))
    parse_safetensors_file_metadata = MagicMock(
        return_value=SimpleNamespace(tensors={"layers.0.mlp.down_proj.qweight": tensor})
    )
    api = SimpleNamespace(
        get_safetensors_metadata=MagicMock(
            side_effect=NotASafetensorsRepoError("not a safetensors repo")
        ),
        list_repo_files=MagicMock(return_value=["config.json", weights]),
        parse_safetensors_file_metadata=parse_safetensors_file_metadata,
    )

    with patch.object(config_module, "hf_api", lambda: api):
        metadata = get_safetensors_params_metadata("some/old-gptq-model")

    assert metadata["layers.0.mlp.down_proj.qweight"]["dtype"] == "I32"
    parse_safetensors_file_metadata.assert_called_once_with(
        "some/old-gptq-model", weights, revision=None
    )


@pytest.mark.parametrize(
    ("section_key", "mrope_section", "expected_num_dims"),
    [
        ("mrope_section", [16, 24, 24], 3),
        ("mrope_section", [16, 16, 16, 16], 4),
        # Interleaved M-RoPE takes 2 sections but still consumes 3D positions
        ("mrope_section", [32, 32], 3),
        # HunYuan-VL checkpoints ship the section under its legacy name
        ("xdrope_section", [16, 16, 16, 16], 4),
    ],
)
def test_mrope_num_dims(section_key, mrope_section, expected_num_dims):
    config = PreTrainedConfig()
    config.rope_parameters = {"rope_type": "default", section_key: mrope_section}

    assert uses_mrope(config)
    assert mrope_num_dims(config) == expected_num_dims


@pytest.mark.parametrize("section_name", ["mrope_section", "xdrope_section"])
def test_mrope_num_dims_from_config_attribute(section_name):
    """Some configs expose the section as an attribute rather than under
    `rope_parameters`."""
    config = PreTrainedConfig()
    setattr(config, section_name, [16, 16, 16, 16])

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_from_nested_rope_parameters():
    """Sections nested by layer type must be found, not silently defaulted."""
    config = PreTrainedConfig()
    config.rope_parameters = {
        "full_attention": {"mrope_section": [16, 16, 16, 16]},
        "linear_attention": {"rope_type": "default"},
    }

    assert uses_mrope(config)
    assert mrope_num_dims(config) == 4


def test_mrope_num_dims_without_mrope():
    assert mrope_num_dims(PreTrainedConfig()) == 0
