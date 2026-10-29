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
    try_get_generation_config,
    uses_mrope,
)
from vllm.transformers_utils.configs.mistral import adapt_config_dict


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


def _mistral_large_4_params() -> dict:
    """A minimal Mistral Large 4 params.json (block-FP8 quantized)."""
    return {
        "dim": 256,
        "n_layers": 8,
        "head_dim": 128,
        "hidden_dim": 512,
        "n_heads": 4,
        "n_kv_heads": 4,
        "norm_eps": 1e-5,
        "vocab_size": 1024,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "nope": "rope,rope,rope,nope",
        "softmax_tanh_gating": 30.0,
        "ssmax_scaling": True,
        "lerope": True,
        "quantization": {
            "qscheme_act": "BLOCK",
            "qformat_weight": "fp8_e4m3",
            "qscheme_weight": "BLOCK",
            "block_size_out": 128,
            "block_size_in": 128,
        },
        "yarn": {
            "original_max_position_embeddings": 512,
            "factor": 8,
            "apply_scale": False,
            "beta": 32,
            "alpha": 1,
        },
        "kv_sharing": {"source_layers_str": "0,1,2,3,4,5,6,3"},
        "moe": {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "expert_hidden_dim": 128,
            "first_k_dense_replace": 1,
            "route_every_n": 1,
            "routed_scale": 1.0,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
        },
        "dsa": {
            "index_topk": 64,
            "index_n_heads": 4,
            "index_head_dim": 128,
            "index_rope_head_dim": 64,
            "indexer_match_nope": True,
            "indexer_ssmax_scaling": True,
            "indexer_norm_eps": 1e-6,
            "index_sharing": "full,shared,full,full",
            "fp8_indexer": True,
            "fp8_kv_cache": True,
        },
    }


_UNPREFIXED_DSA = {
    "topk": 64,
    "n_heads": 4,
    "head_dim": 128,
    "rope_head_dim": 64,
    "match_nope": True,
    "ssmax_scaling": True,
    "norm_eps": 1e-6,
    "sharing": "full,shared,full,full",
    "fp8_indexer": True,
    "fp8_kv_cache": True,
}


@pytest.mark.parametrize("dsa_keys", ["prefixed", "unprefixed"])
def test_mistral_large_4_per_layer_patterns(dsa_keys):
    """Mistral Large 4 patterns expand into per-layer lists and layer specs,
    whichever spelling of the `dsa` keys the checkpoint uses."""
    from vllm.models.mistral.common.layer_spec import get_layer_specs

    params = _mistral_large_4_params()
    if dsa_keys == "unprefixed":
        params["dsa"] = dict(_UNPREFIXED_DSA)

    config = adapt_config_dict(params, defaults={})

    assert config.architectures == ["MistralLarge4ForCausalLM"]
    assert config.quantization_config["weight_block_size"] == [128, 128]
    assert (
        config.index_topk,
        config.index_n_heads,
        config.index_head_dim,
        config.indexer_norm_eps,
        config.indexer_ssmax_scaling,
    ) == (64, 4, 128, 1e-6, True)
    assert config.nope_layer_flags == [False, False, False, True] * 2
    assert config.indexer_nope_layer_flags == config.nope_layer_flags
    assert config.kv_source_layers == [0, 1, 2, 3, 4, 5, 6, 3]
    assert config.index_sharing == ["full", "shared", "full", "full"] * 2
    assert config.fp8_kv_cache

    specs = get_layer_specs(config)
    assert [s.layer_idx for s in specs if s.shares_kv] == [7]
    assert [s.has_indexer for s in specs] == [True, False, True, True] * 2
    assert [s.is_moe for s in specs] == [False] + [True] * 7
    assert [s.use_rope for s in specs] == [True, True, True, False] * 2


def test_mistral_large_4_dsa_key_spellings():
    """Both spellings of a `dsa` key are accepted only when they agree."""
    params = _mistral_large_4_params()
    params["dsa"] = {**params["dsa"], "topk": 64}
    assert adapt_config_dict(params, defaults={}).index_topk == 64

    params = _mistral_large_4_params()
    params["dsa"] = {**params["dsa"], "topk": 128}
    with pytest.raises(AssertionError, match="Conflicting dsa keys"):
        adapt_config_dict(params, defaults={})


@pytest.mark.parametrize("vision", [False, True])
def test_mistral_large_4_keeps_modelopt_quantization_config(vision):
    """A flat ModelOpt `quantization_config` keeps its checkpoint layer names
    (the model maps them to its modules) and selects the mixed-precision
    method, for the text model and for its Pixtral host."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )

    quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "ignore": [],
        "quantized_layers": {
            "layers.1.attention.wq": {"quant_algo": "FP8_PB_WO"},
            "layers.1.experts": {"group_size": 16, "quant_algo": "NVFP4"},
        },
    }
    params = _mistral_large_4_params()
    params.pop("quantization")
    params["quantization_config"] = quantization_config
    if vision:
        params["vision_encoder"] = {"hidden_size": 64, "num_hidden_layers": 1}

    config = adapt_config_dict(params, defaults={})

    text_config = config.text_config if vision else config
    assert text_config.architectures == ["MistralLarge4ForCausalLM"]
    assert config.quantization_config == quantization_config
    assert (
        ModelOptMixedPrecisionConfig.override_quantization_method(
            config.quantization_config, None
        )
        == "modelopt_mixed"
    )


def _mistral_large_4_dspark_config(**mtp) -> PreTrainedConfig:
    """Draft config for the `mtp` block of the HF checkpoint, scaled down."""
    from vllm.config.speculative import SpeculativeConfig

    params = _mistral_large_4_params()
    params["mtp"] = {
        "n_layers": 3,
        "depth": 3,
        "ragged_attention": "2048,2048,2048",
        "target_activation_layers": "1,3,6",
        "norm_strategy": "eagle",
        "dspark": {
            "markov_rank": 16,
            "head_type": "rnn",
            "draft_input": "position_specific",
        },
        **mtp,
    }
    target = adapt_config_dict(params, defaults={})
    return SpeculativeConfig.hf_config_override(target)


def test_mistral_large_4_dspark_draft_config():
    """The params.json `mtp` block becomes a BF16, non-causal, windowed MLA
    draft with the target YaRN, reading 0-based target layer outputs (aux ids
    are +1)."""
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    draft = _mistral_large_4_dspark_config()

    assert draft.architectures == ["MistralLarge4DSparkModel"]
    assert draft.num_hidden_layers == 3
    assert draft.dspark_block_size == draft.n_predict == 3
    assert (draft.dspark_head_type, draft.dspark_draft_input) == (
        "rnn",
        "position_specific",
    )
    assert draft.num_target_layers == 3
    assert draft.dflash_config["mask_token_id"] == draft.vocab_size
    assert not draft.dflash_config["causal"]
    assert not draft.ssmax_scaling
    assert draft.quantization_config is None
    assert draft.layer_types == ["sliding_attention"] * 3
    assert draft.sliding_window == 2048
    assert draft.rope_parameters["rope_type"] == "yarn"
    assert not hasattr(draft, "mtp_config")
    spec = SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft))
    assert get_eagle3_aux_layers_from_config(spec) == (2, 4, 7)


def test_mistral_large_4_dspark_variants_and_rejections():
    draft = _mistral_large_4_dspark_config(
        ragged_attention=None, dspark={"markov_rank": 16}
    )
    assert (draft.dspark_head_type, draft.dspark_draft_input) == (
        "vanilla",
        "shared_mask",
    )
    assert draft.layer_types == ["full_attention"] * 3
    assert draft.sliding_window is None
    with pytest.raises(NotImplementedError, match="one sliding window"):
        _mistral_large_4_dspark_config(ragged_attention="2048,1024")
    with pytest.raises(AssertionError, match="Unparsed"):
        _mistral_large_4_dspark_config(dspark={"markov_rank": 16, "qk_norm": True})
    with pytest.raises(ValueError, match="Unsupported DSpark draft input"):
        _mistral_large_4_dspark_config(
            dspark={"markov_rank": 16, "draft_input": "slot_deltas"}
        )
    with pytest.raises(ValueError, match="depth > 1"):
        _mistral_large_4_dspark_config(depth=1)


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
