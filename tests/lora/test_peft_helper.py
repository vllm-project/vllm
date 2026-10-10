# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import shutil

import pytest

from vllm.config.lora import LoRAConfig
from vllm.lora.peft_helper import PEFTHelper

ERROR_CASES = [
    (
        "test_rank",
        {"r": 1024},
        "is greater than max_lora_rank",
    ),
    ("test_dora", {"use_dora": True}, "does not yet support DoRA"),
    (
        "test_modules_to_save",
        {"modules_to_save": ["lm_head"]},
        "Unsupported modules_to_save",
    ),
    (
        "test_rank_pattern",
        {"r": 8, "rank_pattern": {"q_proj": 32}},
        "LoRA rank 32 is greater than max_lora_rank",
    ),
    ("test_rank_zero", {"r": 0}, "must be a positive integer"),
    ("test_rank_negative", {"r": -8}, "must be a positive integer"),
    ("test_lora_bias", {"lora_bias": True}, "does not support LoRA bias"),
    ("test_pissa", {"init_lora_weights": "pissa"}, "init_lora_weights='pissa'"),
    (
        "test_pissa_niter",
        {"init_lora_weights": "pissa_niter_4"},
        "init_lora_weights='pissa_niter_4'",
    ),
    ("test_olora", {"init_lora_weights": "olora"}, "init_lora_weights='olora'"),
    ("test_corda", {"init_lora_weights": "corda"}, "init_lora_weights='corda'"),
    ("test_loftq", {"init_lora_weights": "loftq"}, "init_lora_weights='loftq'"),
    (
        "test_unknown_init",
        {"init_lora_weights": "future_init"},
        "init_lora_weights='future_init'",
    ),
    ("test_alora", {"alora_invocation_tokens": [1, 2]}, "Activated LoRA"),
    ("test_layer_replication", {"layer_replication": [[0, 2]]}, "layer_replication"),
    ("test_bdlora", {"use_bdlora": {"nblocks": 2}}, "BD-LoRA"),
    ("test_qalora", {"use_qalora": True}, "QALoRA"),
]


def test_peft_helper_pass(llama32_lora_files, tmp_path):
    peft_helper = PEFTHelper.from_local_dir(
        llama32_lora_files, max_position_embeddings=4096
    )
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    peft_helper.validate_legal(lora_config)
    assert peft_helper.r == 8
    assert peft_helper.lora_alpha == 32
    target_modules = sorted(peft_helper.target_modules)

    assert target_modules == [
        "down_proj",
        "embed_tokens",
        "gate_proj",
        "k_proj",
        "lm_head",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    assert peft_helper.vllm_max_position_embeddings == 4096

    # test RSLoRA
    rslora_config = dict(use_rslora=True)
    test_dir = tmp_path / "test_rslora"
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(rslora_config)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)

    peft_helper = PEFTHelper.from_local_dir(test_dir, max_position_embeddings=4096)
    peft_helper.validate_legal(lora_config)
    scaling = peft_helper.lora_alpha / math.sqrt(peft_helper.r)
    assert abs(peft_helper.vllm_lora_scaling_factor - scaling) < 1e-3


@pytest.mark.parametrize("test_name,config_change,expected_error", ERROR_CASES)
def test_peft_helper_error(
    llama32_lora_files,
    tmp_path,
    test_name: str,
    config_change: dict,
    expected_error: str,
):
    test_dir = tmp_path / test_name
    shutil.copytree(llama32_lora_files, test_dir)

    # Load and modify configuration
    config_path = test_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    # Apply configuration changes
    adapter_config.update(config_change)

    # Save modified configuration
    with open(config_path, "w") as f:
        json.dump(adapter_config, f)
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    # Test loading the adapter
    with pytest.raises(ValueError, match=expected_error):
        PEFTHelper.from_local_dir(
            test_dir, max_position_embeddings=4096
        ).validate_legal(lora_config)


@pytest.mark.parametrize("bad_rank", [0, -1, -8])
def test_peft_helper_invalid_rank_direct(bad_rank: int):
    """Regression test: constructing a PEFTHelper with a non-positive rank
    must raise a clear ValueError instead of crashing with an unrelated
    ZeroDivisionError (r=0) or silently succeeding with a sign-flipped
    scaling factor that validate_legal() never catches (r<0, since its only
    rank check is the upper bound against max_lora_rank).

    Network-free: constructs PEFTHelper directly rather than going through
    from_local_dir(), which needs an on-disk adapter_config.json.
    """
    with pytest.raises(ValueError, match="must be a positive integer"):
        PEFTHelper(r=bad_rank, lora_alpha=16, target_modules=["q_proj"])


@pytest.mark.parametrize(
    "init_lora_weights",
    [True, False, "gaussian", "eva", "orthogonal", "mica", "lora_ga"],
)
def test_peft_helper_init_lora_weights_supported(init_lora_weights):
    """A saved adapter with these inits loads in PEFT as a plain LoRA."""
    lora_config = LoRAConfig(max_lora_rank=16, max_cpu_loras=3, max_loras=2)
    PEFTHelper.from_dict(
        {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "init_lora_weights": init_lora_weights,
            "alora_invocation_tokens": None,
            "layer_replication": None,
            "use_bdlora": None,
        }
    ).validate_legal(lora_config)


@pytest.mark.parametrize(
    "config,module_name,expected",
    [
        # rank_pattern only: alpha / r_m
        ({"rank_pattern": {"q_proj": 4}}, "model.layers.0.self_attn.q_proj", (4, 8.0)),
        ({"rank_pattern": {"q_proj": 4}}, "model.layers.0.self_attn.v_proj", (16, 2.0)),
        # alpha_pattern only: alpha_m / r
        (
            {"alpha_pattern": {"v_proj": 64}},
            "model.layers.0.self_attn.v_proj",
            (16, 4.0),
        ),
        # keys only match at a "." boundary, like in PEFT
        ({"rank_pattern": {"proj": 4}}, "model.layers.0.self_attn.q_proj", (16, 2.0)),
        # fused MoE experts from PEFT `target_parameters`: gate_up_proj is stored
        # as `experts.base_layer`, down_proj as `experts`
        (
            {"rank_pattern": {"experts.gate_up_proj": 2}},
            "model.layers.0.mlp.experts.base_layer",
            (2, 16.0),
        ),
        (
            {"rank_pattern": {"experts.down_proj": 2}},
            "model.layers.0.mlp.experts",
            (2, 16.0),
        ),
        (
            {"rank_pattern": {"experts.gate_up_proj": 2}},
            "model.layers.0.mlp.experts",
            (16, 2.0),
        ),
    ],
)
def test_peft_helper_rank_and_alpha_pattern(config, module_name, expected):
    peft_helper = PEFTHelper(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], **config
    )
    rank, scaling = peft_helper.get_rank_and_scaling(module_name)
    assert rank == expected[0]
    assert scaling == pytest.approx(expected[1])


def test_peft_helper_pattern_rslora():
    peft_helper = PEFTHelper(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj"],
        use_rslora=True,
        rank_pattern={"q_proj": 4},
    )
    assert peft_helper.get_rank_and_scaling("model.q_proj") == (4, 32 / math.sqrt(4))


def test_peft_helper_dynamic_rank_conversion():
    """PEFT's `save_as_lora` with a dynamic rank writes r=1, lora_alpha=1 and the
    same per-module values in rank_pattern and alpha_pattern, so every module
    keeps a scaling of 1."""
    peft_helper = PEFTHelper(
        r=1,
        lora_alpha=1,
        target_modules=["q_proj", "v_proj"],
        rank_pattern={"layers.0.self_attn.q_proj": 32, "layers.0.self_attn.v_proj": 3},
        alpha_pattern={"layers.0.self_attn.q_proj": 32, "layers.0.self_attn.v_proj": 3},
    )
    assert peft_helper.get_rank_and_scaling("model.layers.0.self_attn.q_proj") == (
        32,
        1.0,
    )
    assert peft_helper.get_rank_and_scaling("model.layers.0.self_attn.v_proj") == (
        3,
        1.0,
    )
    with pytest.raises(ValueError, match="LoRA rank 32 is greater than max_lora_rank"):
        peft_helper.validate_legal(LoRAConfig(max_lora_rank=16))


def test_peft_helper_null_patterns():
    peft_helper = PEFTHelper.from_dict(
        {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "rank_pattern": None,
            "alpha_pattern": None,
        }
    )
    assert peft_helper.get_rank_and_scaling("model.q_proj") == (8, 2.0)
