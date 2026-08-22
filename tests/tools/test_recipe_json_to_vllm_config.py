# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tools.recipes.recipe_json_to_vllm_config import argv_to_config


def test_argv_to_config_parses_dotted_compilation_config_alias() -> None:
    config = argv_to_config(
        [
            "vllm",
            "serve",
            "openai/gpt-oss-120b",
            "--attention-backend",
            "ROCM_AITER_UNIFIED_ATTN",
            "-cc.pass_config.fuse_rope_kvcache=True",
            "-cc.use_inductor_graph_partition=True",
        ]
    )

    assert config["attention-backend"] == "ROCM_AITER_UNIFIED_ATTN"
    assert config["compilation-config"] == {
        "pass_config": {"fuse_rope_kvcache": True},
        "use_inductor_graph_partition": True,
    }


def test_argv_to_config_preserves_scalar_short_aliases() -> None:
    config = argv_to_config(
        [
            "vllm",
            "serve",
            "openai/gpt-oss-120b",
            "-tp",
            "2",
            "-pp",
            "3",
            "-dp",
            "4",
        ]
    )

    assert config["tensor-parallel-size"] == 2
    assert config["pipeline-parallel-size"] == 3
    assert config["data-parallel-size"] == 4


def test_argv_to_config_rejects_dotted_non_compilation_alias() -> None:
    with pytest.raises(ValueError, match="Unexpected positional/short argument"):
        argv_to_config(
            [
                "vllm",
                "serve",
                "openai/gpt-oss-120b",
                "-tp.foo=1",
            ]
        )
