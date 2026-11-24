# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling2 module paths that moved from `nvidia/` to `common/` stay
importable: each old path is the moved module object (`nvidia/__init__.py`,
`_MOVED`), the names that moved out of the partly moved `nvidia/ops`
modules resolve to the moved objects, and the package exports load the
`common/` model classes."""

import importlib

import pytest

from vllm.models.inkling2 import nvidia

# Pinned here, not read from `_MOVED`, so that a dropped entry fails.
MOVED_PATHS = {
    "attention": "common.attention",
    "loop": "common.loop",
    "model": "common.model",
    "moe": "common.moe",
    "sconv": "common.sconv",
    "ops.attention_sm90": "common.ops.attention_sm90",
    "ops.attn_res": "common.ops.attn_res",
    "ops.conv_mix": "common.ops.conv_mix",
    "ops.e4m3": "common.ops.e4m3",
    "ops.linear": "common.ops.linear",
    "ops.loop": "common.ops.loop",
    "ops.norm": "common.ops.norm",
    "ops.pdl": "common.ops.pdl",
    "ops.sconv": "common.ops.sconv",
}


def test_moved_paths_are_pinned():
    assert nvidia._MOVED == MOVED_PATHS


@pytest.mark.parametrize("old,new", sorted(MOVED_PATHS.items()))
def test_old_module_path_is_the_moved_module(old: str, new: str):
    moved = importlib.import_module(f"vllm.models.inkling2.{new}")
    assert importlib.import_module(f"vllm.models.inkling2.nvidia.{old}") is moved


@pytest.mark.parametrize(
    "old,name,new",
    [
        ("ops.moe", "combine", "ops.moe"),
        ("ops.moe_sm90", "select_experts", "ops.moe"),
        ("ops.attention", "rel_decode_attention", "ops.attention"),
        ("ops.activation", "_div_rn", "ops.activation"),
        ("ops.moe_sm90", "_situ", "ops.activation"),
    ],
)
def test_names_moved_out_resolve_from_the_old_module(old: str, name: str, new: str):
    old_module = importlib.import_module(f"vllm.models.inkling2.nvidia.{old}")
    new_module = importlib.import_module(f"vllm.models.inkling2.common.{new}")
    assert getattr(old_module, name) is getattr(new_module, name)


@pytest.mark.parametrize(
    "name,module",
    [
        ("Inkling2ForConditionalGeneration", "common.model"),
        ("Inkling2LoopForConditionalGeneration", "common.loop"),
    ],
)
def test_package_exports_load_the_common_model_classes(name: str, module: str):
    """The model classes resolve through the platform entry module
    (`nvidia.model` / `nvidia.loop` on CUDA, old paths with no file)."""
    inkling2 = importlib.import_module("vllm.models.inkling2")
    assert getattr(inkling2, name).__module__ == f"vllm.models.inkling2.{module}"
