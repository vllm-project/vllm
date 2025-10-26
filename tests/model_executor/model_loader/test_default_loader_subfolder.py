# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def test_default_loader_prefers_llm_subfolder_and_filters_with_index(tmp_path):
    # Create local repo layout with llm/ subfolder
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()

    keep = llm_dir / "model-00001-of-00002.safetensors"
    drop = llm_dir / "model-00002-of-00002.safetensors"
    keep.write_bytes(b"0")
    drop.write_bytes(b"0")

    # Create index file within llm/ that only references the first shard
    index = llm_dir / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": keep.name}}))

    # Default loader in auto format should find llm/*.safetensors and use the subfolder index
    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    hf_folder, files, use_safetensors = loader._prepare_weights(
        str(tmp_path), revision=None, fall_back_to_pt=True, allow_patterns_overrides=None
    )

    assert hf_folder == str(tmp_path)
    assert use_safetensors is True
    assert files == [str(keep)]

