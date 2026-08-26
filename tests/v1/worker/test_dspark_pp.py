# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

from vllm.config import LoadConfig, VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _draft_safetensors_load_config,
)


def test_last_stage_draft_uses_non_collective_safetensors_loader():
    target_load_config = LoadConfig(
        load_format="fastsafetensors",
        download_dir="/models",
    )
    vllm_config = cast(
        VllmConfig,
        SimpleNamespace(load_config=target_load_config),
    )

    draft_load_config = _draft_safetensors_load_config(vllm_config)

    assert draft_load_config is not target_load_config
    assert draft_load_config.load_format == "safetensors"
    assert draft_load_config.download_dir == target_load_config.download_dir
    assert target_load_config.load_format == "fastsafetensors"
