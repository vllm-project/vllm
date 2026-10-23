# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import LoadConfig
from vllm.v1.worker.gpu.spec_decode import utils as spec_decode_utils


def test_last_stage_draft_uses_non_collective_safetensors_loader(monkeypatch):
    target_load_config = LoadConfig(
        load_format="fastsafetensors",
        download_dir="/models",
    )
    monkeypatch.setattr(
        spec_decode_utils, "get_pp_group", lambda: SimpleNamespace(world_size=2)
    )

    draft_load_config = spec_decode_utils.get_pp_safe_draft_load_config(
        target_load_config
    )

    assert draft_load_config is not target_load_config
    assert draft_load_config.load_format == "auto"
    assert draft_load_config.download_dir == target_load_config.download_dir
    assert target_load_config.load_format == "fastsafetensors"
