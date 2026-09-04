# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import LoadConfig
from vllm.v1.worker.gpu.spec_decode import utils as spec_decode_utils


@pytest.mark.parametrize(
    "pp_size,load_format,expected_format,returns_copy",
    [
        (2, "fastsafetensors", "auto", True),
        (1, "fastsafetensors", "fastsafetensors", False),
        (2, "auto", "auto", False),
    ],
)
def test_pp_safe_draft_load_config(
    monkeypatch, pp_size, load_format, expected_format, returns_copy
):
    monkeypatch.setattr(
        spec_decode_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=pp_size),
    )
    load_config = LoadConfig(load_format=load_format)

    result = spec_decode_utils.get_pp_safe_draft_load_config(load_config)

    assert result.load_format == expected_format
    assert (result is not load_config) == returns_copy
    assert load_config.load_format == load_format
