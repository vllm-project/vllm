# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`indexer_kv_dtype='mxfp4'` on ROCm runs on aiter's paged MXFP4 kernels,
which the indexer imports directly. Selecting it needs the aiter package, not
the global VLLM_ROCM_USE_AITER switch.
"""

import types

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

import vllm._aiter_ops as aiter_ops
import vllm.platforms.rocm as rocm_platform
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4


def _config(model_type: str = "deepseek_v4"):
    return types.SimpleNamespace(
        attention_config=AttentionConfig(indexer_kv_dtype="mxfp4"),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=model_type)
        ),
    )


@pytest.fixture
def cdna4_without_aiter_switch(monkeypatch):
    monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: 4)
    monkeypatch.setattr(aiter_ops.rocm_aiter_ops, "_AITER_ENABLED", False)


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
def test_mxfp4_indexer_does_not_need_aiter_switch(
    cdna4_without_aiter_switch, monkeypatch, model_type
):
    monkeypatch.setattr(aiter_ops, "IS_AITER_FOUND", True)
    assert dsa_indexer_uses_fp4(_config(model_type))


def test_mxfp4_indexer_needs_aiter_package(cdna4_without_aiter_switch, monkeypatch):
    monkeypatch.setattr(aiter_ops, "IS_AITER_FOUND", False)
    with pytest.raises(ValueError, match="install aiter"):
        dsa_indexer_uses_fp4(_config())
