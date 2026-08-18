# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _resolve_dspark_attention_backend,
)


@pytest.mark.parametrize(
    ("model_type", "draft_backend", "target_backend", "expected"),
    [
        pytest.param(
            "deepseek_v4",
            None,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
            id="deepseek-v4-inherits-target",
        ),
        pytest.param(
            "qwen3",
            None,
            AttentionBackendEnum.FLASHINFER_MLA,
            None,
            id="qwen-draft-auto-selects",
        ),
        pytest.param(
            "deepseek_v4",
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
            AttentionBackendEnum.FLASH_ATTN,
            id="explicit-draft-backend-wins",
        ),
    ],
)
def test_resolve_dspark_attention_backend(
    model_type: str,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
    expected: AttentionBackendEnum | None,
):
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type)
    )

    assert (
        _resolve_dspark_attention_backend(
            draft_model_config, draft_backend, target_backend
        )
        is expected
    )
