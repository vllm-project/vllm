# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash indexer kernel block sizes are DeepGEMM pool pages."""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.attention.backends.mla.indexer import Glm5NextIndexerBackend

KPOOL = 4


@pytest.mark.parametrize(
    "rocm,sm120,expected",
    [
        (False, False, [KPOOL * 32, KPOOL * 64]),
        # SM120 DeepGEMM takes only 64-entry pages for the fp8 cache.
        (False, True, [KPOOL * 64]),
        # ROCm's AITER paged-MQA logits take the same pool pages.
        (True, False, [KPOOL * 32, KPOOL * 64]),
    ],
)
def test_glm5next_indexer_kernel_block_sizes(monkeypatch, rocm, sm120, expected):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family: sm120 and family == 120,
    )
    model_config = SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=KPOOL))
    config = SimpleNamespace(model_config=model_config)
    monkeypatch.setattr(indexer, "get_current_vllm_config", lambda: config)

    spec = SimpleNamespace(tokens_per_state=KPOOL)
    assert Glm5NextIndexerBackend.get_supported_kernel_block_sizes(spec) == expected
    # Platform block-size selection queries without a spec.
    assert Glm5NextIndexerBackend.get_supported_kernel_block_sizes() == expected
