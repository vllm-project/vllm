# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize(
    "backend",
    ["TORCH_SDPA", AttentionBackendEnum.TORCH_SDPA],
)
def test_torch_sdpa_cannot_be_used_for_decoder_attention(backend):
    with pytest.raises(
        ValueError,
        match="only supported for multi-modal encoder attention",
    ):
        AttentionConfig(backend=backend)
