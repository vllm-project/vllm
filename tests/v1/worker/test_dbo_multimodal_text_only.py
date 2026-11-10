# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode, VllmConfig


def _config(supports_mm_inputs, is_multimodal_model=True):
    return SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(
            is_multimodal_model=is_multimodal_model,
            supports_multimodal_inputs=supports_mm_inputs,
            is_encoder_decoder=False,
            is_hybrid=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
        is_mm_encoder_only=False,
    )


@pytest.mark.parametrize(
    "supports_mm_inputs,allowed",
    [
        # A multimodal checkpoint served text-only, e.g. --language-model-only.
        (False, True),
        (True, False),
    ],
)
def test_dbo_rejects_only_real_multimodal_inputs(supports_mm_inputs, allowed):
    unsupported = VllmConfig._get_dbo_unsupported_features(_config(supports_mm_inputs))
    assert ("dual batch overlap with multimodal models" not in unsupported) is allowed
