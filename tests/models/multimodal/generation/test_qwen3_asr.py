# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

pytestmark = pytest.mark.skip_global_cleanup


def test_post_process_output_strips_language_prefix():
    raw = "language English<asr_text>hello world"
    assert (
        Qwen3ASRForConditionalGeneration.post_process_output(raw) == "hello world"
    )
