# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...standard._voxtral import run_hf_reference, run_online_serving


def test_online_serving(vllm_runner, audio_assets):
    run_online_serving(vllm_runner, audio_assets)


@pytest.mark.skip(
    reason="VoxtralProcessor.apply_chat_template() in transformers v5 "
    "doesn't resolve chat_template=None to the default template"
)
def test_hf_reference(hf_runner, vllm_runner, audio_assets):
    run_hf_reference(hf_runner, vllm_runner, audio_assets)
