# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...standard._whisper import (
    resampled_assets as resampled_assets,
)
from ...standard._whisper import (
    run_beam_search_encoder_decoder,
    run_parse_language_detection_output,
)
from ...standard._whisper import (
    use_spawn_for_whisper as use_spawn_for_whisper,
)


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [64])
def test_beam_search_encoder_decoder(
    hf_runner,
    vllm_runner,
    dtype: str,
    max_tokens: int,
    resampled_assets,  # noqa: F811
) -> None:
    run_beam_search_encoder_decoder(
        hf_runner, vllm_runner, dtype, max_tokens, resampled_assets
    )


def test_parse_language_detection_output():
    run_parse_language_detection_output()
