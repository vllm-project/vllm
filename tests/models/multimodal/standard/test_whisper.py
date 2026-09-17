# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ....utils import create_new_process_for_each_test, multi_gpu_test
from ._whisper import (
    input_audios as input_audios,
)
from ._whisper import (
    resampled_assets as resampled_assets,
)
from ._whisper import (
    run_encoder_cache_cleanup,
    run_models,
    run_models_distributed,
)
from ._whisper import (
    use_spawn_for_whisper as use_spawn_for_whisper,
)


@pytest.mark.core_model
@pytest.mark.cpu_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("dtype", ["half", "float"])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_models(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    num_logprobs: int,
    input_audios,  # noqa: F811
    enforce_eager: bool,
) -> None:
    run_models(
        hf_runner,
        vllm_runner,
        model,
        dtype,
        num_logprobs,
        input_audios,
        enforce_eager,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.core_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [200])
@pytest.mark.parametrize("num_logprobs", [5])
@create_new_process_for_each_test("spawn")
def test_models_distributed(
    hf_runner,
    vllm_runner,
    model: str,
    distributed_executor_backend: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    input_audios,  # noqa: F811
) -> None:
    run_models_distributed(
        hf_runner,
        vllm_runner,
        model,
        distributed_executor_backend,
        dtype,
        max_tokens,
        num_logprobs,
        input_audios,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("model", ["openai/whisper-large-v3-turbo"])
def test_encoder_cache_cleanup(
    vllm_runner,
    model: str,
    input_audios,  # noqa: F811
    monkeypatch,
) -> None:
    run_encoder_cache_cleanup(vllm_runner, model, input_audios, monkeypatch)
