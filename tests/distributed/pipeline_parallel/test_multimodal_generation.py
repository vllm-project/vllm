# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.model import RunnerOption

from ...utils import create_new_process_for_each_test
from ._common import MULTIMODAL_MODELS, ParallelSetup, PPTestOptions, _compare_tp

# NOTE: You can update this on your local machine to run specific tests
TEST_MODELS = [
    "OpenGVLab/InternVL3-1B",
    "microsoft/Phi-3.5-vision-instruct",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
]


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "distributed_backend", "runner", "test_options"),
    [
        params
        for model_id, settings in MULTIMODAL_MODELS.items()
        for params in settings.iter_params(model_id)
        if model_id in TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_tp_multimodal_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: PPTestOptions,
    num_gpus_available,
):
    _compare_tp(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        method="generate",
        is_multimodal=True,
    )
