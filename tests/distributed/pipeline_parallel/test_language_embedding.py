# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.model import RunnerOption

from ...utils import create_new_process_for_each_test
from ._common import EMBEDDING_MODELS, ParallelSetup, PPTestOptions, _compare_tp

# NOTE: You can update this on your local machine to run specific tests
TEST_MODELS = [
    "intfloat/e5-mistral-7b-instruct",
    "BAAI/bge-multilingual-gemma2",
]


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "distributed_backend", "runner", "test_options"),
    [
        params
        for model_id, settings in EMBEDDING_MODELS.items()
        for params in settings.iter_params(model_id)
        if model_id in TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_tp_language_embedding(
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
        method="encode",
        is_multimodal=False,
    )
