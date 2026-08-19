# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.scoring.serving import ServingScores


def test_scoring_preprocessing_uses_dedicated_executor():
    renderer_executor = ThreadPoolExecutor(max_workers=1)

    def init_pooling_serving(self, *args, **kwargs):
        self._executor = renderer_executor
        self._preprocessing = lambda *args, **kwargs: None

    model_config = SimpleNamespace(
        architecture="XLMRobertaForSequenceClassification",
        renderer_num_workers=2,
        get_pooling_task=lambda _: "classify",
    )
    engine_client = SimpleNamespace(model_config=model_config)

    try:
        with patch.object(PoolingServing, "__init__", init_pooling_serving):
            serving = ServingScores(
                engine_client,
                supported_tasks=("classify",),
            )

        assert serving._score_preprocessing_executor is not renderer_executor
        assert serving._score_preprocessing_executor._max_workers == 2

        serving.shutdown()
        assert serving._score_preprocessing_executor._shutdown
    finally:
        renderer_executor.shutdown(wait=False)
