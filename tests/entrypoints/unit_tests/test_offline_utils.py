# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm import PoolingParams, SamplingParams
from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.lora.request import LoRARequest
from vllm.utils.counter import Counter


@pytest.fixture
def mixin() -> OfflineInferenceMixin:
    return object.__new__(OfflineInferenceMixin)


def test_rejects_mismatched_params(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._params_to_seq([SamplingParams()], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and params (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_lora_requests(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._lora_request_to_seq([None], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and lora_request (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_priority(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._priority_to_seq([0], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and priority (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_matching_lengths_pass_through(mixin: OfflineInferenceMixin):
    lora = LoRARequest("l", 1, "/tmp/l")

    assert mixin._params_to_seq([SamplingParams()], num_requests=1) == [
        SamplingParams()
    ]
    assert mixin._lora_request_to_seq([lora], num_requests=1) == [lora]
    assert mixin._priority_to_seq([3], num_requests=1) == [3]


def _configure_request_submission(mixin: OfflineInferenceMixin) -> MagicMock:
    engine = MagicMock()
    engine.vllm_config.lora_config = None
    mixin.llm_engine = engine
    mixin.request_counter = Counter()
    return engine


def test_text_pooling_requests_use_bulk_submission(mixin: OfflineInferenceMixin):
    engine = _configure_request_submission(mixin)
    engine.add_requests.return_value = ["0", "1"]
    prompts = [tokens_input([1]), tokens_input([2])]

    request_ids = mixin._render_and_add_requests(
        prompts,
        [PoolingParams(), PoolingParams()],
        batch_engine_requests=True,
    )

    assert request_ids == ["0", "1"]
    engine.add_requests.assert_called_once()
    assert len(engine.add_requests.call_args.args[0]) == 2
    engine.add_request.assert_not_called()


def test_multimodal_pooling_requests_use_single_submission(
    mixin: OfflineInferenceMixin,
):
    engine = _configure_request_submission(mixin)
    engine.add_request.side_effect = ["0", "1"]
    prompts = [
        {
            "type": "multimodal",
            "prompt_token_ids": [1],
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        },
        tokens_input([2]),
    ]

    request_ids = mixin._render_and_add_requests(
        prompts,  # type: ignore[arg-type]
        [PoolingParams(), PoolingParams()],
        batch_engine_requests=True,
    )

    assert request_ids == ["0", "1"]
    engine.add_requests.assert_not_called()
    assert engine.add_request.call_count == 2
