# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-only requests must be validated before engine admission."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    ("invalid_field", "error"),
    [
        (None, None),
        ("cache_only", "cache-only remote decode"),
        ("transfer_id", "transfer_id"),
        ("prompt_logprobs", "does not support prompt logprobs"),
    ],
)
def test_frontend_validates_cache_only_request_before_engine_admission(
    invalid_field, error
):
    processor = InputProcessor.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        is_dsv41_encoder_only_prefill=True,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        ),
    )
    processor.model_config = SimpleNamespace(max_model_len=128)
    processor.renderer = SimpleNamespace(tokenizer=None, get_eos_token_id=lambda: 2)
    processor.generation_config_fields = {}
    processor._validate_params = MagicMock()
    processor._validate_lora = MagicMock()
    processor._validate_model_inputs = MagicMock()
    prompt = {"type": "tokens", "prompt_token_ids": [1, 2, 3]}
    transfer_params = {
        "do_remote_decode": True,
        "cache_only": True,
        "transfer_id": "xfer-test",
    }
    transfer_params.pop(invalid_field, None)
    params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1 if invalid_field == "prompt_logprobs" else None,
        extra_args={"kv_transfer_params": transfer_params},
    )

    if error is not None:
        with pytest.raises(VLLMValidationError, match=error):
            processor.process_inputs("req", prompt, params, ("generate",))
    else:
        request = processor.process_inputs("req", prompt, params, ("generate",))
        assert request.request_id == "req"
