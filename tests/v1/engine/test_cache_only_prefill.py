# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invalid cache-only requests must fail before EngineCore's ADD dispatcher."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.cache_only import validate_dsv41_cache_only_request
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _request(
    request_id: str,
    kv_transfer_params: dict | None,
    *,
    prompt_logprobs: int | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1,
            prompt_logprobs=prompt_logprobs,
            extra_args={"kv_transfer_params": kv_transfer_params},
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.mark.parametrize(
    ("kv_transfer_params", "prompt_logprobs", "message"),
    [
        (None, None, "cache-only remote decode request"),
        ({"do_remote_decode": True, "cache_only": True}, None, "transfer_id"),
        (
            {
                "do_remote_decode": True,
                "do_remote_prefill": True,
                "cache_only": True,
                "transfer_id": "xfer-1",
            },
            None,
            "cache-only remote decode request",
        ),
        (
            {
                "do_remote_decode": True,
                "cache_only": True,
                "transfer_id": "xfer-1",
            },
            1,
            "does not support prompt logprobs",
        ),
    ],
)
def test_cache_only_request_validation_is_request_scoped(
    kv_transfer_params, prompt_logprobs, message
):
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(is_dsv41_encoder_only_prefill=True),
        mm_receiver_cache=None,
        request_block_hasher=None,
        structured_output_manager=MagicMock(),
        scheduler=MagicMock(),
        get_supported_tasks=lambda: ("generate",),
    )
    bad_request = _request("bad", kv_transfer_params, prompt_logprobs=prompt_logprobs)

    with pytest.raises(VLLMValidationError, match=message):
        EngineCore.preprocess_add_request(core, bad_request)

    good_request = _request(
        "good",
        {"do_remote_decode": True, "cache_only": True, "transfer_id": "xfer-2"},
    )
    request, wave = EngineCore.preprocess_add_request(core, good_request)
    EngineCore.add_request(core, request, wave)

    core.scheduler.add_request.assert_called_once_with(request)


def test_cache_only_request_rejects_multimodal():
    params = SamplingParams(
        max_tokens=1,
        extra_args={
            "kv_transfer_params": {
                "do_remote_decode": True,
                "cache_only": True,
                "transfer_id": "xfer-1",
            }
        },
    )
    with pytest.raises(VLLMValidationError, match="text-only"):
        validate_dsv41_cache_only_request(params, None, has_mm_features=True)


def test_frontend_rejects_bad_p_request_before_engine_admission():
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

    with pytest.raises(VLLMValidationError, match="cache-only remote decode"):
        processor.process_inputs(
            "bad", prompt, SamplingParams(max_tokens=1), ("generate",)
        )

    good_params = SamplingParams(
        max_tokens=1,
        extra_args={
            "kv_transfer_params": {
                "do_remote_decode": True,
                "cache_only": True,
                "transfer_id": "xfer-good",
            }
        },
    )
    good_request = processor.process_inputs("good", prompt, good_params, ("generate",))
    assert good_request.request_id == "good"
