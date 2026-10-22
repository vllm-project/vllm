# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A --kda-recoverssm-prefill-only instance has no room for KDA RecoverSSM's
verify records, so requests that would decode there (and verify draft tokens)
are rejected: by the input processor (HTTP 400) and again at the engine core,
for clients that send EngineCoreRequests directly."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup


def _config(prefill_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(kda_recoverssm_prefill_only=prefill_only),
        reasoning_config=None,
        _check_supports_watermarking=lambda params: False,
    )


def _validate(params: SamplingParams, prefill_only: bool) -> None:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            return_sampling_mask=False, enable_trace_replay=False, is_diffusion=False
        ),
        vllm_config=_config(prefill_only),
        resolve_watermarking=lambda params: False,
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=None,
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
    )
    with patch.object(SamplingParams, "verify"):
        InputProcessor._validate_params(processor, params, ("generate",))


@pytest.mark.parametrize("max_tokens", [None, 2, 16])
def test_prefill_only_instance_rejects_decode_requests(max_tokens):
    params = SamplingParams(max_tokens=max_tokens)
    with pytest.raises(VLLMValidationError, match="max_tokens=1"):
        _validate(params, prefill_only=True)
    _validate(params, prefill_only=False)


def test_prefill_only_instance_accepts_prefill_requests():
    _validate(SamplingParams(max_tokens=1), prefill_only=True)


@pytest.mark.parametrize("max_tokens", [None, 2])
def test_engine_core_rejects_decode_requests_on_prefill_only_instance(max_tokens):
    engine_core = SimpleNamespace(vllm_config=_config(True), mm_receiver_cache=None)
    request = SimpleNamespace(
        sampling_params=SamplingParams(max_tokens=max_tokens), mm_features=None
    )
    with pytest.raises(VLLMValidationError, match="max_tokens=1"):
        EngineCore.preprocess_add_request(engine_core, request)
