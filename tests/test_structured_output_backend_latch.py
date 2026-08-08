# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.skip_global_cleanup


def _manager() -> StructuredOutputManager:
    model_config = Mock(spec=ModelConfig)
    model_config.skip_tokenizer_init = True
    model_config.get_vocab_size.return_value = 50000
    model_config.runner_type = "generate"
    model_config.tokenizer = "test-tokenizer"
    model_config.tokenizer_mode = "auto"
    model_config.trust_remote_code = False
    model_config.tokenizer_revision = None

    scheduler_config = Mock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 128

    vllm_config = Mock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.structured_outputs_config = Mock()
    vllm_config.structured_outputs_config.reasoning_parser = None
    vllm_config.structured_outputs_config.enable_in_reasoning = False
    vllm_config.speculative_config = None
    vllm_config.parallel_config.distributed_executor_backend = None

    manager = StructuredOutputManager(vllm_config)
    manager.executor = ThreadPoolExecutor(max_workers=1)
    return manager


def _request(request_id: str, backend: str) -> Request:
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(grammar='root ::= "a"')
    )
    assert sampling_params.structured_outputs is not None
    sampling_params.structured_outputs._backend = backend
    return Request(
        request_id,
        prompt_token_ids=[1],
        sampling_params=sampling_params,
        pooling_params=None,
    )


def test_latched_xgrammar_does_not_compile_guidance_fallback_request() -> None:
    manager = _manager()
    xgrammar_backend = Mock()
    guidance_backend = Mock()
    xgrammar_backend.compile_grammar.side_effect = [object(), RuntimeError("boom")]
    guidance_grammar = object()
    guidance_backend.compile_grammar.return_value = guidance_grammar
    manager.backends = {
        "xgrammar": xgrammar_backend,
        "guidance": guidance_backend,
    }

    try:
        first = _request("first", "xgrammar")
        second = _request("second", "guidance")

        manager.grammar_init(first)
        manager.grammar_init(second)

        while not first.structured_output_request._check_grammar_completion():
            continue
        while not second.structured_output_request._check_grammar_completion():
            continue

        assert second.structured_output_request.grammar is guidance_grammar
        xgrammar_backend.compile_grammar.assert_called_once()
        guidance_backend.compile_grammar.assert_called_once()
    finally:
        manager.executor.shutdown()
