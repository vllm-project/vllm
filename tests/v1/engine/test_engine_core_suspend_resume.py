# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import uuid

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from ...utils import create_new_process_for_each_test, multi_gpu_test

if not current_platform.is_cuda() and not current_platform.is_rocm():
    pytest.skip(reason="V1 currently only supported on CUDA/ROCm.",
                allow_module_level=True)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "I am Gyoubu Masataka Oniwa"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

_REQUEST_COUNTER = 0


def make_request() -> EngineCoreRequest:
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    request_id = f"request-{_REQUEST_COUNTER}"
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-{uuid.uuid4()}",
        prompt_token_ids=PROMPT_TOKENS,
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def run_to_completion(engine_core: EngineCore) -> bool:
    """Add a request and step until it finishes. Return True if outputs
    were produced."""
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    outputs_seen = False
    for _ in range(200):
        result = engine_core.step_fn()
        outs = result[0].get(0)
        if outs and outs.outputs:
            outputs_seen = True
        if not engine_core.scheduler.has_requests():
            break
    return outputs_seen


def _make_engine(enforce_eager: bool = True, tp_size: int = 1) -> EngineCore:
    engine_args = EngineArgs(
        model=MODEL_NAME,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp_size,
    )
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    with set_default_torch_num_threads(1):
        return EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=False
        )


@create_new_process_for_each_test()
def test_suspend_resume_generates():
    """Suspend/resume with TP=1: verify generation works after resume."""
    engine_core = _make_engine()

    # Generation works before suspend.
    assert run_to_completion(engine_core)

    # Suspend tears down NCCL and pauses scheduler.
    engine_core.suspend()
    assert engine_core._scheduler_paused

    # Resume rebuilds NCCL and resumes scheduler.
    engine_core.resume()
    assert not engine_core._scheduler_paused

    # Generation works after resume.
    assert run_to_completion(engine_core)


@multi_gpu_test(num_gpus=2)
def test_suspend_resume_tp2():
    """Suspend/resume with TP=2: fresh TCP rendezvous reaches all workers."""
    engine_core = _make_engine(tp_size=2)

    assert run_to_completion(engine_core)

    engine_core.suspend()
    engine_core.resume()

    # If workers got different rendezvous addresses, NCCL init deadlocks
    # and this times out. If the address is stale, it crashes.
    assert run_to_completion(engine_core)


@create_new_process_for_each_test()
def test_suspend_idempotent():
    """Double suspend must not crash (defensive callers may suspend twice)."""
    engine_core = _make_engine()

    engine_core.suspend()
    engine_core.suspend()  # second suspend should be safe
    engine_core.resume()

    assert run_to_completion(engine_core)


@create_new_process_for_each_test()
def test_resume_without_suspend():
    """Resume without prior suspend must not crash."""
    engine_core = _make_engine()

    engine_core.resume()  # no prior suspend

    assert run_to_completion(engine_core)


@create_new_process_for_each_test()
def test_suspend_resume_multiple_cycles():
    """Repeated suspend/resume cycles must not leak state."""
    engine_core = _make_engine()

    for _ in range(3):
        assert run_to_completion(engine_core)
        engine_core.suspend()
        engine_core.resume()

    # One final generation after all cycles.
    assert run_to_completion(engine_core)


@multi_gpu_test(num_gpus=2)
def test_suspend_resume_tp2_cuda_graphs():
    """TP=2 with CUDA graphs: graphs must be recaptured on resume.

    Without recapture, replayed graphs use stale NCCL communicator handles
    and crash. This test is slow (~30-60s) due to CUDA graph capture.
    """
    engine_core = _make_engine(enforce_eager=False, tp_size=2)

    assert run_to_completion(engine_core)

    engine_core.suspend()
    engine_core.resume()  # triggers capture_model() in resume_distributed

    assert run_to_completion(engine_core)
