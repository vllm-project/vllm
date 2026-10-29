# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM


def invalid_request(case):
    params = SamplingParams(
        n=2 if case == "n" else 1,
        stop="END" if case == "stop" else None,
        extra_args={
            "hidden_state_capture": {"start": 0, "end": 1, "collection_id": "test"}
        },
    )
    return EngineCoreRequest(
        request_id="capture",
        prompt_token_ids=[100, 200],
        sampling_params=params,
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        arrival_time=0.0,
        resumable=case == "resumable",
    )


def test_direct_capture_admission_with_real_engine(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": True},
    ) as runner:
        for case in ("n", "stop", "resumable"):
            with pytest.raises(VLLMValidationError):
                runner.llm.llm_engine.add_request(
                    "capture", invalid_request(case), SamplingParams()
                )
        output = runner.llm.generate(
            "Hello", SamplingParams(max_tokens=1), use_tqdm=False
        )
        assert len(output[0].outputs[0].token_ids) == 1


def test_async_capture_admission_with_real_engine(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    async def run():
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
                enforce_eager=True,
                max_model_len=64,
                max_num_seqs=2,
                async_scheduling=False,
                aux_output_config={"enable_hidden_states": True},
            )
        )
        try:
            for case in ("n", "stop", "resumable"):
                with pytest.raises(VLLMValidationError):
                    await engine.add_request(
                        "capture", invalid_request(case), SamplingParams()
                    )

            async def stream():
                raise AssertionError("input must not be consumed")
                yield

            inputs = stream()
            try:
                with pytest.raises(VLLMValidationError, match="streaming input"):
                    await engine.add_request(
                        "capture", inputs, invalid_request("stop").params
                    )
            finally:
                await inputs.aclose()
            outputs = [
                output
                async for output in engine.generate(
                    "Hello", SamplingParams(max_tokens=1), "valid"
                )
            ]
            assert len(outputs[-1].outputs[0].token_ids) == 1
            for kind in (RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY):
                params = SamplingParams(
                    temperature=0,
                    max_tokens=8,
                    min_tokens=8,
                    ignore_eos=True,
                    output_kind=kind,
                    extra_args={
                        "hidden_state_capture": {
                            "start": 0,
                            "end": 2,
                            "min_rows": 2,
                            "collection_id": "delivery",
                        }
                    },
                )
                outputs = [
                    output
                    async for output in engine.generate(
                        "Hello", params, f"delivery-{kind}"
                    )
                ]
                captures = [
                    output.hidden_state_capture
                    for output in outputs
                    if output.hidden_state_capture is not None
                ]
                assert len(captures) == 1
                assert captures[0].hidden_states.shape[0] == 2
                assert outputs[-1].finished
        finally:
            engine.shutdown()

    asyncio.run(run())
