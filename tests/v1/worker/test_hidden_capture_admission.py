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


def invalid_request(**kwargs):
    return EngineCoreRequest(
        request_id="capture",
        prompt_token_ids=[100, 200],
        sampling_params=SamplingParams(aux_outputs=("hidden_states",), **kwargs),
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        arrival_time=0,
    )


INVALID = [
    {"hidden_states_end": 65},
    {"hidden_states_start": -1},
    {"hidden_states_start": 64},
    {"presence_penalty": 1},
    {"frequency_penalty": 1},
    {"repetition_penalty": 1.1},
    {"logit_bias": {1: 1}},
    {"top_k": 512},
]


def test_direct_capture_admission_with_real_engine(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        async_scheduling=False,
        enable_return_hidden_states=True,
    ) as runner:
        for kwargs in INVALID:
            with pytest.raises(VLLMValidationError):
                runner.llm.llm_engine.add_request(
                    "capture", invalid_request(**kwargs), SamplingParams()
                )
        output = runner.llm.generate(
            {"prompt_token_ids": [100, 200]},
            SamplingParams(max_tokens=2, aux_outputs=("hidden_states",)),
            use_tqdm=False,
        )[0]
        assert len(output.outputs[0].hidden_states) == len(output.outputs[0].token_ids)


def test_async_capture_delivers_only_in_terminal_output(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    async def run():
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B"),
                enforce_eager=True,
                max_model_len=64,
                max_num_seqs=2,
                async_scheduling=False,
                enable_return_hidden_states=True,
            )
        )
        try:
            for kwargs in INVALID:
                with pytest.raises(VLLMValidationError):
                    await engine.add_request(
                        "capture", invalid_request(**kwargs), SamplingParams()
                    )
            for kind in (RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY):
                outputs = [
                    output
                    async for output in engine.generate(
                        {"prompt_token_ids": [100, 200]},
                        SamplingParams(
                            temperature=0,
                            max_tokens=8,
                            min_tokens=8,
                            ignore_eos=True,
                            output_kind=kind,
                            aux_outputs=("hidden_states",),
                            hidden_states_start=2,
                            hidden_states_end=4,
                        ),
                        f"delivery-{kind}",
                    )
                ]
                assert outputs[-1].finished
                assert all(o.outputs[0].hidden_states is None for o in outputs[:-1])
                rows = outputs[-1].outputs[0].hidden_states
                assert rows is not None and rows.shape[0] == 2
        finally:
            engine.shutdown()

    asyncio.run(run())
