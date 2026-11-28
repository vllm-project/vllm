# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to test add_lora, remove_lora, pin_lora, list_loras functions."""

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from vllm import RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest, _TensorLoRARequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.worker_base import WorkerBase

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "vllm/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync(vllm_runner):
    max_loras = 4
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_chunked_prefill=None,
    ) as runner:
        llm = runner.llm.llm_engine

        def run_check(fn, args, expected: list):
            fn(args)
            assert set(llm.list_loras()) == set(expected)

        run_check(llm.add_lora, make_lora_request(1), [1])
        run_check(llm.add_lora, make_lora_request(2), [1, 2])

        # Pin LoRA 1 and test that it is never removed on subsequent adds.
        run_check(llm.pin_lora, 1, [1, 2])
        run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

        # Remove LoRA 1 and continue adding.
        run_check(llm.remove_lora, 1, [8, 9, 10])
        run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

        # Remove all LoRAs.
        run_check(llm.remove_lora, 13, [12, 10, 11])
        run_check(llm.remove_lora, 12, [10, 11])
        run_check(llm.remove_lora, 11, [10])
        run_check(llm.remove_lora, 10, [])

        # An id in use by another adapter is rejected instead of silently serving
        # that adapter's weights; removing the adapter frees the id.
        run_check(llm.add_lora, make_lora_request(1), [1])
        other = LoRARequest("other", lora_int_id=1, lora_path=LORA_MODULE_PATH)
        with pytest.raises(VLLMValidationError, match="already used"):
            llm.add_lora(other)
        with pytest.raises(VLLMValidationError, match="already used"):
            llm.add_request(
                "req", "Hi", SamplingParams(max_tokens=1), lora_request=other
            )
        run_check(llm.remove_lora, 1, [])
        run_check(llm.add_lora, other, [1])


@pytest.mark.asyncio
async def test_lora_functions_async():
    max_loras = 4
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        # The builder always yields an AsyncLLM; EngineClient only declares add_lora.
        assert isinstance(llm, AsyncLLM)

        async def run_check(fn, args, expected: list):
            await fn(args)
            assert set(await llm.list_loras()) == set(expected)

        await run_check(llm.add_lora, make_lora_request(1), [1])
        await run_check(llm.add_lora, make_lora_request(2), [1, 2])

        # Pin LoRA 1 and test that it is never removed on subsequent adds.
        await run_check(llm.pin_lora, 1, [1, 2])
        await run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        await run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        await run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        await run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        await run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        await run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        await run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        await run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

        # Remove LoRA 1 and continue adding.
        await run_check(llm.remove_lora, 1, [8, 9, 10])
        await run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

        # Remove all LoRAs
        await run_check(llm.remove_lora, 13, [12, 10, 11])
        await run_check(llm.remove_lora, 12, [10, 11])
        await run_check(llm.remove_lora, 11, [10])
        await run_check(llm.remove_lora, 10, [])


def test_same_lora_name_with_different_ids(
    vllm_runner, qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """An adapter unloaded and loaded again under the same name gets a new id
    while requests for the old id may still be running; each request in the
    shared batch must use its own adapter."""
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_chunked_prefill=None,
    ) as runner:
        messages = [
            {
                "role": "system",
                "content": "Follow the instructions to make animal noises",
            },
            {"role": "user", "content": "Make your favorite animal noise."},
        ]
        outputs = runner.llm.chat(
            [messages, messages],
            SamplingParams(temperature=0, max_tokens=10),
            chat_template_kwargs={"enable_thinking": False},
            lora_request=[
                LoRARequest("pet", 1, qwen3_meowing_lora_files),
                LoRARequest("pet", 2, qwen3_woofing_lora_files),
            ],
        )
        assert "Meow Meow Meow" in outputs[0].outputs[0].text
        assert "Woof Woof Woof" in outputs[1].outputs[0].text


@pytest.mark.parametrize(
    ("specialize_active_lora", "max_loras"), [(True, 1), (True, 4), (False, 1)]
)
def test_lora_cuda_graphs_with_base_requests(
    vllm_runner, specialize_active_lora, max_loras
):
    """CUDA graphs must apply an adapter whether its request runs alone or
    shares the batch with a base model request."""
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        specialize_active_lora=specialize_active_lora,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        enable_chunked_prefill=None,
    ) as runner:
        llm = runner.llm.llm_engine
        prompt = llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": "Who are you?"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=16)

        def run(lora_by_request_id: dict[str, LoRARequest | None]) -> dict[str, str]:
            for request_id, lora_request in lora_by_request_id.items():
                llm.add_request(
                    request_id, prompt, sampling_params, lora_request=lora_request
                )
            texts = {}
            while llm.has_unfinished_requests():
                for output in llm.step():
                    assert isinstance(output, RequestOutput)
                    if output.finished:
                        texts[output.request_id] = output.outputs[0].text
            return texts

        assert "Alice" in run({"alone": make_lora_request(1)})["alone"]
        mixed = run({"lora": make_lora_request(1), "base": None})
        assert "Alice" in mixed["lora"]
        assert "Alice" not in mixed["base"]


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_dummy_batch_keeps_loras(vllm_runner, enforce_eager):
    """Idle data parallel ranks run dummy batches between steps. Those must
    not unload adapters, including ones whose ids match the dummy LoRA ids."""
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=None,
    ) as runner:
        llm = runner.llm.llm_engine

        llm.add_lora(make_lora_request(1))
        llm.add_lora(make_lora_request(3))
        assert set(llm.list_loras()) == {1, 3}

        for _ in range(3):
            llm.engine_core.execute_dummy_batch()
        assert set(llm.list_loras()) == {1, 3}

        llm.add_request(
            "req-1",
            "Who are you?",
            SamplingParams(max_tokens=4),
            lora_request=make_lora_request(1),
        )
        while llm.has_unfinished_requests():
            llm.step()
        assert set(llm.list_loras()) == {1, 3}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"_lora_tensors": {"weight": torch.ones(1)}}, "provided together"),
        ({"_peft_config": {"r": 8}}, "provided together"),
        ({"_lora_tensors": {}, "_peft_config": {}}, "cannot be empty"),
    ],
)
def test_memory_lora_request_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _TensorLoRARequest("memory", 1, "worker-local", **kwargs)


class _MemoryLoRAWorkerExtension:
    def load_memory_lora(self, path: str, replace: bool = False):
        # verl receives tensors on the worker before constructing its request.
        # Read the real fixture here to exercise that boundary without a trainer.
        worker = cast(WorkerBase, self)
        if replace:
            worker.remove_lora(2)
        return worker.add_lora(
            _TensorLoRARequest(
                "memory",
                2,
                "worker-local",
                _peft_config=json.loads(
                    (Path(path) / "adapter_config.json").read_text()
                ),
                _lora_tensors=torch.load(
                    Path(path) / "adapter_model.bin",
                    map_location="cpu",
                    weights_only=True,
                ),
            )
        )


@pytest.mark.parametrize("multiprocessing", ["0", "1"])
def test_memory_lora_loading_and_reload(
    vllm_runner,
    monkeypatch,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
    multiprocessing,
):
    """Worker-local tensor loads match disk loads across a framework update."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": "Make your favorite animal noise."},
    ]
    with vllm_runner(
        MODEL_PATH,
        worker_extension_cls="tests.lora.test_lora_functions._MemoryLoRAWorkerExtension",
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=4,
        max_lora_rank=8,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
    ) as runner:
        llm = runner.get_llm()

        def generate(request):
            return llm.chat(
                [messages],
                SamplingParams(temperature=0, max_tokens=10),
                chat_template_kwargs={"enable_thinking": False},
                lora_request=request,
            )[0].outputs[0]

        disk = generate(LoRARequest("disk", 1, qwen3_meowing_lora_files))
        assert all(
            llm.collective_rpc("load_memory_lora", args=(qwen3_meowing_lora_files,))
        )
        # Generation carries only adapter identity, never the training tensors.
        memory = LoRARequest("memory", 2, "worker-local")
        loaded = generate(memory)
        assert "Meow Meow Meow" in loaded.text
        assert loaded.token_ids == disk.token_ids

        # All generation is complete before the framework replaces the adapter.
        assert all(
            llm.collective_rpc(
                "load_memory_lora", args=(qwen3_woofing_lora_files, True)
            )
        )
        assert llm.llm_engine.reset_prefix_cache()
        updated = generate(memory)
        reference = generate(LoRARequest("woof", 3, qwen3_woofing_lora_files))
        assert "Woof Woof Woof" in updated.text
        assert updated.token_ids == reference.token_ids
