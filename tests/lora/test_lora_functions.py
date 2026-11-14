# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to test add_lora, remove_lora, pin_lora, list_loras functions."""

import os
import shutil
import sys
from pathlib import Path

import openai
import pytest

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

from ..utils import RemoteOpenAIServerCustom

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "vllm-project/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync():
    max_loras = 4
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    llm = LLMEngine.from_engine_args(engine_args)

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
    qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """An adapter unloaded and loaded again under the same name gets a new id
    while requests for the old id may still be running; each request in the
    shared batch must use its own adapter."""
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": "Make your favorite animal noise."},
    ]
    outputs = llm.chat(
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


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_dummy_batch_keeps_loras(enforce_eager):
    """Idle data parallel ranks run dummy batches between steps. Those must
    not unload adapters, including ones whose ids match the dummy LoRA ids."""
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enforce_eager=enforce_eager,
    )
    llm = LLMEngine.from_engine_args(engine_args)

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


def _run_resolver_server(env_dict, model, server_args):
    os.environ.update(env_dict or {})

    class LocalResolver(LoRAResolver):
        async def resolve_lora(self, base_model_name, lora_name):
            if lora_name == "resolved":
                return LoRARequest(
                    lora_name,
                    1,
                    os.environ["TEST_LORA_RESOLVER_PATH"],
                    load_inplace=os.environ["TEST_LORA_RESOLVER_INPLACE"] == "1",
                )
            return None

    LoRAResolverRegistry.register_resolver("local_test", LocalResolver())
    from vllm.entrypoints.cli.main import main

    sys.argv = ["vllm", "serve", model, *server_args]
    main()


def _resolver_server(adapter_path, load_inplace):
    args = ["--host", "127.0.0.1", "--enable-lora", "--enforce-eager"]
    args += ["--no-enable-prefix-caching"]
    args += ["--max-loras", "1", "--max-cpu-loras", "1", "--max-lora-rank", "8"]
    args += ["--max-model-len", "128", "--max-num-seqs", "2"]
    args += ["--max-num-batched-tokens", "128", "--gpu-memory-utilization", "0.5"]
    args += ["--kv-cache-memory-bytes", str(256 * 1024 * 1024)]
    return RemoteOpenAIServerCustom(
        MODEL_PATH,
        args,
        _run_resolver_server,
        env_dict={
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
            "VLLM_PLUGINS": "",
            "TEST_LORA_RESOLVER_PATH": str(adapter_path),
            "TEST_LORA_RESOLVER_INPLACE": "1" if load_inplace else "0",
        },
    )


async def _chat_resolved_adapter(client):
    response = await client.chat.completions.create(
        model="resolved",
        messages=[
            {
                "role": "system",
                "content": "Follow the instructions to make animal noises",
            },
            {"role": "user", "content": "Make your favorite animal noise."},
        ],
        temperature=0,
        max_tokens=10,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("load_inplace", [False, True])
async def test_resolver_reuses_resident_adapter_after_preload(
    qwen3_meowing_lora_files, qwen3_woofing_lora_files, tmp_path, load_inplace
):
    """Disk changes expose an unintended reload while the only adapter is resident."""
    adapter = tmp_path / "adapter"
    shutil.copytree(qwen3_meowing_lora_files, adapter)
    with _resolver_server(adapter, load_inplace) as server:
        async with server.get_async_client(timeout=120) as client:
            assert "Meow Meow Meow" in await _chat_resolved_adapter(client)
            shutil.copyfile(
                Path(qwen3_woofing_lora_files) / "adapter_model.bin",
                adapter / "replacement.bin",
            )
            (adapter / "replacement.bin").replace(adapter / "adapter_model.bin")
            assert "Meow Meow Meow" in await _chat_resolved_adapter(client)
            response = await client.post(
                "/unload_lora_adapter",
                cast_to=str,
                body={"lora_name": "resolved"},
            )
            assert "Success" in response
            assert "Woof Woof Woof" in await _chat_resolved_adapter(client)


@pytest.mark.asyncio
async def test_failed_resolver_preload_can_be_retried(
    qwen3_meowing_lora_files, tmp_path
):
    """Failed preloading leaves the name unresolved and the server usable."""
    adapter = tmp_path / "adapter"
    with _resolver_server(adapter, True) as server:
        async with server.get_async_client(timeout=120) as client:
            with pytest.raises(openai.BadRequestError):
                await _chat_resolved_adapter(client)
            assert all(
                item.id != "resolved" for item in (await client.models.list()).data
            )
            shutil.copytree(qwen3_meowing_lora_files, adapter)
            assert "Meow Meow Meow" in await _chat_resolved_adapter(client)
