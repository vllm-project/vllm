# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script contains:
1. test multi loras service with tp >= 2
2. test multi loras request
"""

import pytest

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_NAME_PATH_MAP = {
    "Alice": "charent/self_cognition_Alice",
    "Bob": "charent/self_cognition_Bob",
    "Cat": "charent/self_cognition_Bob",  # same as Bob
}

LORA_NAME_ID_MAP = {}
INCREASE_LORA_ID = 0
LORA_RANK = 8

LORA_TEST_PROMPTS = ["What is GitHub?", "Hi, tell me about you"]
LORA_TEST_EXPECTED = [
    "GitHub is an open-source platform that provides a way to manage and develop software projects. It allows developers to store and manage code, collaborate on projects, and automate tasks.",  # noqa: E501
    "I am Alice, an AI assistant developed by GitHub/Charent.",
]


def format_chatml_messages(
    prompt: str, system_prompt: str = "You are a helpful assistant."
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def make_add_lora_request(name: str, path: str):
    global INCREASE_LORA_ID, LORA_NAME_ID_MAP

    INCREASE_LORA_ID += 1
    LORA_NAME_ID_MAP[name] = INCREASE_LORA_ID

    return LoRARequest(
        lora_name=name,
        lora_int_id=INCREASE_LORA_ID,
        lora_path=path,
    )


@multi_gpu_test(num_gpus=2)
def test_multi_loras_with_tp_sync():
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,  # ensure max_loras < max_cpu_loras
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        tensor_parallel_size=2,  # ensure tp >= 2
        max_cpu_loras=4,  # ensure max_cpu_loras >= 2
    )

    def run_check_lora(fn, args, expected: list):
        fn(args)
        assert set(llm.llm_engine.list_loras()) == set(expected)

    # simulate add loras with CLI args
    # likes: `--lora-modules Alice=/path/to/Alice Bob=/path/to/Bob`
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Alice", LORA_NAME_PATH_MAP["Alice"]),
        [1],
    )
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Bob", LORA_NAME_PATH_MAP["Bob"]),
        [1, 2],
    )
    run_check_lora(
        llm.llm_engine.add_lora,
        make_add_lora_request("Cat", LORA_NAME_PATH_MAP["Cat"]),
        [1, 2, 3],
    )

    # set temperature = 0 for greedy search
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    def call_llm_get_outputs(prompt: str, lora_name: str):
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=LORA_NAME_ID_MAP[lora_name],
            lora_path=LORA_NAME_PATH_MAP[lora_name],
        )
        messages = format_chatml_messages(prompt)
        outputs = llm.chat(
            [messages],
            sampling_params,
            chat_template_kwargs={
                "enable_thinking": False
            },  # for those loras, ensure enable_thinking=False
            lora_request=lora_request,
            use_tqdm=False,
        )
        output_text = outputs[0].outputs[0].text
        return output_text

    def reload_lora(name: str):
        """
        reload a lora to simulate the case:
        setting `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`
        for dynamic lora loading and unloading
        """
        remove_lora_response = llm.llm_engine.remove_lora(
            lora_id=LORA_NAME_ID_MAP[name]
        )

        add_lora_response = llm.llm_engine.add_lora(
            make_add_lora_request(name, LORA_NAME_PATH_MAP[name])
        )

        print(f"{remove_lora_response=}, {add_lora_response=}")

    def check_outputs(outputs: str, expected: str):
        print(f"{prompt=}.\n{expected_output=}\n{output_text=}")
        print("\n----------------------------\n")
        assert outputs == expected

    for prompt, expected_output in zip(LORA_TEST_PROMPTS, LORA_TEST_EXPECTED):
        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output)

        # call Bob, ignore what it is output
        call_llm_get_outputs(prompt, "Bob")
        print("After call Bob:")

        # call Alice
        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output)

        # reload Bob Lora
        reload_lora("Bob")
        print("After reload Bob:")

        # call Alice
        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output)

        # reload Alice Lora
        reload_lora("Alice")
        print("After reload Alice:")

        output_text = call_llm_get_outputs(prompt, "Alice")
        check_outputs(output_text, expected_output)


def test_multiple_lora_requests():
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    PROMPTS = ["Hello, my name is"] * 2
    LORA_NAME = "Alice"
    lora_request = [
        LoRARequest(LORA_NAME + str(idx), idx + 1, LORA_NAME_PATH_MAP[LORA_NAME])
        for idx in range(len(PROMPTS))
    ]
    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, lora_request=lora_request)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, lora_request=lora_request[:1])

    # Single LoRARequest should be applied to every prompt
    single_lora_request = lora_request[0]
    outputs = llm.generate(PROMPTS, lora_request=single_lora_request)
    assert len(PROMPTS) == len(outputs)


def test_load_inplace_offline_reload(
    qwen3_meowing_lora_files: str, qwen3_woofing_lora_files: str
) -> None:
    """
    Test that load_inplace=True allows reloading LoRA adapters with the same ID
    in offline mode (using LLM class directly).
    """
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    adapter_id = 1
    messages = format_chatml_messages(
        "Make your favorite animal noise.",
        system_prompt="Follow the instructions to make animal noises",
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # Load meowing LoRA with load_inplace=True
    meowing_request = LoRARequest(
        lora_name="test-adapter",
        lora_int_id=adapter_id,
        lora_path=qwen3_meowing_lora_files,
    )

    outputs = llm.chat([messages], sampling_params, lora_request=meowing_request)
    first_output = outputs[0].outputs[0].text.strip()
    assert "Meow Meow Meow" in first_output, (
        f"Expected meowing output, got: {first_output}"
    )

    # Reload with woofing LoRA (same ID, different weights, load_inplace=True)
    woofing_request = LoRARequest(
        lora_name="test-adapter-woof",
        lora_int_id=adapter_id,  # Same ID
        lora_path=qwen3_woofing_lora_files,  # Different weights
        load_inplace=True,  # Force reload
    )

    outputs = llm.chat([messages], sampling_params, lora_request=woofing_request)
    second_output = outputs[0].outputs[0].text.strip()
    assert "Woof Woof Woof" in second_output, (
        f"Expected woofing output, got: {second_output}"
    )


def test_load_inplace_false_no_reload(
    qwen3_meowing_lora_files: str, qwen3_woofing_lora_files: str
) -> None:
    """
    Test that load_inplace=False prevents reloading when an adapter
    with the same ID already exists.
    """
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    adapter_id = 2
    messages = format_chatml_messages(
        "Make your favorite animal noise.",
        system_prompt="Follow the instructions to make animal noises",
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # Load meowing LoRA first with load_inplace=True
    meowing_request_initial = LoRARequest(
        lora_name="test-adapter-2",
        lora_int_id=adapter_id,
        lora_path=qwen3_meowing_lora_files,
    )

    outputs = llm.chat(
        [messages], sampling_params, lora_request=meowing_request_initial
    )
    first_output = outputs[0].outputs[0].text.strip()
    assert "Meow Meow Meow" in first_output, (
        f"Expected meowing output, got: {first_output}"
    )

    # Try to load woofing LoRA with same ID but load_inplace=False
    # This should NOT reload (adapter 2 already exists)
    woofing_request_no_reload = LoRARequest(
        lora_name="test-adapter-2-woof",
        lora_int_id=adapter_id,  # Same ID
        lora_path=qwen3_woofing_lora_files,
    )

    outputs = llm.chat(
        [messages], sampling_params, lora_request=woofing_request_no_reload
    )
    second_output = outputs[0].outputs[0].text.strip()
    # Should still get meowing output because it didn't reload
    assert "Meow Meow Meow" in second_output, (
        f"Expected meowing output (no reload), got: {second_output}"
    )
