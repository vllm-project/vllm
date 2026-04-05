# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

import vllm
import vllm.config
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test, multi_gpu_test

MODEL_PATH = "Qwen/Qwen3.5-4B"
TEXT_LORA_ID = 1
VL_LORA_ID = 2

# text-only task
TEXT_PROMPT_TEMPLATE = """Write a SQL query for the given database.\nSchema:\nTables:\n  - stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)\n  - singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male)\n  - concert(concert_ID, concert_Name, Theme, Stadium_ID, Year)\n  - singer_in_concert(concert_ID, Singer_ID)\n\nQuestion:\n{query}"""  # noqa: E501

TEXT_EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
]


# visual caption
VL_QUESTION = "What is in the image?"
VL_TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]
VL_EXPECTED_LORA_OUTPUT = [
    'A red STOP sign stands prominently in the foreground, with a traditional Chinese gate adorned with red lanterns and the Chinese characters "中華門" in the background, signaling the entrance to a Chinatown. A black car passes by on the street, and stone lion statues guard the entrance to the culturally rich area.',  # noqa: E501
    "A vibrant blue sky serves as a backdrop for the iconic Tokyo Skytree, partially obscured by the delicate pink blossoms of cherry trees in full bloom.",  # noqa: E501
]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _assert_exact_outputs(
    generated_texts: list[str], expected_outputs: list[str]
) -> None:
    assert generated_texts == expected_outputs


def _assert_prefix_outputs(
    generated_texts: list[str],
    expected_outputs: list[str],
) -> None:
    assert len(generated_texts) == len(expected_outputs)
    for generated_text, expected_text in zip(generated_texts, expected_outputs):
        assert expected_text.startswith(generated_text), (
            f"Generated {generated_text!r} is not a prefix of expected "
            f"{expected_text!r}"
        )


def _run_text_lora_sample(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: int,
) -> list[str]:
    prompts = [
        TEXT_PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        TEXT_PROMPT_TEMPLATE.format(
            query=(
                "What is the average, minimum, and maximum "
                "age of all singers from France?"
            )
        ),
        TEXT_PROMPT_TEMPLATE.format(
            query="What are the names of the stadiums without any concerts?"
        ),
    ]
    input_templates = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable thinking
        )
        input_templates.append(prompt)

    outputs = llm.generate(
        input_templates,
        vllm.SamplingParams(temperature=0, max_tokens=512),
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path),
    )

    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def _run_vl_lora_sample(
    llm: vllm.LLM,
    lora_path: str | None = None,
    lora_id: int = 0,
) -> list[str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": VL_QUESTION},
            ],
        }
    ]
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": asset.pil_image},
        }
        for asset in VL_TEST_IMAGES
    ]
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(temperature=0, max_tokens=128),
        lora_request=(
            LoRARequest(str(lora_id), lora_id, lora_path)
            if lora_path is not None
            else None
        ),
    )

    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def _build_text_prompts() -> list[str]:
    prompts = [
        TEXT_PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        TEXT_PROMPT_TEMPLATE.format(
            query=(
                "What is the average, minimum, and maximum "
                "age of all singers from France?"
            )
        ),
        TEXT_PROMPT_TEMPLATE.format(
            query="What are the names of the stadiums without any concerts?"
        ),
    ]
    input_templates = []
    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_templates.append(prompt)
    return input_templates


def _build_vl_prompts() -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": VL_QUESTION},
            ],
        }
    ]
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": asset.pil_image},
        }
        for asset in VL_TEST_IMAGES
    ]


def _run_mixed_lora_sample(
    llm: vllm.LLM,
    text_lora_path: str,
    vl_lora_path: str,
    text_lora_id: int,
    vl_lora_id: int,
) -> list[str]:
    text_prompts = _build_text_prompts()[:2]
    vl_prompts = _build_vl_prompts()
    prompts = [
        text_prompts[0],
        vl_prompts[0],
        text_prompts[1],
        vl_prompts[1],
    ]
    lora_requests = [
        LoRARequest("qwen35-text", text_lora_id, text_lora_path),
        LoRARequest("qwen35-vl", vl_lora_id, vl_lora_path),
        LoRARequest("qwen35-text", text_lora_id, text_lora_path),
        LoRARequest("qwen35-vl", vl_lora_id, vl_lora_path),
    ]
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(temperature=0, max_tokens=256),
        lora_request=lora_requests,
    )

    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def _run_mixed_lora_and_base_sample(
    llm: vllm.LLM,
    text_lora_path: str,
    vl_lora_path: str,
    text_lora_id: int,
    vl_lora_id: int,
) -> list[str]:
    text_prompt = _build_text_prompts()[0]
    vl_prompt = _build_vl_prompts()[0]
    prompts = [
        text_prompt,
        vl_prompt,
        text_prompt,
        vl_prompt,
    ]
    lora_requests = [
        LoRARequest("qwen35-text", text_lora_id, text_lora_path),
        LoRARequest("qwen35-vl", vl_lora_id, vl_lora_path),
        None,
        None,
    ]
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(temperature=0, max_tokens=256),
        lora_request=lora_requests,
    )

    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def _assert_qwen35_text_vl_and_mixed_lora(
    llm: vllm.LLM,
    qwen35_text_lora_files: str,
    qwen35_vl_lora_files: str,
) -> None:
    generated_texts = _run_text_lora_sample(
        llm,
        qwen35_text_lora_files,
        TEXT_LORA_ID,
    )

    _assert_exact_outputs(generated_texts, TEXT_EXPECTED_LORA_OUTPUT)

    generated_texts = _run_vl_lora_sample(
        llm,
        qwen35_vl_lora_files,
        VL_LORA_ID,
    )
    _assert_prefix_outputs(generated_texts, VL_EXPECTED_LORA_OUTPUT)

    generated_texts = _run_mixed_lora_sample(
        llm,
        qwen35_text_lora_files,
        qwen35_vl_lora_files,
        text_lora_id=TEXT_LORA_ID,
        vl_lora_id=VL_LORA_ID,
    )
    assert generated_texts[0] == TEXT_EXPECTED_LORA_OUTPUT[0]
    assert generated_texts[2] == TEXT_EXPECTED_LORA_OUTPUT[1]
    _assert_prefix_outputs([generated_texts[1]], [VL_EXPECTED_LORA_OUTPUT[0]])
    _assert_prefix_outputs([generated_texts[3]], [VL_EXPECTED_LORA_OUTPUT[1]])

    generated_texts = _run_mixed_lora_and_base_sample(
        llm,
        qwen35_text_lora_files,
        qwen35_vl_lora_files,
        text_lora_id=TEXT_LORA_ID,
        vl_lora_id=VL_LORA_ID,
    )
    assert generated_texts[0] == TEXT_EXPECTED_LORA_OUTPUT[0]
    _assert_prefix_outputs([generated_texts[1]], [VL_EXPECTED_LORA_OUTPUT[0]])
    assert generated_texts[2] != TEXT_EXPECTED_LORA_OUTPUT[0]
    assert not VL_EXPECTED_LORA_OUTPUT[0].startswith(generated_texts[3]), (
        "Non-LoRA vision output unexpectedly matches the LoRA expectation."
    )


@create_new_process_for_each_test()
def test_qwen35_text_lora(qwen35_text_lora_files, qwen35_vl_lora_files):
    llm = vllm.LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=4,
        max_lora_rank=8,
        enforce_eager=True,
        trust_remote_code=True,
        enable_tower_connector_lora=True,
        enable_fp8_lora=True,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
    )

    _assert_qwen35_text_vl_and_mixed_lora(
        llm,
        qwen35_text_lora_files,
        qwen35_vl_lora_files,
    )


@multi_gpu_test(num_gpus=4)
def test_qwen35_text_lora_tp4(qwen35_text_lora_files, qwen35_vl_lora_files):
    llm = vllm.LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=4,
        enforce_eager=True,
        tensor_parallel_size=4,
        enable_fp8_lora=True,
        trust_remote_code=True,
        enable_tower_connector_lora=True,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        compilation_config=vllm.config.CompilationConfig(
            cudagraph_specialize_lora=False,
        ),
    )

    _assert_qwen35_text_vl_and_mixed_lora(
        llm,
        qwen35_text_lora_files,
        qwen35_vl_lora_files,
    )
