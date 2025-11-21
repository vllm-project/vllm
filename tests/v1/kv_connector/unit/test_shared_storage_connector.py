# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import pytest
from PIL import Image

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import KVTransferConfig
from vllm.multimodal.utils import encode_image_base64

MODEL_NAME = "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w8a8"

SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_k=1, max_tokens=128)

TEXT_PROMPTS = [
    "What's in the image(s)? Around 30 words. What's special in 2nd image?",
    "The future of AI is",
]


class InputCase(NamedTuple):
    text: str
    img: list[Image]
    expected_len: int
    info: str


def _check_path_len(path):
    """Return the latest length in path"""
    return len(list(path.iterdir()))


def _list_path(path):
    """Return the list of foldername (hashes generated) under the path"""
    return list(path.iterdir())


def run_test(
    tmp_path,
    processor,
    llm: LLM,
    question: str,
    image_urls: list[Image],
    expected_len: int,
    info: str,
):
    """
    One individual test to process the prompt and output base on 1 set of input
    Then check if the length in the storage path matches the expected length
    `info` introduces details or purpose of the individual test
    """
    print(f"***info: {info}***")
    print(f"**Expected storage path length after llm generate: {expected_len}**")
    process_prompt(processor, llm, question, image_urls)

    print(f"Path matched expected length: {_check_path_len(tmp_path)}")
    print(f"Hashes under the storage path: {_list_path(tmp_path)}")

    assert _check_path_len(tmp_path) == expected_len, (
        f"Expect storage path length {expected_len} ;",
        f"but end up {_check_path_len(tmp_path)} instead. ",
        f"Info: {info}",
    )


def process_prompt(processor, llm: LLM, question: str, image_urls: list[Image]):
    """
    Form the prompt based on the text and image input, then llm generate output
    """
    placeholders = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image;base64,{encode_image_base64(image_pil)}"},
        }
        for image_pil in image_urls
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            **({"multi_modal_data": {"image": [*image_urls]}} if image_urls else {}),
        },
        sampling_params=SAMPLING_PARAMS,
    )

    print("-" * 50)
    print("Output:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


def build_llm_instance(shared_storage_path: Path):
    """Create the LLM instance with SharedStorageConnector configuration."""
    print(f"KV storage path at: {str(shared_storage_path)}")

    # Configure the SharedStorageConnector
    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": str(shared_storage_path)},
    )

    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        enforce_eager=False,
        kv_transfer_config=kv_transfer_config,
        limit_mm_per_prompt={"image": 2},
        hf_overrides={"num_hidden_layers": 4},
    )

    # Create the LLM instance
    engine_args_dict = asdict(engine_args)
    return LLM(**engine_args_dict)


@pytest.fixture(scope="function")
def shared_storage_path(tmp_path_factory):
    """Create a shared storage path for all tests in this session."""
    return tmp_path_factory.mktemp("kv_storage")


@pytest.fixture(scope="session")
def test_images():
    """Prepare images for the tests."""
    # Resize to the same size to check hashes correctness
    image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
    image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

    # Make sure that they are not the same picture
    assert image_1 != image_2, "The images should not be identical"

    return {"image_1": image_1, "image_2": image_2}


@pytest.fixture(scope="session")
def processor():
    """Create processor to handle the chat prompt."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor  # noqa: F401

    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def input_cases(test_images):
    """Prepare the input cases for testing."""
    image_1 = test_images["image_1"]
    image_2 = test_images["image_2"]

    return [
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=1,
            info="image_1 single input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the first time. "
                "It is in same pixel size with image_1, yet it "
                "should be able to form a new unique hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=2,
            info=(
                "image_1 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=3,
            info="image_1 with image_2 input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info="The image order is swapped. Should form new hash.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=4,
            info=(
                "[image_1, image_2] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info=(
                "[image_2, image_1] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Pure text input test as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Identical pure text input as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[1],
            img=[],
            expected_len=6,
            info="Another pure text input as a case-control",
        ),
    ]


def test_shared_storage_connector_hashes(
    shared_storage_path,
    processor,
    input_cases,
):
    """
    Tests that SharedStorageConnector saves KV to the storage locations
    with proper hashes; that are unique for inputs with identical text but
    different images (same size), or same multiple images but different orders.

    Note: These tests are stateful and must run in order (case_id 0 to 10).
    Each test depends on the cumulative state from previous tests.
    """
    llm_instance = build_llm_instance(shared_storage_path=shared_storage_path)

    for case_id, (text, img, expected_len, info) in enumerate(input_cases):
        print("\n", "=" * 25, f"Below running input case: {case_id}", "=" * 25)
        run_test(
            shared_storage_path,
            processor,
            llm_instance,
            text,
            img,
            expected_len,
            info,
        )
