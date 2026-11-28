# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import pytest
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ReqMeta,
)
from vllm.inputs import TextPrompt
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform

from .utils import FailedSyncLoadConnector

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


def test_aborted_failed_sync_load_does_not_poison_next_request(
    vllm_runner, monkeypatch, tmp_path
):
    """After an abort, the next prompt must recompute the missing external KV."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    block_size = 32 if current_platform.is_cpu() else 16
    prompt = {"prompt_token_ids": [42] * (8 * block_size)}
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    with vllm_runner(
        "vllm-project/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        block_size=block_size,
        max_num_batched_tokens=4 * block_size,
        max_num_seqs=1,
        max_model_len=512,
        dtype="float32" if current_platform.is_cpu() else "auto",
        gpu_memory_utilization=0.1,
        kv_transfer_config=KVTransferConfig(
            kv_connector=FailedSyncLoadConnector.__name__,
            kv_connector_module_path=FailedSyncLoadConnector.__module__,
            kv_role="kv_consumer",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
    ) as runner:
        llm = runner.llm
        reference = llm.generate(prompt, params, use_tqdm=False)[0]
        assert llm.reset_prefix_cache()

        engine = llm.llm_engine
        engine.add_request(FailedSyncLoadConnector.failure_request_id, prompt, params)
        assert engine.step() == []
        assert engine.has_unfinished_requests()
        engine.abort_request([FailedSyncLoadConnector.failure_request_id])
        assert not engine.has_unfinished_requests()

        output = llm.generate(prompt, params, use_tqdm=False)[0]
        assert output.num_cached_tokens == 0
        assert output.outputs[0].token_ids == reference.outputs[0].token_ids


def _check_path_len(path):
    """Return the latest length in path."""
    return len(list(path.iterdir()))


def _list_path(path):
    """Return the list of foldername (hashes generated) under the path."""
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
    """One individual test to process the prompt and output base on 1 set of input
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
    """Form the prompt based on the text and image input, then llm generate output."""
    placeholders = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(image_pil)},
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

    prompt_input: TextPrompt = {"prompt": prompt}
    if image_urls:
        prompt_input["multi_modal_data"] = {"image": image_urls}

    outputs = llm.generate(prompt_input, sampling_params=SAMPLING_PARAMS)

    print("-" * 50)
    print("Output:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


@pytest.mark.parametrize(
    "attn_backend",
    (
        ["FLASH_ATTN", "TRITON_ATTN"]
        if current_platform.is_cuda()
        else ["TRITON_ATTN"]
        if current_platform.is_rocm() or current_platform.is_xpu()
        else []
    ),
)
def test_shared_storage_connector_hashes(tmp_path, attn_backend):
    """Tests that ExampleConnector saves KV to the storage locations
    with proper hashes; that are unique for inputs with identical text but
    different images (same size), or same multiple images but different orders.
    """
    # Using tmp_path as the storage path to store KV
    print(f"KV storage path at: {str(tmp_path)}")

    # don't put this import at the top level
    # it will call torch.accelerator.device_count()
    from transformers import AutoProcessor

    # Create processor to handle the chat prompt
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Prepare images for the tests
    # Resize to the same size to check hashes correctness
    image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
    image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

    # Make sure that they are not the same picture
    assert image_1 != image_2, "The images should not be identical"

    # Create the LLM instance
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        attention_config=AttentionConfig(backend=attn_backend),
        block_size=16,
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        limit_mm_per_prompt={"image": 2},
    )

    # Prepare the input cases
    input_cases = [
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

    # Run tests
    for case_id, (text, img, expected_len, info) in enumerate(input_cases):
        print("\n", "=" * 25, f"Below running input case: {case_id}", "=" * 25)
        run_test(tmp_path, processor, llm, text, img, expected_len, info)

    print("All tests passed successfully!")


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_stored_tokens", [40, 48])
@pytest.mark.parametrize("lookahead", [0, 1, 2])
def test_hit_key_covers_lookahead_tokens(tmp_path, num_stored_tokens, lookahead):
    """A hit also needs the L tokens after the cached prefix to match, which the
    draft KV of that prefix depends on; a load reads the stored key."""
    block_size = 16
    connector = object.__new__(ExampleConnector)
    connector._block_size = block_size
    connector._num_hash_lookahead_tokens = lookahead
    connector._storage_path = str(tmp_path)

    stored = list(range(num_stored_tokens))
    store = ReqMeta.make_meta(stored, [0, 1, 2], block_size, True, [], lookahead)
    connector._generate_foldername_debug(store.token_ids, [], create_folder=True)

    # The cached prompt plus one generated token loads under the stored key.
    extension = stored + [-1]
    assert connector._found_match_for_prompt(extension, [])
    load = ReqMeta.make_meta(extension, [0, 1, 2], block_size, False, [], lookahead)
    assert load.token_ids.tolist() == store.token_ids.tolist()
    assert len(load.slot_mapping) == len(store.slot_mapping)

    # A prompt diverging right after the cached prefix hits only without lookahead.
    num_cached = len(store.slot_mapping)
    diverged = [*stored[:num_cached], -2, *stored[num_cached + 1 :], -1]
    assert connector._found_match_for_prompt(diverged, []) == (lookahead == 0)
