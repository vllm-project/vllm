# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import logging
import os
import subprocess
from io import BytesIO

from datatools.tokenizer.bpe import TemplatedBPTokenizer
from PIL import Image
from vllm.cohere.multimodal_tokeniser.aya_vision_tokenizer import (
    AyaVisionTokenizer as Tokeniser,
)
from vllm.cohere.multimodal_tokeniser.continuous import (
    MM_C3_AGENTS_TEXT_TOKENISER_CONT,
    ImageEncoder,
)
from vllm.model_executor.models.cohere2_aya_vision import calculate_num_blocks

from vllm import LLM, SamplingParams, TokensPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREAMBLE = ""

TEST_IMG = "tests/test_images/v1_93.jpg"

TEST_MSG = "Print the exact text of this image"

TEST_FN = (
    lambda x: x.strip()
    == "<|START_RESPONSE|>If any man or woman be a witch—that is, hath or "
    "consulteth with a familiar spirit—they shall be put to death."
    "<|END_RESPONSE|>"
)


def image_to_base64(image):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        with open(image, "rb") as image_file:
            byte_string = image_file.read()
    elif isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        byte_string = buffer.getvalue()
    else:
        raise ValueError("image_path must be a file path or a PIL Image object.")
    base64_encoded = base64.b64encode(byte_string)
    return f"data:image/jpeg;base64,{base64_encoded.decode('utf-8')}"


def process_image(image):
    encoder = ImageEncoder(
        min_image_size=364,
        max_image_size=364,
        downsampling_ratio=14,
        max_crops=12,
    )
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            image = Image.open(image_file)
            image = image.copy()  # Read image into memory before closing file
    imgs = encoder.encode(image)
    res = {key: [im[key] for im in imgs] for key in imgs[0]}
    return res["image"], res["size"], res["original_size"]


def encode_with_turns(msg, tokenizer_txt, tokenizer_img, preamble=None):
    token_ids = tokenizer_img.encode(msg)["token_ids"].tolist()[1:-1]
    encoded_txt = tokenizer_img.text_tokeniser.decode(
        token_ids, skip_special_tokens=False, ignore_oov=False
    )

    conversation = [
        {
            "role": "User",
            "message": [
                {"text": encoded_txt},
            ],
        }
    ]
    if preamble:
        conversation.insert(0, {"role": "System", "message": [{"text": preamble}]})
    return tokenizer_txt.encode_turns(conversation)[0]


def test_token_id_input(llm_instance, sampling_params, tokenizer_txt, tokenizer_img):
    """Test model outputs using tokenizer approach."""

    msg = [{"image_sizes": process_image(TEST_IMG)[1]}, {"text": TEST_MSG}]
    # Tokenize conversations
    token_ids = encode_with_turns(msg, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input = TokensPrompt(
        prompt_token_ids=token_ids,
        multi_modal_data={"image": [image_to_base64(TEST_IMG)]},
    )

    # Generate outputs
    llm_instance.reset_prefix_cache()
    outputs = llm_instance.generate(
        [engine_input],
        sampling_params=sampling_params,
    )

    # Verify outputs
    output = outputs[0]
    generated_text = output.outputs[0].text
    assert TEST_FN(generated_text), (
        f"Unexpected output from token_id_input: {generated_text}"
    )


def format_as_turns(message: str, image_paths: list[str], preamble: str | None = None):
    content = []
    # Add any additional messages as text content
    for image_path in image_paths:
        image_base64 = image_to_base64(image_path)
        content.append({"type": "image_url", "image_url": {"url": image_base64}})

    content.append({"type": "text", "text": message})
    # Create the full prompt structure
    prompt = [
        {"role": "user", "content": content},
    ]
    if preamble:
        prompt.insert(0, {"role": "system", "content": preamble})

    return prompt


def test_string_input(llm_instance, sampling_params):
    """Test model outputs using string approach."""

    prompt = format_as_turns(TEST_MSG, [TEST_IMG], PREAMBLE)

    llm_instance.reset_prefix_cache()
    prompts = [prompt]
    # setting tools=[] instead of tools=None as a workaround
    # to use chat template from processor instead of tokenizer
    outputs = llm_instance.chat(prompts, sampling_params, use_tqdm=True, tools=[])

    # Verify outputs
    output = outputs[0]
    generated_text = output.outputs[0].text
    assert TEST_FN(generated_text), (
        f"Unexpected output from string_input: {generated_text}"
    )


def test_image_splits():
    """Test that image splits calculation matches between tokenizer and model."""

    tokenizer = TemplatedBPTokenizer(
        img_size=364, img_patch_size=14 * 2, max_splits_per_img=12
    )

    image = Image.open(TEST_IMG)

    expected_image_splits = tokenizer.make_img_splits(
        tokenizer.scale_to_optimal_aspect_ratio(image)
    )

    actual_image_splits, _, _ = calculate_num_blocks(
        image.width, image.height, 12, 12, 364, False
    )

    assert len(expected_image_splits) == actual_image_splits, (
        f"Expected {len(expected_image_splits)} image splits, "
        f"but got {actual_image_splits}"
    )


if __name__ == "__main__":
    model_dir = "aya_vision_8b_hf_fp16"
    if not os.path.isdir(model_dir):
        logger.info(
            "Model directory '%s' not found. Downloading from GCS...", model_dir
        )
        gcs_path = (
            "gs://cohere-dev-central-2/saurabh_data/vision/"
            f"migration_testing/{model_dir}"
        )
        command = f"gsutil -m cp -r {gcs_path} ."
        subprocess.run(command, shell=True, check=True)
        logger.info("Download complete.")

    tokenizer_txt = TemplatedBPTokenizer(
        MM_C3_AGENTS_TEXT_TOKENISER_CONT,
        chat_template_name="chat-command-turn_tokens-v2",
    )
    tokenizer_img = Tokeniser(
        min_image_size=364, max_image_size=364, downsample_ratio=14, max_crops=12
    )
    sampling_params = SamplingParams(top_k=1, max_tokens=256)

    # Aya Vision model checkpoint
    llm_instance = LLM(
        model="./aya_vision_8b_hf_fp16/poseidon/",
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        limit_mm_per_prompt={"image": 4},  # default 999
        enforce_eager=True,
    )

    logger.info("running test_token_id_input")
    test_token_id_input(llm_instance, sampling_params, tokenizer_txt, tokenizer_img)
    logger.info("running test_string_input")
    test_string_input(llm_instance, sampling_params)
    logger.info("running test_image_splits")
    test_image_splits()
