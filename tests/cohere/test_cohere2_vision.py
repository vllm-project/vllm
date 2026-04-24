# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import logging
import os
from io import BytesIO

from datatools.tokenizer.bpe import TemplatedBPTokenizer
from PIL import Image
from vllm.cohere.multimodal_tokeniser.continuous import (
    MM_C3_AGENTS_TEXT_TOKENISER_CONT,
    ImageEncoder,
    Tokeniser,
)

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.sampling_params import GuidedDecodingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREAMBLE = (
    "You are Coral, a brilliant, sophisticated, AI-assistant chatbot trained "
    "to assist human users by providing thorough responses. You are powered by "
    "Command, a large language model built by the company Cohere. Today's date "
    "is Wednesday, May 07, 2025."
)

IMG_1 = "tests/test_images/6144x512.png"
IMG_2 = "tests/test_images/receipt.png"

MSG_1 = "Describe this image."
MSG_2 = "Describe the receipt from the restaurant."
MSG_2_JSON = "Describe the receipt from the restaurant. Output in JSON format."

MSG_3 = "What is the capital of France?"
MSG_3_JSON = "What is the capital of France? Output in JSON format."

TEST_FN_1 = lambda x: "6144" in x.lower()
TEST_FN_2 = lambda x: "One Three Seafood Restaurant".lower() in x.lower()
TEST_FN_3 = lambda x: "paris" in x.lower()


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
        min_image_size=512,
        max_image_size=512,
        downsampling_ratio=16,
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

    msg_1 = [{"image_sizes": process_image(IMG_1)[1]}, {"text": MSG_1}]
    # Tokenize conversations
    token_ids_1 = encode_with_turns(msg_1, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input_1 = TokensPrompt(
        prompt_token_ids=token_ids_1,
        multi_modal_data={"image": [image_to_base64(IMG_1)]},
    )

    msg_2 = [
        {"image_sizes": process_image(IMG_1)[1]},
        {"image_sizes": process_image(IMG_2)[1]},
        {"text": MSG_2},
    ]
    # Tokenize conversations
    token_ids_2 = encode_with_turns(msg_2, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input_2 = TokensPrompt(
        prompt_token_ids=token_ids_2,
        multi_modal_data={"image": [image_to_base64(IMG_1), image_to_base64(IMG_2)]},
    )

    msg_3 = [{"text": MSG_3}]
    # Tokenize conversations
    token_ids_3 = encode_with_turns(msg_3, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input_3 = TokensPrompt(
        prompt_token_ids=token_ids_3,
    )

    # Generate outputs
    llm_instance.reset_prefix_cache()
    outputs = llm_instance.generate(
        [engine_input_1, engine_input_2, engine_input_3],
        sampling_params=sampling_params,
    )

    # Verify outputs
    test_fn = [TEST_FN_1, TEST_FN_2, TEST_FN_3]
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Output {idx}: {generated_text}")
        if not test_fn[idx](generated_text):
            print(f"[UNEXPECTED] Output {idx} did not match the expected criteria.")


def test_token_id_input_gg(llm_instance, sampling_params, tokenizer_txt, tokenizer_img):
    """Test model outputs using tokenizer approach."""

    msg_2 = [
        {"image_sizes": process_image(IMG_2)[1]},
        {"text": MSG_2_JSON},
    ]
    # Tokenize conversations
    token_ids_2 = encode_with_turns(msg_2, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input_2 = TokensPrompt(
        prompt_token_ids=token_ids_2,
        multi_modal_data={"image": [image_to_base64(IMG_2)]},
    )

    msg_3 = [{"text": MSG_3_JSON}]
    # Tokenize conversations
    token_ids_3 = encode_with_turns(msg_3, tokenizer_txt, tokenizer_img, PREAMBLE)
    # Prepare engine inputs
    engine_input_3 = TokensPrompt(
        prompt_token_ids=token_ids_3,
    )

    # Generate outputs
    llm_instance.reset_prefix_cache()
    outputs = llm_instance.generate(
        [engine_input_2, engine_input_3],
        sampling_params=sampling_params,
    )

    # Verify outputs
    test_fn = [TEST_FN_2, TEST_FN_3]
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Output {idx}: {generated_text}")
        if not test_fn[idx](generated_text):
            print(f"[UNEXPECTED] Output {idx} did not match the expected criteria.")


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

    prompt_1 = format_as_turns(MSG_1, [IMG_1], PREAMBLE)
    prompt_2 = format_as_turns(MSG_2, [IMG_1, IMG_2], PREAMBLE)
    prompt_3 = format_as_turns(MSG_3, [], PREAMBLE)

    llm_instance.reset_prefix_cache()
    prompts = [prompt_1, prompt_2, prompt_3]
    # setting tools=[] instead of tools=None as a workaround
    # to use chat template from processor instead of tokenizer
    outputs = llm_instance.chat(prompts, sampling_params, use_tqdm=True, tools=[])

    # Verify outputs
    test_fn = [TEST_FN_1, TEST_FN_2, TEST_FN_3]
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"Output {idx}: {generated_text}")
        if not test_fn[idx](generated_text):
            print(f"[UNEXPECTED] Output {idx} did not match the expected criteria.")


if __name__ == "__main__":
    tokenizer_txt = TemplatedBPTokenizer(
        MM_C3_AGENTS_TEXT_TOKENISER_CONT,
        chat_template_name="chat-command-turn_tokens-v2",
    )
    tokenizer_img = Tokeniser(
        min_image_size=512, max_image_size=512, downsample_ratio=16, max_crops=12
    )
    sampling_params = SamplingParams(top_p=0.7, temperature=0.3, max_tokens=256)

    # example checkpoint available at gs://cohere-icebox/tif/vllm/vision/IFT_Cohort20_Mammoth_SI_dedup_PSG_Eagle2_v1_32k/
    llm_instance = LLM(
        model="/host/ckpts/tmp_tif_export_7b/poseidon/",
        max_model_len=256000,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        # limit_mm_per_prompt={"image": 4},  # default 999
        guided_decoding_backend="xgrammar",
        # disable_mm_preprocessor_cache=True,
        # enable_prefix_caching=False,
    )

    logger.info("running test_token_id_input")
    test_token_id_input(llm_instance, sampling_params, tokenizer_txt, tokenizer_img)
    logger.info("running test_string_input")
    test_string_input(llm_instance, sampling_params)
    sampling_params.guided_decoding = GuidedDecodingParams(
        grammar="default", backend="xgrammar"
    )
    logger.info("running test_token_id_input_gg")
    test_token_id_input_gg(llm_instance, sampling_params, tokenizer_txt, tokenizer_img)

    # TODO: test dynamic guided generation, which will require a preamble
    # TODO: add interleaved test cases for image and text
