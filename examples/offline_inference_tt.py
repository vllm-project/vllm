# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import regex as re
import requests
import uvloop
from PIL import Image as PIL_Image
from pkg_resources import resource_filename
from tqdm import tqdm
from transformers import AutoTokenizer

import vllm.envs as envs
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.utils import merge_async_iterators


def register_tt_models():
    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif llama_text_version == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]"
        )

    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    ModelRegistry.register_model(
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    ModelRegistry.register_model("TTQwen2ForCausalLM", path_qwen_text)
    ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen_text)

    # Qwen2.5 - Vision
    ModelRegistry.register_model(
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # Mistral
    ModelRegistry.register_model(
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    # Gemma3
    ModelRegistry.register_model(
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # DeepseekV3
    ModelRegistry.register_model(
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )

    # GPT-OSS
    ModelRegistry.register_model(
        "TTGptOssForCausalLM",
        "models.tt_transformers.tt.generator_vllm:GptOssForCausalLM",
    )


register_tt_models()  # Import and register models from tt-metal


def get_sample_multi_modal_llama_inputs():
    """
    Prepare 4 sample multi-modal prompts for Llama3.2-11B
    """
    MLLAMA_IMAGE_TOKEN = "<|image|>"
    IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))
    relative_img_paths = [None, "pasta.jpeg", "ocr_image.jpeg", "clutter.jpeg"]
    questions = [
        "Write a haiku.",
        "What is for dinner?",
        "What is the full text of this image? Do OCR",
        "What objects are in this image?",
    ]
    inputs = []
    for relative_img_path, question in zip(relative_img_paths, questions):
        if relative_img_path is not None:
            with open(IMG_PATH / relative_img_path, "rb") as f:
                img = PIL_Image.open(f).convert("RGB")
            prompt = f"{MLLAMA_IMAGE_TOKEN}{question}"
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        else:
            inputs.append({"prompt": question})
    return inputs


def get_sample_multi_modal_inputs(model: str, multi_image: bool):
    """
    Build sample multi-modal inputs for vision-language models.
    Currently supports Qwen2.5-VL and Gemma-3.

    Args:
        model (str): Hugging Face model identifier

    Returns:
        list[dict]: A list of input dicts ready for model.generate
    """
    text_prompts = []
    imgs = []

    text_prompts_content = [
        [{"type": "text", "text": "Count to 20."}],
        [{"type": "text", "text": "What is the capital of France?"}],
        [{"type": "text", "text": "Describe the band Oasis in 300 words."}],
        [{"type": "text", "text": "Write a haiku about an orange."}],
    ]

    single_image_prompts_content = [
        [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
        [
            {"type": "text", "text": "Is there a cat?"},
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
        [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg",
            },
        ],
    ]

    multi_image_prompts_content = [
        [
            {"type": "text", "text": "Compare these images."},
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg",
            },
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
            },
        ],
    ]

    content = []
    if "Qwen2.5-VL" in model:
        # [INFO] Qwen-VL currently does not support a mixture of
        # text-image and text-only inputs
        content += single_image_prompts_content
    else:
        content += text_prompts_content + single_image_prompts_content
        if multi_image:
            content += multi_image_prompts_content

    prompts = []
    for c in content:
        prompts.append([{"role": "user", "content": c}])

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    for prompt in prompts:
        chat_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs = None
        if "Qwen2.5-VL" in model:
            assert not multi_image, (
                "Multi-image inputs not supported yet for Qwen2.5-VL"
            )
            # Lazy import only when needed
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(prompt)
            assert video_inputs is None, "Video inputs not supported yet"
            image_inputs = image_inputs[0] if image_inputs else None
        elif "gemma-3" in model:
            image_inputs = [
                ctnt["image"]
                for entry in prompt
                for ctnt in entry["content"]
                if ctnt["type"] == "image"
            ]
            image_inputs = [
                PIL_Image.open(requests.get(image_url, stream=True).raw)
                if image_url is not None
                else None
                for image_url in image_inputs
            ]
        else:
            raise NotImplementedError(
                f"Multi-modal preprocessing not implemented for model: {model}"
            )

        imgs.append(image_inputs)
        text_prompts.append(chat_prompt)

    # Pack inputs
    inputs = []
    for img, text_prompt in zip(imgs, text_prompts):
        entry = {"prompt": text_prompt}
        if img and all(item is not None for item in img):
            entry["multi_modal_data"] = {"image": img}
        inputs.append(entry)

    return inputs


def check_tt_model_supported(model):
    supported_models = [
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.3-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-R1-0528",
    ]
    assert model in supported_models, f"Invalid model: {model}"


def run_seq_len_tests(engine_kw_args, sampling_params):
    """
    Test generation of a few simple counting prompts
    with arbitrary increasing sequence lengths
    """

    model = engine_kw_args["model"]
    is_instruct = "Instruct" in model
    count_sizes = [10, 100, 2000, 16000, 40000]

    if is_instruct:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    prompts = []
    for size in count_sizes:
        prompt = "Continue this counting sequence (with no explanation): " + " ".join(
            str(i) for i in range(1, size + 1)
        )
        if is_instruct:
            prompt = {"role": "user", "content": prompt}
            prompt = tokenizer.apply_chat_template(
                [prompt], tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)

    llm = LLM(**engine_kw_args)

    # Run generation one prompt at a time
    for i in range(len(count_sizes)):
        generate_tokens(llm, [prompts[i]], sampling_params, print_output=True)


def run_inference(
    model,
    prompts_json,
    max_tokens=128,
    max_seqs_in_batch=32,
    num_repeat_prompts=2,
    measure_perf=False,
    perf_prompt_len=None,
    greedy_sampling=False,  # Use greedy decoding instead of top-k/p
    async_engine=False,
    num_scheduler_steps=10,
    disable_async_output_proc=False,
    multi_modal=False,
    multi_image=False,
    mm_processor_kwargs=None,
    test_increasing_seq_lens=False,
    override_tt_config=None,
    max_model_len=None,
    max_num_batched_tokens=None,
    data_parallel_size=1,
    block_size=64,
):
    check_tt_model_supported(model)

    if multi_modal:
        supported_models = [
            "Llama-3.2",
            "Qwen2.5-VL",
            "gemma",
        ]
        assert any(name in model for name in supported_models), (
            "The multi-modal inference test "
            f"currently only supports {supported_models} models"
        )

    if data_parallel_size > 1:
        assert envs.VLLM_USE_V1, "Data parallel size > 1 is only supported with V1"

    # LLM args
    engine_kw_args = {
        "model": model,
        "block_size": block_size,
        "max_num_seqs": max_seqs_in_batch,
        "max_model_len": max_model_len,
        "disable_log_stats": False,
        "max_num_batched_tokens": max_num_batched_tokens,
        "log_global_stats": measure_perf,
        "num_scheduler_steps": num_scheduler_steps,
        "disable_async_output_proc": disable_async_output_proc,
        "data_parallel_size": data_parallel_size,
    }

    try:
        if override_tt_config:
            engine_kw_args["override_tt_config"] = json.loads(override_tt_config)
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON string for override_tt_config: {err}") from err

    try:
        if mm_processor_kwargs:
            engine_kw_args["mm_processor_kwargs"] = json.loads(mm_processor_kwargs)
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON string for mm_processor_kwargs: {err}") from err

    # Generation args
    ignore_eos = measure_perf

    if greedy_sampling:
        sampling_params = SamplingParams(
            max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            top_k=10,
            top_p=0.9,
            temperature=1.0,
        )

    if test_increasing_seq_lens:
        assert not measure_perf, (
            "measure_perf option not supported with test_increasing_seq_lens"
        )
        assert not async_engine, (
            "async_engine option not supported with test_increasing_seq_lens"
        )
        print("Ignoring prompts json for sequence length testing")
        run_seq_len_tests(engine_kw_args, sampling_params)
        return

    # Prepare inputs
    if not measure_perf:
        if not multi_modal:
            # Load prompts from a JSON file
            with open(prompts_json) as file:
                prompts = json.load(file)
            assert isinstance(prompts, list), "Prompts must be a list of strings"
        else:
            print("Ignoring prompts json for multi-modal inference")
            if "Llama-3.2" in model:
                prompts = get_sample_multi_modal_llama_inputs()
            elif any(name in model for name in ["Qwen2.5-VL", "gemma"]):
                prompts = get_sample_multi_modal_inputs(model, multi_image)
            else:
                raise ValueError(
                    f"Unsupported model for multi-modal inference test: {model}"
                )
        if num_repeat_prompts is not None:
            prompts = prompts * num_repeat_prompts
        print("Number of prompts:", len(prompts))
    else:
        assert perf_prompt_len is not None, (
            "perf_prompt_len is required to generate dummy prompts"
        )
        print("Measuring performance with dummy prompts of length", perf_prompt_len)
        print("Generating prompts with output length", max_tokens)

        # Prompt token ids (dummy prompts)
        prompt_token_ids_user = [0] * perf_prompt_len
        if not multi_modal:
            prompts = [
                {"prompt_token_ids": prompt_token_ids_user}
                for _ in range(max_seqs_in_batch * data_parallel_size)
            ]
        else:
            if "Llama-3.2" in model:
                IMAGE_TOKEN_ID = 128256  # Specific to multi-modal llama
            elif "Qwen2.5-VL" in model:
                IMAGE_TOKEN_ID = 151655  # Specific to multi-modal qwen
            elif "gemma" in model:
                IMAGE_TOKEN_ID = 262144  # Specific to multi-modal gemma
            else:
                raise ValueError(
                    f"Unsupported model for multi-modal inference test in perf "
                    f"mode: {model}"
                )
            prompt_token_ids_user.insert(0, IMAGE_TOKEN_ID)
            random_pixels = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            rand_img = PIL_Image.fromarray(
                random_pixels, "RGB"
            )  # Create a PIL Image from the random pixel data
            prompts = [
                {
                    "prompt_token_ids": prompt_token_ids_user,
                    "multi_modal_data": {"image": rand_img},
                }
                for _ in range(max_seqs_in_batch)
            ]

        # Sampling params
        sampling_params = (
            sampling_params[:max_seqs_in_batch]
            if isinstance(sampling_params, list)
            else sampling_params
        )
        sampling_params.max_tokens = max_tokens

    # Create and run LLM
    if not async_engine:
        llm = LLM(**engine_kw_args)
        if not measure_perf:
            generate_tokens(llm, prompts, sampling_params, print_output=True)
        else:
            max_model_len = llm.llm_engine.model_config.max_model_len
            check_valid_perf_prompt_len(max_model_len, perf_prompt_len, sampling_params)
            run_inference_perf(llm, prompts, sampling_params)
    else:
        print("Using async engine")
        engine_args = AsyncEngineArgs(**engine_kw_args)

        # For DP > 1, send prompts round-robin to DP ranks
        if data_parallel_size > 1:
            print("Will send prompts round-robin to DP ranks")
            dp_ranks = [i % data_parallel_size for i in range(len(prompts))]
        else:
            dp_ranks = None

        async def _run_inference_async():
            async with build_async_engine_client_from_engine_args(engine_args) as llm:
                if not measure_perf:
                    await generate_tokens_async(
                        llm,
                        prompts,
                        sampling_params,
                        dp_ranks=dp_ranks,
                        print_output=True,
                    )
                else:
                    max_model_len = llm.model_config.max_model_len
                    check_valid_perf_prompt_len(
                        max_model_len, perf_prompt_len, sampling_params
                    )
                    await run_inference_perf_async(
                        llm, prompts, sampling_params, dp_ranks=dp_ranks
                    )

        uvloop.run(_run_inference_async())


def check_valid_perf_prompt_len(max_model_len, perf_prompt_len, sampling_params):
    assert_str = (
        f"prompt length ({perf_prompt_len}) + num generated tokens "
        f"({sampling_params.max_tokens}) will exceed max_model_len "
        f"({max_model_len})"
    )
    assert perf_prompt_len + sampling_params.max_tokens <= max_model_len, assert_str


def run_inference_perf(
    llm: LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=3,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        generate_tokens(llm, prompts, sampling_params, print_output=False)
    avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


async def run_inference_perf_async(
    llm: LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=3,
    dp_ranks=None,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        await generate_tokens_async(
            llm, prompts, sampling_params, dp_ranks=dp_ranks, print_output=False
        )
    avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


def _format_prompt_for_display(prompt: str) -> str:
    """Format prompt for display by replacing repetitive image tokens."""
    # Count and replace consecutive image_soft_token occurrences
    pattern = r"(<image_soft_token>)+"

    def replace_tokens(match):
        count = match.group(0).count("<image_soft_token>")
        return f"<image_tokens:{count}>"

    cleaned_prompt = re.sub(pattern, replace_tokens, prompt)
    return cleaned_prompt


def generate_tokens(
    llm: LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True
):
    # Use tokenized prompts if provided
    if prompt_token_ids is not None:
        prompts = []
        for single_prompt_token_ids in prompt_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=single_prompt_token_ids))

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        request_id = int(output.request_id) + 1
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens_prompt = len(output.prompt_token_ids)
        num_tokens_output = len(output.outputs[0].token_ids)
        if print_output:
            prompt = _format_prompt_for_display(prompt)
            print(
                f"Prompt #{request_id} "
                f"({num_tokens_prompt} tokens): {prompt!r}, "
                "Generated text "
                f"({num_tokens_output} tokens): {generated_text!r}\n"
            )


async def generate_tokens_async(
    llm: MQLLMEngineClient,
    prompts,
    sampling_params,
    dp_ranks=None,
    prompt_token_ids=None,
    print_output=True,
):
    # Use tokenized prompts if provided
    if prompt_token_ids is not None:
        prompts = []
        for single_prompt_token_ids in prompt_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=single_prompt_token_ids))

    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    if dp_ranks:
        assert envs.VLLM_USE_V1, "DP ranks are only supported with V1"
        assert len(dp_ranks) == len(prompts), (
            "DP ranks must be the same length as prompts"
        )

    generators = []
    for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
        if dp_ranks:
            generator = llm.generate(
                prompt, sp, request_id=f"test{i}", data_parallel_rank=dp_ranks[i]
            )
        else:
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
        generators.append(generator)
    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        request_id = res.request_id
        prompt = res.prompt
        generated_text = res.outputs[0].text
        num_tokens_prompt = len(res.prompt_token_ids)
        num_tokens_output = len(res.outputs[0].token_ids)
        if print_output and res.finished:
            prompt = _format_prompt_for_display(prompt)
            print(
                f"Prompt {request_id} "
                f"({num_tokens_prompt} tokens): {prompt!r}, "
                "Generated text "
                f"({num_tokens_output} tokens): {generated_text!r}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-70B", help="Model name"
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="tt_metal/prompts.json",
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--measure_perf", action="store_true", help="Measure performance"
    )
    parser.add_argument(
        "--perf_prompt_len",
        type=int,
        default=128,
        help="Length of dummy prompts for performance measurement",
    )
    parser.add_argument("--max_tokens", type=int, default=128, help="Length of outputs")
    parser.add_argument(
        "--greedy_sampling",
        action="store_true",
        help="Use greedy decoding instead of top-k/p",
    )
    parser.add_argument(
        "--max_seqs_in_batch",
        type=int,
        default=32,
        help="Maximum batch size for inference",
    )
    parser.add_argument(
        "--num_repeat_prompts",
        type=int,
        default=2,
        help="Number of times to repeat prompts",
    )
    parser.add_argument("--async_engine", action="store_true", help="Use async engine")
    parser.add_argument(
        "--disable_async_output_proc",
        action="store_true",
        help="Disable async output processing",
    )
    parser.add_argument(
        "--num_scheduler_steps", type=int, default=10, help="Number of scheduler steps"
    )
    parser.add_argument(
        "--multi_modal",
        action="store_true",
        help="Run multi-modal inference (vision + text)",
    )
    parser.add_argument(
        "--multi_image", action="store_true", help="Run multi-image inference"
    )
    parser.add_argument(
        "--test_increasing_seq_lens",
        action="store_true",
        help="Test generations of small to large sequences",
    )
    parser.add_argument(
        "--override_tt_config",
        type=str,
        default=None,
        help="Custom TT options as Json string",
    )
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model len")
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Max num batched tokens",
    )
    parser.add_argument(
        "--mm_processor_kwargs",
        type=str,
        default=None,
        help="Multi-modal processor kwargs",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Data parallel size",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="KV cache block size",
    )

    args = parser.parse_args()

    run_inference(
        args.model,
        args.prompts_json,
        measure_perf=args.measure_perf,
        perf_prompt_len=args.perf_prompt_len,
        max_tokens=args.max_tokens,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch,
        num_repeat_prompts=args.num_repeat_prompts,
        async_engine=args.async_engine,
        num_scheduler_steps=args.num_scheduler_steps,
        disable_async_output_proc=args.disable_async_output_proc,
        multi_modal=args.multi_modal,
        multi_image=args.multi_image,
        mm_processor_kwargs=args.mm_processor_kwargs,
        test_increasing_seq_lens=args.test_increasing_seq_lens,
        override_tt_config=args.override_tt_config,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        data_parallel_size=args.data_parallel_size,
        block_size=args.block_size,
    )
