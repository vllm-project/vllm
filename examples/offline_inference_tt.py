import json
import argparse
from tqdm import tqdm
import uvloop
import os
import time
from pathlib import Path
from pkg_resources import resource_filename
from PIL import Image as PIL_Image
import numpy as np
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import merge_async_iterators
from vllm.inputs.data import TokensPrompt
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.model_executor.models.mllama import MLLAMA_IMAGE_TOKEN, MLLAMA_IMAGE_TOKEN_ID

def register_tt_models():
    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM
    elif llama_text_version == "llama3_subdevices":
        from models.demos.llama3_subdevices.tt.generator_vllm import LlamaForCausalLM
    elif llama_text_version == "llama2_70b":
        from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM as LlamaForCausalLM
    else:
        raise ValueError(f"Unsupported TT Llama version: {llama_text_version}, pick one of [tt_transformers, llama3_subdevices, llama2_70b]")

    ModelRegistry.register_model("TTLlamaForCausalLM", LlamaForCausalLM)

    from models.tt_transformers.tt.generator_vllm import MllamaForConditionalGeneration
    ModelRegistry.register_model("TTMllamaForConditionalGeneration", MllamaForConditionalGeneration)

    from models.tt_transformers.tt.generator_vllm import Qwen2ForCausalLM
    ModelRegistry.register_model("TTQwen2ForCausalLM", Qwen2ForCausalLM)

register_tt_models()  # Import and register models from tt-metal


def get_sample_multi_modal_llama_inputs():
    '''
    Prepare 4 sample multi-modal prompts for Llama3.2-11B
    '''
    IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))
    relative_img_paths = [None, "pasta.jpeg", "ocr_image.jpeg", "clutter.jpeg"]
    questions = [
        "Write a haiku.",
        "What is for dinner?",
        "What is the full text of this image? Do OCR",
        "What objects are in this image?"
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
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.3-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ]
    assert model in supported_models, f"Invalid model: {model}"


def run_seq_len_tests(engine_kw_args, sampling_params):
    '''
    Test generation of a few simple counting prompts with arbitrary increasing sequence lengths
    '''

    model = engine_kw_args["model"]
    is_instruct = "Instruct" in model
    count_sizes = [10, 100, 2000, 16000, 40000]
    
    if is_instruct:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    prompts = []
    for size in count_sizes:
        prompt = "Continue this counting sequence (with no explanation): " + " ".join(str(i) for i in range(1, size+1))
        if is_instruct:
            prompt = {"role": "user", "content": prompt}
            prompt = tokenizer.apply_chat_template([prompt],
                                                    tokenize=False,
                                                    add_generation_prompt=True)
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
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
    async_engine=False,
    num_scheduler_steps=10,
    disable_async_output_proc=False,
    multi_modal=False,
    test_increasing_seq_lens=False,
    sample_on_device_decode=False,
    dispatch_core_axis=None,
):
    check_tt_model_supported(model)
    
    override_tt_config = {}
    if sample_on_device_decode:
        override_tt_config["sample_on_device_decode"] = True
    if dispatch_core_axis:
        override_tt_config["dispatch_core_axis"] = dispatch_core_axis.lower()
    
    # LLM args
    engine_kw_args = {
        "model": model,
        "block_size": 64,
        "max_num_seqs": max_seqs_in_batch,
        "max_model_len": 131072,
        "disable_log_stats": False,
        "max_num_batched_tokens": 131072,
        "log_global_stats": True if measure_perf else False,
        "num_scheduler_steps": num_scheduler_steps,
        "disable_async_output_proc": disable_async_output_proc,
        "override_tt_config": override_tt_config,
    }
    
    # Generation args
    ignore_eos = True if measure_perf else False

    if greedy_sampling:
        sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, top_k=10, top_p=0.9, temperature=1.0)

    if test_increasing_seq_lens:
        assert not measure_perf, "measure_perf option not supported with test_increasing_seq_lens"
        assert not async_engine, "async_engine option not supported with test_increasing_seq_lens"
        print("Ignoring prompts json for sequence length testing")
        run_seq_len_tests(engine_kw_args, sampling_params)
        return

    # Prepare inputs
    if not measure_perf:
        if not multi_modal:
            # Load prompts from a JSON file
            with open(prompts_json, 'r') as file:
                prompts = json.load(file)
            assert isinstance(prompts, list), "Prompts must be a list of strings"
        else:
            print("Ignoring prompts json for multi-modal inference")
            prompts = get_sample_multi_modal_llama_inputs() 
        if num_repeat_prompts is not None:
            prompts = prompts * num_repeat_prompts
        print("Number of prompts:", len(prompts))
    else:
        assert perf_prompt_len is not None, "perf_prompt_len is required to generate dummy prompts"
        print("Measuring performance with dummy prompts of length", perf_prompt_len)
        print("Generating prompts with output length", max_tokens)
        
        # Prompt token ids (dummy prompts)
        prompt_token_ids_user = [0]*perf_prompt_len
        if not multi_modal:
            prompts = [{"prompt_token_ids": prompt_token_ids_user} for _ in range(max_seqs_in_batch)]
        else:
            prompt_token_ids_user.insert(0, MLLAMA_IMAGE_TOKEN_ID)
            random_pixels = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            rand_img = PIL_Image.fromarray(random_pixels, 'RGB')  # Create a PIL Image from the random pixel data
            prompts = [{"prompt_token_ids": prompt_token_ids_user, "multi_modal_data": {"image": rand_img}} for _ in range(max_seqs_in_batch)]
        
        # Sampling params
        sampling_params = sampling_params[:max_seqs_in_batch] if isinstance(sampling_params, list) else sampling_params
        sampling_params.max_tokens = max_tokens

        max_model_len = engine_kw_args["max_model_len"]
        assert_str = f"prompt length ({perf_prompt_len}) + num generated tokens ({sampling_params.max_tokens}) will exceed max_model_len ({max_model_len})"
        assert perf_prompt_len + sampling_params.max_tokens <= max_model_len, assert_str

    # Create and run LLM
    if not async_engine:
        llm = LLM(**engine_kw_args)
        if not measure_perf:
            generate_tokens(llm, prompts, sampling_params, print_output=True)
        else:
            run_inference_perf(llm, prompts, sampling_params)
    else:
        print("Using async engine")
        engine_args = AsyncEngineArgs(**engine_kw_args)
        async def _run_inference_async():
            async with build_async_engine_client_from_engine_args(engine_args) as llm:
                if not measure_perf:
                    await generate_tokens_async(llm, prompts, sampling_params, print_output=True)
                else:
                    await run_inference_perf_async(llm, prompts, sampling_params)
        uvloop.run(_run_inference_async())


def run_inference_perf(
    llm : LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        generate_tokens(llm, prompts, sampling_params, print_output=False)
    avg_time = (time.perf_counter()-start_time) / (N_inference-N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


async def run_inference_perf_async(
    llm : LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        await generate_tokens_async(llm, prompts, sampling_params, print_output=False)
    avg_time = (time.perf_counter()-start_time) / (N_inference-N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


def generate_tokens(llm : LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
    # Print the outputs.
    for output in outputs:
        request_id = int(output.request_id) + 1
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens_prompt = len(output.prompt_token_ids)
        num_tokens_output = len(output.outputs[0].token_ids)
        if print_output:
            print(f"Prompt #{request_id} ({num_tokens_prompt} tokens): {prompt!r}, Generated text ({num_tokens_output} tokens): {generated_text!r}\n")


async def generate_tokens_async(llm : MQLLMEngineClient, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # Use tokenized prompts if provided
    if prompt_token_ids is not None:
        prompts = []
        for single_prompt_token_ids in prompt_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=single_prompt_token_ids))
    
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    
    generators = []
    for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
        generator = llm.generate(prompt, sp, request_id=f"test{i}")
        generators.append(generator)
    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        prompt = res.prompt
        generated_text = res.outputs[0].text
        num_tokens_prompt = len(res.prompt_token_ids)
        num_tokens_output = len(res.outputs[0].token_ids)
        if print_output and res.finished:
            print(f"Prompt ({num_tokens_prompt} tokens): {prompt!r}, Generated text ({num_tokens_output} tokens): {generated_text!r}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B", help="Model name")
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    parser.add_argument("--measure_perf", action="store_true", help="Measure performance")
    parser.add_argument("--perf_prompt_len", type=int, default=128, help="Length of dummy prompts for performance measurement")
    parser.add_argument("--max_tokens", type=int, default=128, help="Length of outputs")
    parser.add_argument("--greedy_sampling", action="store_true", help="Use greedy decoding instead of top-k/p")
    parser.add_argument("--max_seqs_in_batch", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--num_repeat_prompts", type=int, default=2, help="Number of times to repeat prompts")
    parser.add_argument("--async_engine", action="store_true", help="Use async engine")
    parser.add_argument("--disable_async_output_proc", action="store_true", help="Disable async output processing")
    parser.add_argument("--num_scheduler_steps", type=int, default=10, help="Number of scheduler steps")
    parser.add_argument("--multi_modal", action="store_true", help="Run multi-modal inference with Llama3.2-11b")
    parser.add_argument("--test_increasing_seq_lens", action="store_true", help="Test generations of small to large sequences")
    parser.add_argument("--sample_on_device_decode", action="store_true", help="Enable sampling on device during decode")
    parser.add_argument("--dispatch_core_axis", type=str, choices=["row", "col", None], default=None, help="Dispatch core axis [row, col]")
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
        test_increasing_seq_lens=args.test_increasing_seq_lens,
        sample_on_device_decode=args.sample_on_device_decode,
        dispatch_core_axis=args.dispatch_core_axis,
    )
