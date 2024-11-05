"""Benchmark guided decoding throughput."""
import argparse, json, random, time
import torch, uvloop

from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import DEVICE_OPTIONS, AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "User Profile",
  "type": "object",
  "properties": {
    "userId": {
      "type": "string",
      "description": "Unique identifier for the user."
    },
    "personalInfo": {
      "type": "object",
      "properties": {
        "firstName": {
          "type": "string",
          "description": "The user's first name."
        },
        "lastName": {
          "type": "string",
          "description": "The user's last name."
        },
        "age": {
          "type": "integer",
          "minimum": 0,
          "description": "The user's age."
        },
        "phoneNumbers": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "enum": ["home", "work", "mobile"],
                "description": "Type of phone number."
              },
              "number": {
                "type": "string",
                "pattern": "^\\+?[1-9]\\d{1,14}$",
                "description": "Phone number in E.164 format."
              }
            },
            "required": ["type", "number"]
          },
          "description": "List of phone numbers associated with the user."
        }
      },
      "required": ["firstName", "lastName"]
    },
    "address": {
      "type": "object",
      "properties": {
        "street": {
          "type": "string",
          "description": "Street address."
        },
        "city": {
          "type": "string",
          "description": "City name."
        },
        "state": {
          "type": "string",
          "description": "State or province."
        },
        "postalCode": {
          "type": "string",
          "pattern": "^\\d{5}(-\\d{4})?$",
          "description": "Postal code."
        },
        "country": {
          "type": "string",
          "description": "Country name."
        }
      },
      "required": ["street", "city", "state", "postalCode", "country"]
    },
    "preferences": {
      "type": "object",
      "properties": {
        "newsletterSubscribed": {
          "type": "boolean",
          "description": "Indicates if the user is subscribed to the newsletter."
        },
        "favoriteCategories": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "List of user's favorite categories."
        }
      },
      "required": ["newsletterSubscribed"]
    },
    "accountStatus": {
      "type": "string",
      "enum": ["active", "inactive", "suspended"],
      "description": "Current status of the user's account."
    },
    "registrationDate": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 formatted date-time of user registration."
    }
  },
  "required": ["userId", "personalInfo", "address", "accountStatus", "registrationDate"]
}

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    disable_async_output_proc: bool = False,
    guided_decoding: bool = False,
    debug: bool = False,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=EngineArgs.load_format,
        num_scheduler_steps=num_scheduler_steps,
        disable_async_output_proc=disable_async_output_proc,
        guided_decoding_backend="outlines",
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
                guided_decoding=GuidedDecodingParams(json=SCHEMA) if guided_decoding else None,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start


async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    disable_async_output_proc: bool = False,
    guided_decoding: bool = False,
    debug: bool = False,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    from vllm import SamplingParams
    engine_args = AsyncEngineArgs(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=EngineArgs.load_format,
        num_scheduler_steps=num_scheduler_steps,
        disable_async_output_proc=disable_async_output_proc,
        worker_use_ray=False,
        disable_log_requests=False,
        guided_decoding_backend="outlines",
    )

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # Add the requests to the engine.
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        for prompt, _, output_len in requests:
            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=output_len,
                    guided_decoding=GuidedDecodingParams(json=SCHEMA) if guided_decoding else None,
                ))

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
          if debug: print(res.outputs[0].text)
        end = time.perf_counter()
        return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Synthesize a prompt with the given input length.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    prompt = f"Generate an example of a user profile given the following schema: {json.dumps(SCHEMA)}"
    input_len = len(tokenizer(prompt).input_ids)
    requests = [(prompt, input_len, args.output_len)
                for _ in range(args.num_prompts)]

    run_args = [
        requests, args.model, args.tokenizer,
        args.tensor_parallel_size, args.seed, args.n,
        args.trust_remote_code, args.dtype, args.max_model_len,
        args.enforce_eager, args.kv_cache_dtype, args.device,
        args.enable_prefix_caching, args.enable_chunked_prefill,
        args.max_num_batched_tokens, args.distributed_executor_backend,
        args.gpu_memory_utilization, args.num_scheduler_steps,
        args.disable_async_output_proc, args.guided_decoding, args.debug,
    ]

    if args.async_engine:
        run_args.append(args.disable_frontend_multiprocessing)
        elapsed_time = uvloop.run(run_vllm_async(*run_args))
    else:
        elapsed_time = run_vllm(*run_args)
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark guided decoding.")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=10,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument("--device",
                        type=str,
                        default="auto",
                        choices=DEVICE_OPTIONS,
                        help='device type for vLLM execution')
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="Enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        "--disable-async-output-proc",
        action='store_true',
        default=False,
        help="Disable async output processor for vLLM backend.")
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--guided-decoding",
                        action='store_true',
                        default=False,
                        help="Whether to enable JSON decoding or not.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)
