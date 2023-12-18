"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import os
import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.anyscale.lora.utils import LoRARequest

SAMPLE_PROMPTS = [
    "The president of the United States is",
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]


def add_lora(llm, batch_size):
    LORA_FILE1 = "/mnt/local_storage/lora/"
    for i in range(batch_size):
        lora_request = LoRARequest(lora_id=f"lora_{i + 1}",
                                   lora_int_id=i + 1,
                                   lora_local_path=LORA_FILE1)
        assert llm.llm_engine.add_lora(lora_request)


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=40960,
        trust_remote_code=args.trust_remote_code,
        load_format="dummy" if args.use_dummy_weights else "auto",
        enable_lora=args.enable_lora,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_cache_size=args.cuda_graph_cache_size,
        dtype=args.dtype,
        flash_style=args.flash_style,
        max_chunked_prefill_len=args.max_chunked_prefill_len,
        max_num_prompt_seqs=args.max_num_prompt_seqs,
        block_size=32 if args.flash_style else args.block_size,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_model_uses_tp_1=args.speculative_model_uses_tp_1,
        ray_workers_use_nsight=args.run_with_nsight,
        disable_shared_memory=args.disable_shared_memory,
        worker_use_ray=args.worker_use_ray,
        disable_log_stats=not args.log_engine_stats,
    )

    if args.enable_lora:
        lora_request = add_lora(llm, args.batch_size)
    else:
        lora_request = None

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0 if args.use_sample else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion():
        start_time = time.perf_counter()

        if args.use_sample:
            batch = (
                SAMPLE_PROMPTS *
                (args.batch_size // len(SAMPLE_PROMPTS) + 1))[:args.batch_size]
            outputs = llm.generate(prompts=batch,
                                   sampling_params=sampling_params,
                                   use_tqdm=False,
                                   lora_request=lora_request)
        else:
            outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                                   sampling_params=sampling_params,
                                   use_tqdm=False,
                                   lora_request=lora_request)

        end_time = time.perf_counter()

        if args.verbose:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(
                    f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        latency = end_time - start_time
        return latency

    if args.profile and args.enable_cuda_graph:
        # Workaround to enable profiling cuda graphs.
        # https://github.com/pytorch/pytorch/issues/75504#issuecomment-1467065935
        llm.llm_engine.start_profile(
            profile_ray_workers=args.profile_ray_workers)
        llm.llm_engine.stop_profile(
            profile_ray_workers=args.profile_ray_workers)

    print("Warming up...")
    run_to_completion()

    if args.profile:
        model_name = args.model.replace("/", "-")
        profile_logdir_name = os.path.join(
            args.profile_logdir,
            f"{model_name}_tp-{args.tensor_parallel_size}_input-len{args.input_len}_output-len{args.output_len}_batch-size{args.batch_size}"
            .lstrip("-"))
        llm.llm_engine.start_profile(
            profile_ray_workers=args.profile_ray_workers,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                profile_logdir_name),
            with_stack=True)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion())
    print(f'Avg latency: {np.mean(latencies)} seconds')
    print(
        f'Avg ITL: {1000*np.mean(latencies)/args.output_len:.02f} milliseconds'
    )
    print(f'Peak Cuda memory: {torch.cuda.max_memory_allocated()}')

    if args.profile:
        llm.llm_engine.stop_profile(
            profile_ray_workers=args.profile_ray_workers, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--enable-lora',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument('--enable-cuda-graph',
                        action='store_true',
                        help='enable cuda graph for decoding')
    parser.add_argument('--cuda-graph-cache-size',
                        type=int,
                        default=200,
                        help='number of cuda graphs to cache')
    parser.add_argument('--use-dummy-weights',
                        action='store_true',
                        help='use-dummy-weights')
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--speculative-model-uses-tp-1',
                        action='store_true',
                        help='speculative model uses tp1')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--run-with-nsight', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-logdir', type=str, default=None)
    parser.add_argument('--profile-ray-workers', action='store_true')
    parser.add_argument('--max-chunked-prefill-len', type=int, default=-1)
    parser.add_argument('--max-num-prompt-seqs', type=int, default=1000)
    parser.add_argument('--flash-style',
                        action='store_true',
                        help='enable flash attention')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument('--use-sample',
                        action='store_true',
                        help='use sample input instead of dummy input')
    parser.add_argument('--disable-shared-memory',
                        action='store_true',
                        help='disable shared memory')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='print generated text')
    parser.add_argument('--log-engine-stats',
                        action='store_true',
                        help='log engine stats')
    parser.add_argument('--worker-use-ray',
                        action='store_true',
                        help='use Ray worker')
    args = parser.parse_args()
    main(args)
