import random
import time
import argparse

from vllm import LLM, SamplingParams

NUM_REQUESTS_DEFAULT = 256
MAX_SEQ_LEN_DEFAULT = 1024
MAX_TOKENS_DEFAULT = 128
SAMPLE_PROMPTS = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    "The future of AI is",
]


def run_bench(model_name,
              model_revision,
              is_sparse,
              quant_method,
              max_seq_len,
              max_tokens,
              num_requests,
              num_gpus,
              num_warmup_iters=1,
              num_bench_iters=5,
              possible_prompts=SAMPLE_PROMPTS,
              enforce_eager=True):
    print("Run bench with:")
    print(f"  model_name = {model_name}")
    print(f"  model_revision = {model_revision}")
    print(f"  is_sparse = {is_sparse}")
    print(f"  quant_method = {quant_method}")
    print(f"  max_seq_len = {max_seq_len}")
    print(f"  max_tokens = {max_tokens}")
    print(f"  num_requests = {num_requests}")
    print(f"  num_gpus = {num_gpus}")
    print(f"  num_warmup_iters = {num_warmup_iters}")
    print(f"  num_bench_iters = {num_bench_iters}")

    prompts = []
    for _ in range(num_requests):
        index = random.randint(0, len(possible_prompts) - 1)
        prompts.append(possible_prompts[index])

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=max_tokens)

    # Create LLM
    llm = LLM(
        model=model_name,
        revision=model_revision,
        sparsity="sparse_w16a16" if is_sparse else None,
        enforce_eager=enforce_eager,
        #   dtype=torch.bfloat16,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        max_model_len=max_seq_len,
        quantization=quant_method,
    )

    for i in range(num_warmup_iters):
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_time = time.time() - start_time
        print(f"Warmup iter {i} time: {elapsed_time} [secs]")

    iter_times = []
    for i in range(num_bench_iters):
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        iter_times.append(time.time() - start_time)
        print(f"Bench iter {i} time: {iter_times[-1]} [secs]")

    average_iter_time = sum(iter_times) / num_bench_iters
    print(f"Average per iter time: {average_iter_time} [secs]")

    # Print outputs of the last iter
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return average_iter_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument("--quant_method", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS_DEFAULT)
    parser.add_argument("--num_requests",
                        type=int,
                        default=NUM_REQUESTS_DEFAULT)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_warmup_iters", type=int, default=1)
    parser.add_argument("--num_bench_iters", type=int, default=5)

    args = parser.parse_args()

    run_bench(args.model_name, args.model_revision, args.is_sparse,
              args.quant_method, args.max_seq_len, args.max_tokens,
              args.num_requests, args.num_gpus, args.num_warmup_iters,
              args.num_bench_iters)
