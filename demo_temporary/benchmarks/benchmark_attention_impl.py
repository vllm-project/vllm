import os
import random
import time


def benchmark_vllm(args):
    random.seed(args.seed)
    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_impl

    import gc

    import torch

    from vllm.wde.encode_only.arg_utils import (  # noqa: E501
        EncodeOnlyEngineArgs as EngineArgs)
    from vllm.wde.entrypoints.llm import LLMEngine

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    engine_args = EngineArgs(model=args.model,
                             tokenizer=args.tokenizer,
                             seed=args.seed,
                             trust_remote_code=args.trust_remote_code,
                             dtype=args.dtype,
                             max_model_len=args.max_model_len,
                             device=args.device,
                             max_num_seqs=32,
                             scheduling=args.scheduling)

    engine = LLMEngine.from_engine_args(engine_args)

    for batchsize in args.batchsize:
        engine.engine_config.scheduler_config.set_args(max_num_seqs=batchsize)

        start = time.perf_counter()
        for request_id, prompt in enumerate(requests):
            engine.add_request(str(request_id), prompt)

        n_step = 0
        while engine.has_unfinished_requests():
            engine.step()
            n_step += 1
        end = time.perf_counter()

        elapsed_time = end - start
        delay = elapsed_time / n_step

        print(f"Batchsize {batchsize}, Throughput: "
              f"{len(requests) / elapsed_time:.4f} requests/s, "
              f"Delay {delay * 1000:0.2f} ms, n_step {n_step}")

        engine.executor.shutdown_execute_loop()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.num_prompts = 10000

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.seed = 0
    args.max_model_len = None
    args.device = "cuda"
    args.batchsize = [1, 2, 4, 8, 16, 32, 64]
    args.scheduling = "double_buffer"

    from concurrent.futures import ProcessPoolExecutor

    def run_vllm(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    AttentionImpls_fp32 = ["TORCH_SDPA", "XFORMERS", "TORCH_NAIVE"]
    AttentionImpls_fp16 = [
        "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
    ]
    AttentionImpls_bf16 = [
        "FLASH_ATTN", "TORCH_SDPA", "XFORMERS", "FLASHINFER", "TORCH_NAIVE"
    ]

    AttentionImpls = {
        "float": AttentionImpls_fp32,
        "half": AttentionImpls_fp16,
        "bfloat16": AttentionImpls_bf16,
    }

    for dtype, attention_impls in AttentionImpls.items():
        print("dtype:", dtype)
        for attention_impl in attention_impls:
            print("attention_impl:", attention_impl)
            args.attention_impl = attention_impl
            args.dtype = dtype
            run_vllm(args)
