import random
import time


def benchmark(args):
    random.seed(args.seed)

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
                             scheduling=args.scheduling,
                             data_parallel_size=args.data_parallel_size)

    engine = LLMEngine.from_engine_args(engine_args)

    for batchsize in args.batchsize:
        engine.engine_config.scheduler_config.set_args(max_num_seqs=batchsize)
        engine.executor.ensure_start_execute_loop()

        # Because each thread has to load the model separately,
        # the loading may not be completed here.
        # If it is run only once, the measured data parallel speed will be low.

        for i in range(3):
            start = time.perf_counter()
            for request_id, prompt in enumerate(requests):
                engine.add_request(str(request_id), prompt)

            while engine.has_unfinished_requests():
                engine.step()
            end = time.perf_counter()
            elapsed_time = end - start

        print(f"Batchsize {batchsize}, Throughput: "
              f"{len(requests) / elapsed_time:.4f} requests/s")

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
    args.dtype = "half"
    args.device = "cuda"
    args.batchsize = [1, 2, 4, 8, 16]
    args.max_data_parallel_size = 1

    from concurrent.futures import ProcessPoolExecutor

    def run_vllm(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()

    for scheduling in ["async", "double_buffer"]:
        for data_parallel_size in range(args.max_data_parallel_size + 1):
            print("scheduling:", scheduling, "data_parallel_size",
                  data_parallel_size)
            args.data_parallel_size = data_parallel_size
            args.scheduling = scheduling
            run_vllm(args)
