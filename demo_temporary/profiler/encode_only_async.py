import random
import time


def patch():
    from vllm.wde.prefill_only.executor.gpu_executor import GPUAsyncExecutor

    simple_execute_loop = GPUAsyncExecutor.simple_execute_loop

    def p_execute_loop(self, *args, **kwargs):
        import torch
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            simple_execute_loop(self, *args, **kwargs)

        prof.export_chrome_trace("simple_execute_loop.json")

    GPUAsyncExecutor.simple_execute_loop = p_execute_loop

    double_buffer_execute_loop = GPUAsyncExecutor.double_buffer_execute_loop

    def p_execute_loop(self, *args, **kwargs):
        import torch
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            double_buffer_execute_loop(self, *args, **kwargs)
        prof.export_chrome_trace("double_buffer_execute_loop.json")

    GPUAsyncExecutor.double_buffer_execute_loop = p_execute_loop


def benchmark_vllm(args):
    random.seed(args.seed)
    patch()

    from vllm.wde.encode_only.arg_utils import (  # noqa: E501
        EncodeOnlyEngineArgs as EngineArgs)
    from vllm.wde.entrypoints.llm import LLMEngine

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        quantization_param_path=args.quantization_param_path,
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


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.num_prompts = 100

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.seed = 0
    args.quantization = None
    args.quantization_param_path = None
    args.max_model_len = None

    args.dtype = "half"
    args.device = "cuda"
    args.batchsize = [4]

    from concurrent.futures import ProcessPoolExecutor

    def run_vllm(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    for scheduling in ["async", "double_buffer"]:
        print(scheduling)
        args.scheduling = scheduling
        run_vllm(args)
