"""Test the KV cache communication operators.

Run `python test_kvcache_comm.py`.
"""
import argparse
import ray

from vllm import EngineArgs, LLMEngine

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def run_all_workers(engine: LLMEngine, method: str, *args):
    """Run all the workers."""
    ray_worker_outputs = [
        worker.execute_method.remote(method, *args)
        for worker in engine.workers
    ]
    _ = getattr(engine.driver_worker,
            method)(*args)
    ray.get(ray_worker_outputs)


"""Test the kv cache communication."""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = "meta-llama/Llama-2-70b-hf"
    args.tensor_parallel_size = 2
    args.sep_prompt_token = True
    engine = initialize_engine(args)

    run_all_workers(engine, "set_gpucache")
    run_all_workers(engine, "send_recv_kvcache_all")
    run_all_workers(engine, "check_gpucache")

    engine.dismantle_kvcache_comm()
