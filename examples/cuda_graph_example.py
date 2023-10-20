import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams
import time
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import vllm.model_executor.models.llama as llama
# from torch.profiler import profile, ProfilerActivity


def make_predictions(engine: LLMEngine, prompts: list,
                     sampling_params: SamplingParams):
    for index, (prompt, sampling_params) in enumerate(prompts):
        engine.add_request(str(index), prompt, sampling_params)

    # Run the engine by calling `engine.step()` manually.
    while True:
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(f"- {request_output.prompt}{request_output.outputs[0].text}")

        if not (engine.has_unfinished_requests()):
            break


def run_test(args: argparse.Namespace, use_cuda_graph: bool):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.use_cuda_graph = use_cuda_graph
    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.0)
    engine.warm_up_cuda_graph()

    for batch_size in [3, 4, 1, 2, 5]:
        # Test the following prompts.
        test_prompts = [
            ("San Francisco is a", sampling_params),
        ] * batch_size

        # Prevent hanging from profiler. See https://github.com/pytorch/pytorch/issues/60158
        # with profile() as profiler:
        #     pass

        # with profile(record_shapes=True) as prof:
        start = time.time()

        print("start profiling")
        make_predictions(engine, test_prompts, sampling_params)
        print(f"Time taken: {time.time() - start} seconds")
        print(f"Forward time: {llama.forward_time} seconds")

        # prof.export_chrome_trace("trace.json")


def main(args: argparse.Namespace):
    print("Running without cuda graph")
    run_test(args, use_cuda_graph=False)
    destroy_model_parallel()
    print("Running with cuda graph")
    run_test(args, use_cuda_graph=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly")
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    args.max_num_seqs = max(args.batch_size, 5)
    main(args)
