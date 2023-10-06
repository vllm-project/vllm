import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams
import time
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from torch.profiler import profile, ProfilerActivity


def make_predictions(engine: LLMEngine, prompts: list,
                     sampling_params: SamplingParams):
    for index, (prompt, sampling_params) in enumerate(prompts):
        engine.add_request(str(index), prompt, sampling_params)

    # Run the engine by calling `engine.step()` manually.
    while True:
        request_outputs = engine.step()
        # for request_output in request_outputs:
        #     if request_output.finished:
        #         print(f"- {request_output.prompt} {request_output.outputs[0].text}")

        if not (engine.has_unfinished_requests()):
            break


def run_test(args: argparse.Namespace, use_cuda_graph: bool):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.use_cuda_graph = use_cuda_graph
    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.0)
    engine.compile_cuda_graphs()

    # Test the following prompts.
    test_prompts = [
        # ("A robot may not injure a human being", sampling_params),
        # ("To be or not to be,", sampling_params),
        # ("What is the meaning of life?", sampling_params),
        # ("It is only with the heart that one can see rightly", sampling_params),
        # ("A robot may not injure a human being", sampling_params),
        # ("To be or not to be,", sampling_params),
        # ("A robot may not injure a human being", sampling_params),
        # ("To be or not to be,", sampling_params),
        # ("What is the meaning of life?", sampling_params),
        # ("It is only with the heart that one can see rightly", sampling_params),
        # ("A robot may not injure a human being", sampling_params),
        ("San Francisco is a", sampling_params),
    ] * args.batch_size

    # Prevent hanging from profiler. See https://github.com/pytorch/pytorch/issues/60158
    with profile() as profiler:
        pass

    # Warm up
    make_predictions(engine, test_prompts, sampling_params)

    with profile(record_shapes=True) as prof:
        start = time.time()

        print("start profiling")
        make_predictions(engine, test_prompts, sampling_params)
        print(f"Time taken: {time.time() - start} seconds")

    prof.export_chrome_trace("trace.json")


def main(args: argparse.Namespace):
    # run_test(args, use_cuda_graph=False)
    # destroy_model_parallel()
    run_test(args, use_cuda_graph=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly")
    EngineArgs.max_num_seqs = 5
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    main(args)
