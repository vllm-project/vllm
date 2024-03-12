"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate_ <request_rate>

NOTE: This script is a modified version of benchmarks/benchmark_serving.py from
 the upstream vllm repo at commit a4211a4dc.
"""
import argparse
import asyncio
import random
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple, NamedTuple

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from .common import generate_synthetic_requests, print_serving_request_io
from .datasets_registry import get_dataset, DatasetArgs
from .benchmark_result import (BenchmarkResult,
                               BenchmarkServingResultMetadataKeys as
                               ResultMetadataKeys,
                               BenchmarkServingResultMetricTemplates as
                               ResultMetricTemplates)

from neuralmagic.benchmarks.scripts.backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)


@dataclass
class BenchmarkMetrics:

    @dataclass
    class Metadata:
        completed: int
        duration: float
        total_input: int
        total_output: int

    @dataclass
    class Metrics:
        request_throughput: float
        input_throughput: float
        output_throughput: float
        median_request_latency: float
        p90_request_latency: float
        p99_request_latency: float
        mean_ttft_ms: float
        median_ttft_ms: float
        p90_ttft_ms: float
        p99_ttft_ms: float
        mean_tpot_ms: float
        median_tpot_ms: float
        p90_tpot_ms: float
        p99_tpot_ms: float

    metadata: Metadata
    metrics: Metrics

    def update_benchmark_result_metadata(
            self, result: BenchmarkResult) -> BenchmarkResult:
        rmk = ResultMetadataKeys
        metadata = {
            rmk.completed: self.metadata.completed,
            rmk.duration: self.metadata.duration,
            rmk.total_input: self.metadata.total_input,
            rmk.total_output: self.metadata.total_output
        }
        result[BenchmarkResult.METADATA_KEY_].update(metadata)
        return result

    def update_benchmark_result_metrics(
            self, result: BenchmarkResult) -> BenchmarkResult:
        rmt = ResultMetricTemplates
        result.add_metric(rmt.request_throughput,
                          self.metrics.request_throughput)
        result.add_metric(rmt.input_throughput, self.metrics.input_throughput)
        result.add_metric(rmt.output_throughput,
                          self.metrics.output_throughput)
        result.add_metric(rmt.median_request_latency,
                          self.metrics.median_request_latency)
        result.add_metric(rmt.p90_request_latency,
                          self.metrics.p90_request_latency)
        result.add_metric(rmt.p99_request_latency,
                          self.metrics.p99_request_latency)
        result.add_metric(rmt.mean_ttft_ms, self.metrics.mean_ttft_ms)
        result.add_metric(rmt.median_ttft_ms, self.metrics.median_ttft_ms)
        result.add_metric(rmt.p90_ttft_ms, self.metrics.p90_ttft_ms)
        result.add_metric(rmt.p99_ttft_ms, self.metrics.p99_ttft_ms)
        result.add_metric(rmt.mean_tpot_ms, self.metrics.mean_tpot_ms)
        result.add_metric(rmt.median_tpot_ms, self.metrics.median_tpot_ms)
        result.add_metric(rmt.p90_tpot_ms, self.metrics.p90_tpot_ms)
        result.add_metric(rmt.p99_tpot_ms, self.metrics.p99_tpot_ms)
        return result

    def update_benchmark_result(self,
                                result: BenchmarkResult) -> BenchmarkResult:
        result = self.update_benchmark_result_metadata(result)
        result = self.update_benchmark_result_metrics(result)
        return result


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkMetrics:
    total_output = 0
    total_input = 0
    completed = 0
    latencies = []
    tpots = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(tokenizer.encode(outputs[i].generated_text))
            total_output += output_len
            total_input += input_requests[i][1]
            latencies.append(outputs[i].latency)
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            ttfts.append(outputs[i].ttft)
            completed += 1

    metrics = BenchmarkMetrics(
        metadata=BenchmarkMetrics.Metadata(completed=completed,
                                           duration=dur_s,
                                           total_input=total_input,
                                           total_output=total_output),
        metrics=BenchmarkMetrics.Metrics(
            request_throughput=completed / dur_s,
            input_throughput=total_input / dur_s,
            output_throughput=total_output / dur_s,
            median_request_latency=np.median(latencies) * 1000,
            p90_request_latency=np.percentile(latencies, 90) * 1000,
            p99_request_latency=np.percentile(latencies, 99) * 1000,
            mean_ttft_ms=np.mean(ttfts) * 1000,
            median_ttft_ms=np.median(ttfts) * 1000,
            p90_ttft_ms=np.percentile(ttfts, 90) * 1000,
            p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
            mean_tpot_ms=np.mean(tpots) * 1000,
            median_tpot_ms=np.median(tpots) * 1000,
            p90_tpot_ms=np.percentile(tpots, 90) * 1000,
            p99_tpot_ms=np.percentile(tpots, 99) * 1000))

    return metrics


async def benchmark(backend: str, api_url: str, model_id: str,
                    tokenizer: PreTrainedTokenizerBase,
                    input_requests: List[Tuple[str, int, int]], best_of: int,
                    use_beam_search: bool, request_rate: float,
                    disable_tqdm: bool,
                    log_model_io: bool) -> BenchmarkMetrics:
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    print(f"Traffic request rate: {request_rate}")

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Dump model i/o
    if log_model_io:
        print_serving_request_io(input_requests, outputs)

    metrics = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print(f"Successful requests: {metrics.metadata.completed}")
    print(f"Benchmark duration: {metrics.metadata.duration:2f} s")
    print(f"Total input tokens: {metrics.metadata.total_input}")
    print(f"Total generated tokens: {metrics.metadata.total_output}")
    print(
        f"Request throughput: {metrics.metrics.request_throughput:.2f} requests/s"
    )
    print(
        f"Input token throughput: {metrics.metrics.input_throughput:.2f} tokens/s"
    )
    print(
        f"Output token throughput: {metrics.metrics.output_throughput:.2f} tokens/s"
    )
    print(
        f"Median request latency: {metrics.metrics.median_request_latency:.2f} ms"
    )
    print(f"P90 request latency: {metrics.metrics.p90_request_latency:.2f} ms")
    print(f"P99 request latency: {metrics.metrics.p99_request_latency:.2f} ms")
    print(f"Mean TTFT: {metrics.metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.metrics.median_ttft_ms:.2f} ms")
    print(f"P90 TTFT: {metrics.metrics.p90_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.metrics.median_tpot_ms:.2f} ms")
    print(f"P90 TPOT: {metrics.metrics.p90_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.metrics.p99_tpot_ms:.2f} ms")

    return metrics


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    num_prompts, request_rate = (None, None)
    if args.nr_qps_pair_:
        num_prompts, request_rate = (args.nr_qps_pair_.num_prompts,
                                     args.nr_qps_pair_.request_rate)
    else:
        num_prompts, request_rate = (args.num_prompts_, args.request_rate_)
    assert num_prompts is not None and request_rate is not None

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    input_requests = None
    if args.dataset:
        # Get dataset from registry.
        input_requests = get_dataset(name=args.dataset,
                                     tokenizer=tokenizer,
                                     dataset_args=DatasetArgs(
                                         num_samples=num_prompts,
                                         max_len=2048,
                                         seed=42,
                                     ))
    else:
        # Make a synthetic dataset.
        input_requests = generate_synthetic_requests(args.num_input_tokens,
                                                     args.num_output_tokens,
                                                     num_prompts, tokenizer)

    metrics = asyncio.run(
        benchmark(backend=backend,
                  api_url=api_url,
                  model_id=model_id,
                  tokenizer=tokenizer,
                  input_requests=input_requests,
                  best_of=args.best_of,
                  use_beam_search=args.use_beam_search,
                  request_rate=request_rate,
                  disable_tqdm=args.disable_tqdm,
                  log_model_io=args.log_model_io))

    # Save config and results to json
    save_result = args.save_directory is not None
    if save_result:

        def script_args_as_json_dict(script_args: argparse.Namespace):
            # JSON dumps a float("inf") value as INFINITY (no double-quotes).
            # This makes the JSON invalid. The request rate argument can be a
            # float("int") - the fix is to always treat it as a string.
            import copy
            script_args = copy.deepcopy(script_args)
            if script_args.nr_qps_pair_:
                script_args.nr_qps_pair_ = (
                    script_args.nr_qps_pair_.num_prompts,
                    str(script_args.nr_qps_pair_.request_rate))
            if script_args.request_rate_:
                script_args.request_rate_ = str(script_args.request_rate_)
            return vars(script_args)

        current_dt = datetime.now()
        result = BenchmarkResult(
            date=current_dt,
            script_name=Path(__file__).name,
            script_args=script_args_as_json_dict(args),
            tensor_parallel_size=args.server_tensor_parallel_size,
            model=args.model,
            tokenizer=args.tokenizer,
            dataset=args.dataset)

        result = metrics.update_benchmark_result(result)

        # Add information about the derived variables as metadata
        result[BenchmarkResult.METADATA_KEY_][
            ResultMetadataKeys.num_prompts] = num_prompts
        result[BenchmarkResult.METADATA_KEY_][ResultMetadataKeys.request_rate] = \
            request_rate if request_rate < float("inf") else "inf"

        # Save to file
        base_model_id = model_id.split("/")[-1]
        current_dt_str = current_dt.strftime("%Y%m%d-%H%M%S")
        file_name = (
            Path(args.save_directory) /
            f"benchmark_serving-{backend}-{request_rate}qps-{base_model_id}-{current_dt_str}.json"
        )
        result.store(file_name)


class NumPrompts_RequestRate_T(NamedTuple):
    num_prompts: int
    request_rate: float

    @staticmethod
    def from_str(arg: str):
        # The arg_parser has a variant where num_prompts and request_rate can
        # passed in as a pair in the same argument.
        # Example: A string "1000,0.5" will be parsed into a tuple of
        # (int(1000), float(0.5))
        parts = arg.split(',')
        assert len(parts) == 2
        return NumPrompts_RequestRate_T(int(parts[0]), float(parts[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='''Benchmark the online serving throughput.''')
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="N/A",
        help="Version of the serving backend/engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--num-input-tokens",
                        type=int,
                        default=None,
                        help="Number of tokens in the input prompt")
    parser.add_argument("--num-output-tokens",
                        type=int,
                        default=None,
                        help="Number of generated tokens per prompt")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default model tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--log-model-io", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )

    parser.add_argument("--save-directory",
                        type=str,
                        default=None,
                        help="Output directory to store result file")

    # Arguments defining num_prompts and qps
    parser.add_argument(
        "--num-prompts_",
        type=int,
        default=None,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate_",
        type=lambda arg: float(arg),
        default=None,
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--nr-qps-pair_",
                        type=NumPrompts_RequestRate_T.from_str,
                        help="""
                            First argument in the pair is num_prompts: Number of prompts to process.
                            Second argument in the pair is request_rate : Number of requests per second. If this is inf,
                            then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize
                            the request arrival times.
                            """,
                        default=None)

    # Server command args
    parser.add_argument(
        "--server-tensor-parallel-size",
        type=int,
        default=None,
        help=
        "tensor-parallel-size that the benchmarking script was invoked with. It is useful to log this information when storing benchmarking results"
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default=None,
        help=
        "When we are logging the output, it is useful to log the arguments passed to the server"
    )

    def args_sanity_check(args):
        # Sanity check real-dataset vs synthetic-dataset usecase
        if args.dataset is None:
            assert args.num_input_tokens is not None and args.num_output_tokens is not None
        else:
            assert args.num_input_tokens is None and args.num_output_tokens is None
        # Sanity check num_prompts, request_rate as separate args vs joint args usecase
        assert not all([
            args.num_prompts_ is None, args.request_rate_ is None,
            args.nr_qps_pair_ is None
        ])
        if args.nr_qps_pair_ is None:
            assert args.num_prompts_ is not None and args.request_rate_ is not None
        else:
            assert args.num_prompts_ is None and args.request_rate_ is None
        # Sanity check required logging args
        if args.save_directory is not None:
            assert args.server_tensor_parallel_size is not None

    args = parser.parse_args()
    args_sanity_check(args)

    main(args)
