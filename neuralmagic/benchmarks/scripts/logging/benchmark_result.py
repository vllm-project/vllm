"""
Defines a BenchmarkResult class that all the benchmarks use to save results.
"""

import json

from vllm import __version__ as __vllm_version__
from typing import Optional
from types import SimpleNamespace
from pathlib import Path
from ..common import get_benchmarking_context
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# NOTE - PLEASE READ:
# Any modifications that adds/removes the keys in the JSON that BenchmarkResult
# produces should also update the BENCHMARK_RESULTS_SCHEMA_VERSION.
# The primary use case is to establish a set of keys that can be queried.
# TODO (varun) : Initial version is named 0.0.0 as things are under development.
# Update it when things are stable.
BENCHMARK_RESULTS_SCHEMA_VERSION = "0.0.0"


class BenchmarkMetricType(str, Enum):
    # Metrics that are "better" when the value is greater e.g. throughput.
    BiggerIsBetter = "BiggerIsBetter"
    # Metrics that are "better" when the value is smaller e.g. latency.
    SmallerIsBetter = "SmallerIsBetter"
    # Metrics that are too volatile and we primarily use for observation.
    Observation = "Observation"


@dataclass
class MetricTemplate:
    key: str = field(default=None)
    unit: str = field(default=None)
    value: float = field(default=None)
    type: BenchmarkMetricType = field(default=None)

    def from_dict(d: dict):
        template: MetricTemplate = MetricTemplate()
        for key in d:
            setattr(template, key, d.get(key))
        return template


BenchmarkServingResultMetadataKeys = SimpleNamespace(
    completed="completed",
    duration="duration",
    total_input="total_input",
    total_output="total_output",
    num_prompts="num_prompts",
    request_rate="request_rate")

BenchmarkServingResultMetricTemplates = SimpleNamespace(
    request_throughput=MetricTemplate("request_throughput", "prompts/s", None,
                                      BenchmarkMetricType.BiggerIsBetter),
    input_throughput=MetricTemplate("input_throughput", "tokens/s", None,
                                    BenchmarkMetricType.BiggerIsBetter),
    output_throughput=MetricTemplate("output_throughput", "tokens/s", None,
                                     BenchmarkMetricType.BiggerIsBetter),
    median_request_latency=MetricTemplate("median_request_latency", "ms", None,
                                          BenchmarkMetricType.SmallerIsBetter),
    p90_request_latency=MetricTemplate("p90_request_latency", "ms", None,
                                       BenchmarkMetricType.SmallerIsBetter),
    p99_request_latency=MetricTemplate("p99_request_latency", "ms", None,
                                       BenchmarkMetricType.SmallerIsBetter),
    mean_ttft_ms=MetricTemplate("mean_ttft_ms", "ms", None,
                                BenchmarkMetricType.SmallerIsBetter),
    median_ttft_ms=MetricTemplate("median_ttft_ms", "ms", None,
                                  BenchmarkMetricType.SmallerIsBetter),
    p90_ttft_ms=MetricTemplate("p90_ttft_ms", "ms", None,
                               BenchmarkMetricType.SmallerIsBetter),
    p99_ttft_ms=MetricTemplate("p99_ttft_ms", "ms", None,
                               BenchmarkMetricType.SmallerIsBetter),
    mean_tpot_ms=MetricTemplate("mean_tpot_ms", "ms", None,
                                BenchmarkMetricType.SmallerIsBetter),
    median_tpot_ms=MetricTemplate("median_tpot_ms", "ms", None,
                                  BenchmarkMetricType.SmallerIsBetter),
    p90_tpot_ms=MetricTemplate("p90_tpot_ms", "ms", None,
                               BenchmarkMetricType.SmallerIsBetter),
    p99_tpot_ms=MetricTemplate("p99_tpot_ms", "ms", None,
                               BenchmarkMetricType.SmallerIsBetter))

BenchmarkThroughputResultMetricTemplates = SimpleNamespace(
    request_throughput=MetricTemplate("request_throughput", "prompts/s", None,
                                      BenchmarkMetricType.BiggerIsBetter),
    token_throughput=MetricTemplate("token_throughput", "tokens/s", None,
                                    BenchmarkMetricType.BiggerIsBetter))


class BenchmarkResult:

    BENCHMARK_RESULT_SCHEMA_VERSION_KEY_ = "json_schema_version"
    VLLM_VERSION_KEY_ = "vllm_version"
    METADATA_KEY_ = "metadata"
    METRICS_KEY_ = "metrics"
    DESCRIPTION_KEY_ = "description"
    GPU_DESCRIPTION_KEY_ = "gpu_description"
    DATE_KEY_ = "date"
    DATE_EPOCH_KEY_ = "epoch_time"
    SCRIPT_NAME_KEY_ = "script_name"
    SCRIPT_ARGS_KEY_ = "script_args"
    TENSOR_PARALLEL_SIZE_KEY_ = "tensor_parallel_size"
    MODEL_KEY_ = "model"
    TOKENIZER_KEY_ = "tokenizer"
    DATASET_KEY_ = "dataset"
    BENCHMARKING_CONTEXT_KEY_ = "benchmarking_context"

    @staticmethod
    def datetime_as_string(date: datetime):
        return date.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    @staticmethod
    def describe_gpu(bench_ctx: dict, num_gpus_used: int) -> str:
        """
        Return a string that describes the gpus used in benchmarking
        """
        cuda_device_names_key = "cuda_device_names"
        gpu_names = bench_ctx.get(cuda_device_names_key)
        assert gpu_names is not None
        gpu_name = gpu_names[0]

        # Make sure all gpus are the same before we report.
        assert all(map(lambda x: x == gpu_name, gpu_names[:num_gpus_used]))

        return f"{gpu_name} x {num_gpus_used}"

    def __init__(self, description: str, date: datetime, script_name: str,
                 script_args: dict, tensor_parallel_size: int, model: str,
                 tokenizer: Optional[str], dataset: Optional[str]):

        bench_ctx = get_benchmarking_context()

        # TODO (varun) Add githash
        self.result_dict = {
            self.BENCHMARK_RESULT_SCHEMA_VERSION_KEY_:
            BENCHMARK_RESULTS_SCHEMA_VERSION,
            self.VLLM_VERSION_KEY_:
            __vllm_version__,
            self.BENCHMARKING_CONTEXT_KEY_:
            bench_ctx,
            self.DESCRIPTION_KEY_:
            description,
            self.GPU_DESCRIPTION_KEY_:
            BenchmarkResult.describe_gpu(bench_ctx, tensor_parallel_size),
            self.DATE_KEY_:
            BenchmarkResult.datetime_as_string(date),
            self.DATE_EPOCH_KEY_:
            date.timestamp(),
            self.SCRIPT_NAME_KEY_:
            script_name,
            self.TENSOR_PARALLEL_SIZE_KEY_:
            tensor_parallel_size,
            self.MODEL_KEY_:
            model,
            self.TOKENIZER_KEY_:
            tokenizer if tokenizer is not None else model,
            self.DATASET_KEY_:
            dataset if dataset is not None else "synthetic",
            self.SCRIPT_ARGS_KEY_:
            script_args,
            # Any metadata that the caller script wants to store.
            self.METADATA_KEY_: {},
            # Any benchmarking metrics should be stored here.
            self.METRICS_KEY_: {}
        }

    def __setitem__(self, key: str, item: any):
        self.result_dict[key] = item

    def __getitem__(self, key: str, default: any = None) -> any:
        return self.result_dict.get(key, default)

    def add_metric(self, metric_template: MetricTemplate,
                   value: float) -> None:
        metric_template.value = value
        self.result_dict[self.METRICS_KEY_][
            metric_template.key] = metric_template.__dict__

    def store(self, store_path: Path) -> None:
        with open(store_path, "w") as outfile:
            json.dump(self.result_dict, outfile, indent=4)
