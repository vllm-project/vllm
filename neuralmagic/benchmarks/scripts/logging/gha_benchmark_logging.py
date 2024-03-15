"""
Scripts to process GHA benchmarking JSONs produced by BenchmarkResult
that could be consumed by `github-action-benchmark`.
Reference : https://github.com/benchmark-action/github-action-benchmark
"""
import argparse
import json
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import List, Iterable, NamedTuple

from .benchmark_result import (GHABenchmarkToolName, BenchmarkResult,
                               MetricTemplate)


@dataclass
class GHARecord:
    """
    GHARecord is what actually goes into the output JSON.
        - name : Chart title. Unique names map to a unique chart.
        - unit : Y-axis label.
        - value : Value to plot.
        - extra : Any extra information that is passed as a JSON string.
    """
    name: str
    unit: str
    value: float
    extra: str

    @staticmethod
    def extra_from_benchmark_result(br: BenchmarkResult) -> str:
        extra_as_dict = {
            BenchmarkResult.DESCRIPTION_KEY_:
            br.get(BenchmarkResult.DESCRIPTION_KEY_),
            BenchmarkResult.BENCHMARKING_CONTEXT_KEY_:
            br.get(BenchmarkResult.BENCHMARKING_CONTEXT_KEY_),
            BenchmarkResult.SCRIPT_NAME_KEY_:
            br.get(BenchmarkResult.SCRIPT_NAME_KEY_),
            BenchmarkResult.SCRIPT_ARGS_KEY_:
            br.get(BenchmarkResult.SCRIPT_ARGS_KEY_),
            BenchmarkResult.GPU_DESCRIPTION_KEY_:
            br.get(BenchmarkResult.GPU_DESCRIPTION_KEY_)
        }

        return f"{json.dumps(extra_as_dict, indent=2)}"

    @staticmethod
    def from_metric_template(metric_template: MetricTemplate, extra: str = ""):
        return GHARecord(
            name=f"{metric_template.key} ({metric_template.unit})",
            unit=metric_template.unit,
            value=metric_template.value,
            extra=extra)


class Tool_Record_T(NamedTuple):
    tool: GHABenchmarkToolName
    record: GHARecord


def process(json_file_path: Path) -> Iterable[Tool_Record_T]:

    assert json_file_path.exists()

    json_data: dict = None
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    assert json_data is not None

    print(f"processing file : {json_file_path}")

    hover_data = GHARecord.extra_from_benchmark_result(json_data)
    metrics: Iterable[dict] = json_data.get(BenchmarkResult.METRICS_KEY_)
    metrics: Iterable[MetricTemplate] = map(
        lambda md: MetricTemplate.from_dict(md), metrics.values())

    return map(
        lambda metric: Tool_Record_T(
            metric.tool,
            GHARecord.from_metric_template(metric, extra=hover_data)), metrics)


def main(input_directory: Path, bigger_is_better_output_json_file_name: Path,
         smaller_is_better_output_json_file_name: Path) -> None:

    def dump_to_json(gha_records: List[GHARecord], output_path: Path):
        # Make output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make data JSON serializable
        gha_record_dicts = list(map(lambda x: x.__dict__, gha_records))
        with open(output_path, 'w+') as f:
            json.dump(gha_record_dicts, f, indent=4)

    json_file_paths = input_directory.glob('*.json')
    tool_records: List[Tool_Record_T] = list(
        reduce(lambda whole, part: whole + part,
               (map(lambda json_file_path: list(process(json_file_path)),
                    json_file_paths))))

    bigger_is_better: List[GHARecord] = list(
        map(
            lambda tool_record: tool_record.record,
            filter(
                lambda tool_record: tool_record.tool == GHABenchmarkToolName.
                BiggerIsBetter, tool_records)))

    smaller_is_better: List[GHARecord] = list(
        map(
            lambda tool_record: tool_record.record,
            filter(
                lambda tool_record: tool_record.tool == GHABenchmarkToolName.
                SmallerIsBetter, tool_records)))

    dump_to_json(bigger_is_better, bigger_is_better_output_json_file_name)
    dump_to_json(smaller_is_better, smaller_is_better_output_json_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Process the benchmark JSONs produced by BenchmarkResult and output JSONs
        that could be consumed by `github-action-benchmark`
        Reference : https://github.com/benchmark-action/github-action-benchmark
        """)

    parser.add_argument(
        "-i",
        "--input-json-directory",
        required=True,
        type=str,
        help="""Path to the directory containing BenchmarkResult 
                jsons. This is typically the output directory passed 
                to the benchmark runner scripts like 
                neuralmagic/benchmarks/run_benchmarks.py.""")

    parser.add_argument(
        "--bigger-is-better-output-file-path",
        type=str,
        required=True,
        help="""An output file path, where the GHABenchmarkToolName 
                BiggerIsBetter metrics are to be stored.""")

    parser.add_argument(
        "--smaller-is-better-output-file-path",
        type=str,
        required=True,
        help="""An output file path, where the GHABenchmarkToolName 
                SmallerIsBetter metrics are to be stored""")

    args = parser.parse_args()

    main(Path(args.input_json_directory),
         Path(args.bigger_is_better_output_file_path),
         Path(args.smaller_is_better_output_file_path))
