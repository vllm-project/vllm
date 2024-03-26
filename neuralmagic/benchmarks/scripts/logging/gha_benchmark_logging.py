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

from .benchmark_result import (BenchmarkMetricType, BenchmarkResult,
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
    def extra_from_benchmark_result(br: BenchmarkResult) -> dict:
        extra = {
            BenchmarkResult.DESCRIPTION_KEY_:
            br.get(BenchmarkResult.DESCRIPTION_KEY_),
            BenchmarkResult.BENCHMARKING_CONTEXT_KEY_:
            br.get(BenchmarkResult.BENCHMARKING_CONTEXT_KEY_),
            BenchmarkResult.GPU_DESCRIPTION_KEY_:
            br.get(BenchmarkResult.GPU_DESCRIPTION_KEY_),
            BenchmarkResult.SCRIPT_NAME_KEY_:
            br.get(BenchmarkResult.SCRIPT_NAME_KEY_),
            BenchmarkResult.SCRIPT_ARGS_KEY_:
            br.get(BenchmarkResult.SCRIPT_ARGS_KEY_),
            BenchmarkResult.DATE_KEY_:
            br.get(BenchmarkResult.DATE_KEY_),
            BenchmarkResult.MODEL_KEY_:
            br.get(BenchmarkResult.MODEL_KEY_),
            BenchmarkResult.DATASET_KEY_:
            br.get(BenchmarkResult.DATASET_KEY_)
        }
        return extra

    @staticmethod
    def from_metric_template(metric_template: MetricTemplate, extra: dict):
        # Unique names map to unique charts / benchmarks. Pass it as a JSON
        # string with enough information so we may deconstruct it at the UI.
        # TODO (varun) : Convert all additional information in name into a hash
        # if this becomes too cumbersome.
        benchmarking_context = \
                extra.get(BenchmarkResult.BENCHMARKING_CONTEXT_KEY_)
        name = {
            "name":
            metric_template.key,
            BenchmarkResult.DESCRIPTION_KEY_:
            extra.get(BenchmarkResult.DESCRIPTION_KEY_),
            BenchmarkResult.GPU_DESCRIPTION_KEY_:
            extra.get(BenchmarkResult.GPU_DESCRIPTION_KEY_),
            "vllm_version":
            benchmarking_context.get("vllm_version"),
            "python_version":
            benchmarking_context.get("python_version"),
            "torch_version":
            benchmarking_context.get("torch_version")
        }

        return GHARecord(name=f"{json.dumps(name)}",
                         unit=metric_template.unit,
                         value=metric_template.value,
                         extra=f"{json.dumps(extra, indent=2)}")


class Type_Record_T(NamedTuple):
    type: BenchmarkMetricType
    record: GHARecord


def process(json_file_path: Path) -> Iterable[Type_Record_T]:

    assert json_file_path.exists()

    json_data: dict = None
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    assert json_data is not None

    print(f"processing file : {json_file_path}")

    hover_data: dict = GHARecord.extra_from_benchmark_result(json_data)
    metrics: Iterable[dict] = json_data.get(BenchmarkResult.METRICS_KEY_)
    metrics: Iterable[MetricTemplate] = map(
        lambda md: MetricTemplate.from_dict(md), metrics.values())

    return map(
        lambda metric: Type_Record_T(
            metric.type,
            GHARecord.from_metric_template(metric, extra=hover_data)), metrics)


def main(args: argparse.Namespace) -> None:
    input_directory = Path(args.input_directory)

    json_file_paths = input_directory.glob('*.json')

    type_records: List[Type_Record_T] = list(
        reduce(lambda whole, part: whole + part,
               (map(lambda json_file_path: list(process(json_file_path)),
                    json_file_paths))))

    def filter_and_dump_if_non_empty(type_records: List[Type_Record_T],
                                     type: BenchmarkMetricType,
                                     output_path: Path):
        """
        Given a list of type_record tuples, filter the records with the given
        type.
        If there are no records after we filter, don't dump json. otherwise,
        dump all records as JSON.
        """
        # Make output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        gha_records: List[GHARecord] = list(
            map(
                lambda type_record: type_record.record,
                filter(lambda type_record: type_record.type == type,
                       type_records)))

        if len(gha_records) == 0:
            return

        # Make data JSON serializable
        gha_record_dicts = list(map(lambda x: x.__dict__, gha_records))
        with open(output_path, 'w+') as f:
            json.dump(gha_record_dicts, f, indent=4)

    filter_and_dump_if_non_empty(
        type_records, BenchmarkMetricType.BiggerIsBetter,
        Path(args.bigger_is_better_metrics_output_file_path))
    filter_and_dump_if_non_empty(
        type_records, BenchmarkMetricType.SmallerIsBetter,
        Path(args.smaller_is_better_metrics_output_file_path))
    filter_and_dump_if_non_empty(
        type_records, BenchmarkMetricType.Observation,
        Path(args.observation_metrics_output_file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Process the benchmark JSONs produced by BenchmarkResult and output JSONs
        that could be consumed by `github-action-benchmark`.
        The JSONs are not produced if there are no metrics to report for some
        BenchmarkMetricType.
        Reference : https://github.com/benchmark-action/github-action-benchmark
        """)

    parser.add_argument(
        "-i",
        "--input-directory",
        required=True,
        type=str,
        help="""Path to the directory containing BenchmarkResult 
                jsons. This is typically the output directory passed 
                to the benchmark runner scripts like 
                neuralmagic/benchmarks/run_benchmarks.py.""")

    parser.add_argument("--bigger-is-better-metrics-output-file-path",
                        required=True,
                        type=str,
                        help="""
        An output file path, where the BenchmarkMetricType
        BiggerIsBetter metrics are stored.
        """)

    parser.add_argument("--smaller-is-better-metrics-output-file-path",
                        required=True,
                        type=str,
                        help="""
        An output file path, where the BenchmarkMetricType
        SmallerIsBetter metrics are stored.
        """)

    parser.add_argument("--observation-metrics-output-file-path",
                        required=True,
                        type=str,
                        help="""
        An output file path, where the BenchmarkMetricType
        Observation metrics are stored.
        """)

    args = parser.parse_args()

    main(args)
