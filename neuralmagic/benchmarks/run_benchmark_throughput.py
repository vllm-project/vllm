import argparse
from pathlib import Path
from typing import NamedTuple, Optional

from .common import script_args_to_cla, benchmark_configs, max_model_length_from_model_id
from ..tools.call_cmd import call_cmd


def run_benchmark_throughput_script(config: NamedTuple,
                                    output_directory: Optional[Path] = None
                                    ) -> None:

    assert config.script_name == 'benchmark_throughput'

    script_path = f"neuralmagic.benchmarks.scripts.{config.script_name}"

    for model in config.models:

        supported_max_model_len = max_model_length_from_model_id(model)

        # If the requested model-len is too big, try running with the maximum supported for this model.
        max_model_lens = set(
            map(lambda v: min(v, supported_max_model_len),
                config.max_model_lens))
        if (config.max_model_lens != list(max_model_lens)):
            print(
                f"WARNING: max_model_len modified to {max_model_lens} from {config.max_model_lens} for model {model}"
            )

        for max_model_len in max_model_lens:
            for script_args in script_args_to_cla(config):
                bench_cmd = (["python3", "-m", f"{script_path}"] +
                             script_args + ["--model", f"{model}"] +
                             ["--tokenizer", f"{model}"] +
                             ["--max-model-len", f"{max_model_len}"])

                if output_directory:
                    bench_cmd = bench_cmd + [
                        "--save-directory", f"{output_directory}"
                    ]

                call_cmd(bench_cmd, stdout=None, stderr=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Runs the benchmark_throughput.py script as a subprocess")
    parser.add_argument(
        "-i",
        "--input-config-file",
        required=True,
        type=str,
        help="Path to the input config file describing the benhmarks to run",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default=None,
        help="Path to a directory that is the output store",
    )

    args = parser.parse_args()

    output_directory = Path(
        args.output_directory) if args.output_directory is not None else None

    for config in benchmark_configs(Path(args.input_config_file)):
        run_benchmark_throughput_script(config, output_directory)
