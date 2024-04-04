import argparse
import json
from pathlib import Path
from typing import NamedTuple, Optional

from ..tools.call_cmd import call_cmd
from .common import (benchmark_configs, max_model_length_from_model_id,
                     script_args_to_cla)


def run_benchmark_throughput_script(config: NamedTuple,
                                    output_directory: Optional[Path] = None
                                    ) -> None:

    assert config.script_name == 'benchmark_throughput'

    script_path = f"neuralmagic.benchmarks.scripts.{config.script_name}"

    for model in config.models:

        supported_max_model_len = max_model_length_from_model_id(model)

        # If the requested model-len is too big, try running with
        # the maximum supported for this model.
        max_model_lens = set(
            map(lambda v: min(v, supported_max_model_len),
                config.max_model_lens))
        if (config.max_model_lens != list(max_model_lens)):
            print(f"WARNING: max_model_len modified to {max_model_lens} "
                  f"from {config.max_model_lens} for model {model}")

        for max_model_len in max_model_lens:
            for script_args in script_args_to_cla(config):

                description = (f"{config.description}\n"
                               f"model - {model}\n" +
                               f"max_model_len - {max_model_len}\n" +
                               f"{config.script_name} " +
                               f"{json.dumps(script_args, indent=2)}")

                bench_cmd = (["python3", "-m", f"{script_path}"] +
                             ["--description", f"{description}"] +
                             ["--model", f"{model}"] +
                             ["--tokenizer", f"{model}"] +
                             ["--max-model-len", f"{max_model_len}"])
                # Add script args
                for k, v in script_args.items():
                    bench_cmd.append(f"--{k}")
                    if v != "":
                        bench_cmd.append(f"{v}")

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
