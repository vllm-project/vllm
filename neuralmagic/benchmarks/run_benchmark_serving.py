import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

import requests

from ..tools.call_cmd import call_cmd
from .common import (benchmark_configs, download_model,
                     max_model_length_from_model_id, script_args_to_cla)
from .scripts.common import num_available_gpus, warmup_server

BENCH_SERVER_HOST = "localhost"
BENCH_SERVER_PORT = 9000


def get_tensor_parallel_size(config: NamedTuple) -> int:

    num_tp_directives = [
        hasattr(config, 'tensor_parallel_size'),
        hasattr(config, 'use_all_available_gpus')
    ].count(True)
    if num_tp_directives == 0:
        # by default - use just one GPU
        return 1

    # must have exactly one directive
    assert num_tp_directives == 1

    tensor_parallel_size = config.tensor_parallel_size if hasattr(
        config, 'tensor_parallel_size') else num_available_gpus()
    assert tensor_parallel_size > 0 and \
           tensor_parallel_size <= num_available_gpus()
    return tensor_parallel_size


def is_server_running(host: str, port: int, timeout=600) -> bool:

    def try_connection() -> bool:
        try:
            r = requests.get(f"http://{host}:{port}/health")
            return r.status_code == 200
        except Exception as _:
            return False

    timeout_part = 15  # retry every 15 seconds
    time_waited = 0
    while time_waited <= timeout:
        time.sleep(timeout_part)
        if try_connection():
            return True
        time_waited = time_waited + timeout_part

    return False


def run_benchmark_serving_script(config: NamedTuple,
                                 output_directory: Optional[Path] = None
                                 ) -> None:
    assert config.script_name == 'benchmark_serving'

    def run_bench(server_cmd: str, bench_cmd: List[str], model: str) -> None:
        try:
            # start server
            server_process = subprocess.Popen("exec " + server_cmd, shell=True)
            if not is_server_running(BENCH_SERVER_HOST, BENCH_SERVER_PORT):
                raise ValueError(
                    f"Aborting bench run with : server-cmd {server_cmd} , "
                    f"bench-cmd {bench_cmd}. Reason: Cannot start Server")

            # server warmup
            warmup_server(server_host=BENCH_SERVER_HOST,
                          server_port=BENCH_SERVER_PORT,
                          model=model,
                          num_prompts=1000)

            # run bench
            call_cmd(bench_cmd, stdout=None, stderr=None)
        finally:
            # kill the server
            assert server_process is not None
            server_process.kill()

    tensor_parallel_size = get_tensor_parallel_size(config)

    script_path = f"neuralmagic.benchmarks.scripts.{config.script_name}"

    sparsities = [None] if len(config.sparsity) == 0 else config.sparsity

    for model, sparsity in itertools.product(config.models, sparsities):

        # download model beforehand so the server can start without any holdup
        download_model(model)

        supported_max_model_len = max_model_length_from_model_id(model)

        # If the requested model-len is too big, try running with the
        # maximum supported for this model.
        max_model_lens = set(
            map(lambda v: min(v, supported_max_model_len),
                config.max_model_lens))
        if (config.max_model_lens != list(max_model_lens)):
            print(f"WARNING: max_model_len modified to {max_model_lens} "
                  f"from {config.max_model_lens} for model {model}")

        for max_model_len in max_model_lens:

            server_args = {
                "model": model,
                "tokenizer": model,
                "max-model-len": max_model_len,
                "host": BENCH_SERVER_HOST,
                "port": BENCH_SERVER_PORT,
                "tensor-parallel-size": tensor_parallel_size,
                "disable-log-requests": ""
            }
            if sparsity:
                server_args["sparsity"] = sparsity

            server_cmd = "python3 -m vllm.entrypoints.api_server " + \
                            " ".join([f"--{k} {v}"
                                      for k, v in server_args.items()])

            for script_args in script_args_to_cla(config):

                description = (f"{config.description}\n" +
                               f"model - {model}\n" +
                               f"max-model-len - {max_model_len}\n" +
                               f"sparsity - {sparsity}\n" +
                               f"{config.script_name} " +
                               f"{json.dumps(script_args, indent=2)}")

                bench_cmd = (["python3", "-m"
                              f"{script_path}"] +
                             ["--description", f"{description}"] +
                             ["--model", f"{model}"] +
                             ["--tokenizer", f"{model}"] +
                             ["--port", f"{BENCH_SERVER_PORT}"] +
                             ["--host", f"{BENCH_SERVER_HOST}"])
                # Add script args
                for k, v in script_args.items():
                    bench_cmd.append(f"--{k}")
                    if v != "":
                        bench_cmd.append(f"{v}")

                if output_directory:
                    bench_cmd += (["--save-directory", f"{output_directory}"] +
                                  ["--server-args", f"{server_args}"] + [
                                      "--server-tensor-parallel-size",
                                      f"{tensor_parallel_size}"
                                  ])

                run_bench(server_cmd, bench_cmd, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Runs the benchmark_serving.py script as a subprocess")
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
        run_benchmark_serving_script(config, output_directory)
