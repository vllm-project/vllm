# SPDX-License-Identifier: Apache-2.0
import argparse
import sys

from entrypoints.entrypoint_benchmark import EntrypointBenchmark
from entrypoints.entrypoint_server import EntrypointServer
from entrypoints.entrypoint_test import EntrypointTest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EntrypointMain for vllm docker")
    parser.add_argument(
        "mode",
        nargs="?",
        default="server",
        choices=["server", "benchmark", "test"],
        help="Mode to run: server, benchmark, or test",
    )
    parser.add_argument("--config-file", type=str, help="Path to config file")
    parser.add_argument("--config-name",
                        type=str,
                        help="Config name in the config file")
    args = parser.parse_args()

    if args.mode == "server":
        entrypoint = EntrypointServer(
            config_file=args.config_file,
            config_name=args.config_name,
        )
    elif args.mode == "benchmark":
        entrypoint = EntrypointBenchmark(
            config_file=args.config_file,
            config_name=args.config_name,
        )
    elif args.mode == "test":
        entrypoint = EntrypointTest(
            config_file=args.config_file,
            config_name=args.config_name,
        )
    else:
        print(f"[ERROR] Unknown mode '{args.mode}'. Use 'server', "
              "'benchmark' or 'test'.")
        sys.exit(1)

    entrypoint.run()