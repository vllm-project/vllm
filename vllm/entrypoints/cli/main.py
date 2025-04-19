# SPDX-License-Identifier: Apache-2.0

# The CLI entrypoint to vLLM.
import signal
import sys

import vllm.entrypoints.cli.benchmark.main
import vllm.entrypoints.cli.collect_env
import vllm.entrypoints.cli.openai
import vllm.entrypoints.cli.serve
import vllm.version
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser

CMD_MODULES = [
    vllm.entrypoints.cli.openai,
    vllm.entrypoints.cli.serve,
    vllm.entrypoints.cli.benchmark.main,
    vllm.entrypoints.cli.collect_env,
]


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def main():
    cli_env_setup()

    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=vllm.version.__version__)
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
