# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lean entrypoint for a headless vLLM engine (no API server).

`vllm serve --headless` builds the full OpenAI-server CLI schema
(tool/reasoning-parser plugins, MCP, chat templates, ...), none of which
`run_headless` reads. This builds a parser from `AsyncEngineArgs.add_cli_args`
only and calls the same `run_headless`, skipping that overhead.

Uses `FlexibleArgumentParser.parse_args` (not `parse_known_args`) so
`--config` files, dash/underscore normalization, and dotted-JSON options
behave exactly like `vllm serve`, and unknown args still error.

Not a public `vllm` subcommand; invoked via
`python -m vllm.entrypoints.cli.headless_engine` by `vllm-rs serve`'s managed
headless-engine mode (`rust/src/managed-engine`).
"""


def main(argv: list[str] | None = None) -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Launch a headless vLLM engine (no API server).",
        add_json_tip=False,
    )
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args(argv)

    # Headless mode never starts API servers.
    args.api_server_count = 0

    from vllm.entrypoints.cli.serve import run_headless

    run_headless(args)


if __name__ == "__main__":
    main()
