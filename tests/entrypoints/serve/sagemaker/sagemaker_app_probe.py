# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report what `vllm serve` loads before and while it builds the app.

The tests need a process that has not imported `model_hosting_container_standards`
yet: the pytest process may have, because an earlier test built a server with the
integration enabled. The probe therefore runs in a fresh interpreter, imports the
module `vllm serve` starts from, and prints the state the build left behind.
Nothing is subtracted from that state, so a module-level import anywhere on the
startup path shows up in it.

Usage: `python sagemaker_app_probe.py [vllm serve args ...]`

Prints one JSON object: every module of that package in `sys.modules`, and the
middleware classes of that package installed on the app.
"""

import json
import sys

PACKAGE = "model_hosting_container_standards"


def main() -> None:
    # This is the module `vllm serve` starts from; importing it loads the rest
    # of the launch path, so its module-level imports are part of the check.
    import vllm.entrypoints.cli.serve  # noqa: F401
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.entrypoints.launchers.app import build_app
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    serve_parser = parser.add_subparsers().add_parser("serve")
    make_arg_parser(serve_parser)
    args = serve_parser.parse_args(sys.argv[1:])

    app = build_app(args, supported_tasks=())
    print(
        json.dumps(
            {
                "loaded": sorted(
                    name for name in sys.modules if name.startswith(PACKAGE)
                ),
                "middleware": [
                    middleware.cls.__module__ for middleware in app.user_middleware
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
