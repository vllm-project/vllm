# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report what `build_app` leaves on the server, from a fresh interpreter.

The import check is only conclusive in a process that has not imported
`model_hosting_container_standards` yet, so the tests run this script instead
of building the app in the pytest process, where an earlier test may have
imported it already.

Usage: `python sagemaker_app_probe.py [vllm serve args ...]`

Prints one JSON object with the modules of that package the build imported and
the middleware classes of that package it installed.
"""

import json
import sys

PACKAGE = "model_hosting_container_standards"


def main() -> None:
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.entrypoints.launchers.app import build_app
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    serve_parser = parser.add_subparsers().add_parser("serve")
    make_arg_parser(serve_parser)
    args = serve_parser.parse_args(sys.argv[1:])

    imported_before = {name for name in sys.modules if name.startswith(PACKAGE)}
    app = build_app(args, supported_tasks=())
    print(
        json.dumps(
            {
                "imported": sorted(
                    {name for name in sys.modules if name.startswith(PACKAGE)}
                    - imported_before
                ),
                "middleware": [
                    middleware.cls.__module__ for middleware in app.user_middleware
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
