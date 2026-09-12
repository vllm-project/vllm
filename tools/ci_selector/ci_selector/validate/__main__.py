# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ci-validate: confidence tools for the analyzer (not the per-PR path).

Subcommands:
  crosscheck       replay real PRs, compare our selection vs actual CI outcomes

Everything checkable from a plain checkout is a drift-marked test instead:
`pytest tests -m drift`.
"""

from __future__ import annotations

import argparse

from . import crosscheck

_COMMANDS = {"crosscheck": crosscheck}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ci-validate",
        description="Confidence checks for the vLLM CI selector.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module in _COMMANDS.items():
        doc = (module.__doc__ or "").strip().splitlines()
        p = sub.add_parser(name, help=doc[0] if doc else None)
        module.add_args(p)
        p.set_defaults(_run=module.run)
    args = parser.parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
