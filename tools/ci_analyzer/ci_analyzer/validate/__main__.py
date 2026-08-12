# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ci-validate: confidence tools for the analyzer (not the per-PR path).

Subcommands:
  crosscheck       replay real PRs, compare our selection vs actual CI outcomes
  dynamic-sites    prove the import graph has no unmodeled dynamic imports
  docs-refs        flag docs cross-references that no longer resolve
  uninvoked        list test files no live CI job invokes
  demoted-plugins  every demoted plugin still has test coverage
  dropped-edges    every deliberately-dropped edge still has another route
"""

from __future__ import annotations

import argparse

from . import (
    crosscheck,
    demoted_plugins,
    docs_refs,
    dropped_edges,
    dynamic_sites,
    uninvoked,
)

_COMMANDS = {
    "crosscheck": crosscheck,
    "dynamic-sites": dynamic_sites,
    "docs-refs": docs_refs,
    "uninvoked": uninvoked,
    "demoted-plugins": demoted_plugins,
    "dropped-edges": dropped_edges,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ci-validate",
        description="Confidence tools for the vLLM CI analyzer.",
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
