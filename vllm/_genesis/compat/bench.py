# SPDX-License-Identifier: Apache-2.0
"""Genesis bench — unified-CLI shim for `tools/genesis_bench_suite.py`.

The real benchmark suite lives at `tools/genesis_bench_suite.py` (it
needs to ship as a single self-contained script people can curl, run,
and share without installing the full Genesis package). This module is
a thin pass-through that lets it be reached via the unified CLI:

    python3 -m vllm._genesis.compat.cli bench --quick
    python3 -m vllm._genesis.compat.cli bench --mode standard --ctx 8k
    python3 -m vllm._genesis.compat.cli bench --compare a.json b.json

All argv after the `bench` subcommand is forwarded verbatim to
`tools.genesis_bench_suite.main()`.

If the bench script can't be located on `sys.path` (e.g. running from a
deployed package without the source `tools/` directory), the shim
fails with a clear error pointing at `docs/BENCHMARK_GUIDE.md` rather
than a confusing import traceback.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger("genesis.compat.bench")


def _locate_bench_module() -> Path | None:
    """Find tools/genesis_bench_suite.py on disk.

    Search order:
      1. <repo-root>/tools/genesis_bench_suite.py (compat is at
         _genesis/compat/, parents[3] is the repo root)
      2. $GENESIS_REPO_ROOT/tools/genesis_bench_suite.py (operator
         override for slim deployments)
      3. cwd-relative tools/genesis_bench_suite.py
    """
    import os

    candidates: list[Path] = []
    env_root = os.environ.get("GENESIS_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "tools" / "genesis_bench_suite.py")
    candidates.append(
        Path(__file__).resolve().parents[3] / "tools" / "genesis_bench_suite.py"
    )
    candidates.append(Path.cwd() / "tools" / "genesis_bench_suite.py")
    return next((p for p in candidates if p.is_file()), None)


def _load_bench_module():
    """Import tools/genesis_bench_suite.py as a module.

    The script lives outside the package, so we use an explicit spec
    loader rather than `importlib.import_module`.
    """
    bench_path = _locate_bench_module()
    if bench_path is None:
        raise FileNotFoundError(
            "Could not locate tools/genesis_bench_suite.py — this "
            "deployment may not include the source `tools/` directory. "
            "Set GENESIS_REPO_ROOT to point at a checkout, or run the "
            "bench directly per docs/BENCHMARK_GUIDE.md."
        )
    spec = importlib.util.spec_from_file_location(
        "genesis_bench_suite", bench_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {bench_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Forward argv to genesis_bench_suite.main()."""
    if argv is None:
        argv = sys.argv[1:]

    # `--help` / `-h` should work even if the bench script can't be
    # located, so handle it before attempting the import.
    if argv and argv[0] in ("-h", "--help"):
        # Try to surface the real help; fall back to a stub.
        try:
            mod = _load_bench_module()
            return mod.main(["--help"])
        except (FileNotFoundError, ImportError) as e:
            print("genesis bench — Genesis Benchmark Suite (shim)")
            print()
            print("The full benchmark suite lives at "
                  "tools/genesis_bench_suite.py.")
            print()
            print(f"Could not load it from this deployment: {e}")
            print()
            print("See docs/BENCHMARK_GUIDE.md for a manual invocation "
                  "recipe.")
            return 0

    try:
        mod = _load_bench_module()
    except (FileNotFoundError, ImportError) as e:
        print(f"[genesis bench] {e}", file=sys.stderr)
        return 3

    return mod.main(argv)


if __name__ == "__main__":
    sys.exit(main())
