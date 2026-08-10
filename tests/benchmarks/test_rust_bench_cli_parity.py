# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the `vllm bench serve` flag snapshot in sync for the Rust CLI parity test.

`rust/src/bench/tests/cli_parity.rs` asserts that every flag in the snapshot is
either a known `vllm-bench` clap flag/alias or explicitly allowlisted as
Python-only, so `VLLM_USE_RUST_BENCH=1` delegation cannot silently reject
documented flags. This test keeps the snapshot itself current.

Run this file directly to regenerate the snapshot:

    python tests/benchmarks/test_rust_bench_cli_parity.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "rust" / "src" / "bench" / "tests" / "python_serve_flags.txt"
HEADER = """\
# Long CLI flags of the Python `vllm bench serve` parser
# (vllm.benchmarks.serve.add_cli_args), sorted; --help excluded.
# Regenerate: python tests/benchmarks/test_rust_bench_cli_parity.py
# Consumed by rust/src/bench/tests/cli_parity.rs, which requires every flag
# below to be a known vllm-bench flag/alias or a PYTHON_ONLY allowlist entry.
"""


def _current_flags() -> list[str]:
    from vllm.benchmarks.serve import add_cli_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    return sorted(
        {
            opt
            for action in parser._actions
            for opt in action.option_strings
            if opt.startswith("--") and opt != "--help"
        }
    )


def _snapshot_flags() -> list[str]:
    lines = SNAPSHOT.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_serve_flag_snapshot_is_current():
    assert SNAPSHOT.is_file(), f"missing snapshot {SNAPSHOT}"
    assert _current_flags() == _snapshot_flags(), (
        "`vllm bench serve` flags changed. Regenerate the snapshot with "
        "`python tests/benchmarks/test_rust_bench_cli_parity.py`, then for any "
        "added flag either support it in rust/src/bench/src/cli.rs or add it "
        "to PYTHON_ONLY in rust/src/bench/tests/cli_parity.rs"
    )


if __name__ == "__main__":
    SNAPSHOT.write_text(HEADER + "\n".join(_current_flags()) + "\n")
    print(f"wrote {SNAPSHOT}")
