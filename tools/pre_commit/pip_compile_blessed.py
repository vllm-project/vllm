# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blessed-hash gate around `uv pip compile` (the pip-compile hooks).

The naive hook re-resolves the lockfile on every run, so every CI push
needs live index access — and when an index host degrades, the gate goes
red for reasons that have nothing to do with our files.

This wrapper makes the gate honest under vendor degradation:

- Offline fast path: a sha256 over the hook args, the source .in plus
  everything it `-r`/`-c`-includes (transitively), every `-c` constraint,
  every `--override` file, `pyproject.toml` ([tool.uv] is a resolution
  input), and the current lockfile is compared against the blessed store
  (requirements/.pip-compile-blessed.json, committed). Match -> exit 0
  without touching the network.
- Fail-closed on drift: on a hash miss the wrapper resolves. If the
  resolution rewrites the lockfile, pre-commit reports the modification
  and the hook fails — a red gate means our files drifted.
- Fail-open on vendor outage (default): if the resolve fails with a
  network-class error (index unreachable), the wrapper warns and exits 0.
  This defers the drift check to the scheduled drift workflow, which runs
  in strict mode and cannot defer: a resolve that stays unreachable reds
  it, so a persistent failure is never silently green.

PIP_COMPILE_BLESSED=strict skips the fast path and fails closed on
network-class errors — the scheduled drift workflow sets this so its
re-resolve is unconditional and honest. Any other non-empty value skips
the fast path only.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = 2
STORE = Path("requirements/.pip-compile-blessed.json")
FORCE_ENV = "PIP_COMPILE_BLESSED"

# argv tokens that take a value. Only -o/-c/--override plus the source
# positional matter for hashing; the rest just needs to be skipped
# correctly. Every value-taking flag a hook uses MUST be listed here —
# an unlisted value-flag's value would be parsed as the source positional
# and trip the second-source guard below. Flags not listed are skipped
# as booleans (one token).
VALUE_FLAGS = {
    "-o",
    "-c",
    "--override",
    "--no-emit-package",
    "--index-strategy",
    "--torch-backend",
    "--python-platform",
    "--python-version",
    "--annotation-style",
    "--extra-index-url",
    "--index-url",
}

# uv failure text that means "the index host is unreachable", not "the
# lockfile is wrong". Matched case-insensitively against the combined
# output.
NETWORK_PATTERNS = (
    "failed to fetch",
    "http status server error",
    "connection error",
    "connection refused",
    "connection reset",
    "timed out",
    "error sending request",
    "temporary failure",
    "no such host",
)


def parse_args(
    argv: list[str],
) -> tuple[str, str, list[str], list[str]]:
    src = out = None
    constraints: list[str] = []
    overrides: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-o", "-c", "--override"):
            if i + 1 >= len(argv):
                print(
                    f"pip-compile-blessed: {arg} is missing its value in {argv!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if arg == "-o":
                out = argv[i + 1]
            elif arg == "-c":
                constraints.append(argv[i + 1])
            else:
                overrides.append(argv[i + 1])
            i += 2
        elif arg.startswith("-"):
            # Value-taking flags are enumerated; anything else is a
            # boolean and takes one token.
            i += 2 if arg in VALUE_FLAGS else 1
        elif src is None:
            src = arg
            i += 1
        else:
            print(
                f"pip-compile-blessed: unexpected second source {arg!r} in "
                f"{argv!r} — a value-taking flag may be missing from "
                "VALUE_FLAGS",
                file=sys.stderr,
            )
            sys.exit(1)
    if src is None or out is None:
        print(
            f"pip-compile-blessed: cannot parse -o/<source> out of {argv!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    return src, out, constraints, overrides


def content_or_missing(path: str) -> str:
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return "<missing>"


def collect_included_files(path: str, seen: set[str]) -> list[str]:
    """Files pulled in by -r/--requirement and -c/--constraint directives.

    uv inlines these during resolution, so their content is a resolution
    input even though it never appears in the hook argv. Paths are
    resolved relative to the including file and normalized against the
    repo root (the hook's working directory).
    """
    includes: list[str] = []
    if path in seen:
        return includes
    seen.add(path)
    try:
        text = Path(path).read_text()
    except OSError:
        return includes
    base = Path(path).parent
    for line in text.splitlines():
        directive = line.split("#", 1)[0].strip()
        parts = directive.split()
        if len(parts) != 2 or parts[0] not in (
            "-r",
            "--requirement",
            "-c",
            "--constraint",
        ):
            continue
        target = os.path.normpath(base / parts[1])
        includes.append(target)
        includes.extend(collect_included_files(target, seen))
    return includes


def compute_hash(
    argv: list[str],
    src: str,
    out: str,
    constraints: list[str],
    overrides: list[str],
) -> str:
    h = hashlib.sha256()
    h.update(f"v{VERSION}\0".encode())
    h.update(json.dumps(argv).encode())
    seen: set[str] = set()
    src_includes = collect_included_files(src, seen)
    constraint_includes = [
        inc
        for constraint in constraints
        for inc in collect_included_files(constraint, seen)
    ]
    override_includes = [
        inc for override in overrides for inc in collect_included_files(override, seen)
    ]
    inputs = [
        src,
        *src_includes,
        *constraints,
        *constraint_includes,
        *overrides,
        *override_includes,
    ]
    if Path("pyproject.toml").exists():
        # uv reads [tool.uv] from the working directory during resolution.
        inputs.append("pyproject.toml")
    inputs.append(out)
    for path in inputs:
        h.update(f"{path}\0{content_or_missing(path)}\0".encode())
    return h.hexdigest()


def load_store() -> dict:
    try:
        return json.loads(STORE.read_text())
    except FileNotFoundError:
        return {}


def is_network_error(output: str) -> bool:
    lowered = output.lower()
    return any(pattern in lowered for pattern in NETWORK_PATTERNS)


def main() -> int:
    argv = sys.argv[1:]
    src, out, constraints, overrides = parse_args(argv)
    digest = compute_hash(argv, src, out, constraints, overrides)
    store = load_store()
    mode = os.environ.get(FORCE_ENV, "").strip().lower()
    strict = mode == "strict"
    skip_fast_path = bool(mode)

    if not skip_fast_path and store.get(out) == digest:
        print(f"pip-compile-blessed: {out} matches its blessed resolution (offline)")
        return 0

    uv = shutil.which("uv")
    if uv is None:
        print(
            "pip-compile-blessed: uv not found in the hook env; cannot resolve "
            f"{out}. Install uv into the hook (additional_dependencies) or "
            "bless the store.",
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [uv, "pip", "compile", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if is_network_error(output):
            if strict:
                print(
                    f"pip-compile-blessed: index unreachable while resolving "
                    f"{out} — strict mode fails closed so the scheduled "
                    "drift workflow can retry and report the outage.",
                    file=sys.stderr,
                )
                return result.returncode
            print(
                f"pip-compile-blessed: WARNING index unreachable while "
                f"resolving {out} — drift check deferred, gate stays green. "
                "The scheduled drift workflow runs in strict mode and will "
                "red while the index stays unreachable.",
                file=sys.stderr,
            )
            return 0
        print(output, file=sys.stderr)
        return result.returncode

    # Resolution succeeded: bless the post-resolve state. pre-commit fails
    # the hook on its own when the lockfile (or the store) changed, which
    # is the drift-fix contract. Echo the captured resolution log so a
    # hash-miss resolve is diagnosable.
    if output.strip():
        print(output, end="" if output.endswith("\n") else "\n")
    new_digest = compute_hash(argv, src, out, constraints, overrides)
    if store.get(out) != new_digest:
        store[out] = new_digest
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
