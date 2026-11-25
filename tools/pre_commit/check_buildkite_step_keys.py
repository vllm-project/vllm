# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that every Buildkite step the CI generator reads declares a key.

A step with no `key:` is published under one the generator derives from its
label, which makes the job's address a function of its title: retitling the
step silently renames it and breaks retries, `depends_on:` and build history.
The generator only reads the directories a pipeline config names in
`job_dirs:`, so this walks the same files and fails on any step without a key.

Usage:
    python tools/pre_commit/check_buildkite_step_keys.py
"""

import sys
from pathlib import Path

import yaml

BUILDKITE = Path(".buildkite")
CONFIG_GLOB = "ci_config*.yaml"
JOB_GLOB = "*.yaml"


def job_dirs() -> list[Path]:
    """Every directory the pipeline generator reads, per the pipeline configs."""
    configs = sorted(BUILDKITE.glob(CONFIG_GLOB))
    if not configs:
        raise SystemExit(f"no pipeline config matched {BUILDKITE / CONFIG_GLOB}")
    dirs: set[Path] = set()
    for config in configs:
        with open(config, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        declared = doc.get("job_dirs") if isinstance(doc, dict) else None
        if not isinstance(declared, list) or not declared:
            raise SystemExit(f"{config}: `job_dirs` is missing or not a list")
        for name in declared:
            job_dir = Path(name)
            if not job_dir.is_dir():
                raise SystemExit(f"{config}: job_dir `{name}` does not exist")
            dirs.add(job_dir)
    return sorted(dirs)


def scan(steps: object, source: Path) -> list[str]:
    """Label of every step in a `steps:` list that declares no key."""
    if not isinstance(steps, list):
        raise SystemExit(f"{source}: `steps:` is not a list")
    keyless: list[str] = []
    for i, step in enumerate(steps):
        # A scalar shorthand such as `- wait` has nowhere to put a key. Every
        # other step does, so there are no other exemptions. A `mirror:` block
        # is not a step here: the generator publishes it as `<hw>-<parent key>`.
        if not isinstance(step, dict):
            continue
        label = step.get("label") or step.get("group") or f"step {i}"
        if "steps" in step:
            raise SystemExit(
                f"{source}: `{label}` nests its own `steps:`. The generator has "
                "no group step and drops the children without an error, so give "
                "each one its own entry."
            )
        key = step.get("key")
        if not (isinstance(key, str) and key.strip()):
            keyless.append(label)
    return keyless


def keyless_steps(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "steps" not in doc:
        return []
    return scan(doc["steps"], path)


def main() -> int:
    files = sorted({p for d in job_dirs() for p in d.rglob(JOB_GLOB)})
    if not files:
        raise SystemExit("no pipeline files under the declared job_dirs")
    keyless = [(path, label) for path in files for label in keyless_steps(path)]
    if not keyless:
        return 0
    print(f"{len(keyless)} Buildkite step(s) have no explicit key:\n", file=sys.stderr)
    for path, label in keyless:
        print(f"  {path}: {label}", file=sys.stderr)
    print(
        "\nAdd a `key:` naming the test each step runs. Without one the pipeline"
        "\ngenerator derives the key from the label, so retitling the step renames"
        "\nits job and breaks retries and build history.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
