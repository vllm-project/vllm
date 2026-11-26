# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that every Buildkite step the CI generator reads declares a key.

A step with no `key:` is published under one the generator derives from its
label, which makes the job's address a function of its title: retitling the
step silently renames it and breaks retries, `depends_on:` and build history.
The generator only reads the directories a pipeline config names in
`job_dirs:`, so this walks the same files and fails on any step without a key.

A step written with `devices:` is one job per device (and per name under a
device's `variants:`), each keyed from the step's `key` the way ci-infra's
pipeline generator derives it (see `device_job_keys`). Those keys are checked
too: two jobs on one key, in one step or across files, fail.

Usage:
    python tools/pre_commit/check_buildkite_step_keys.py
"""

import sys
from pathlib import Path

import yaml

BUILDKITE = Path(".buildkite")
CONFIG_GLOB = "ci_config*.yaml"
JOB_GLOB = "*.yaml"


def pipelines() -> dict[Path, list[Path]]:
    """The directories each pipeline config has the generator read."""
    configs = sorted(BUILDKITE.glob(CONFIG_GLOB))
    if not configs:
        raise SystemExit(f"no pipeline config matched {BUILDKITE / CONFIG_GLOB}")
    found: dict[Path, list[Path]] = {}
    for config in configs:
        with open(config, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        declared = doc.get("job_dirs") if isinstance(doc, dict) else None
        if not isinstance(declared, list) or not declared:
            raise SystemExit(f"{config}: `job_dirs` is missing or not a list")
        dirs = []
        for name in declared:
            job_dir = Path(name)
            if not job_dir.is_dir():
                raise SystemExit(f"{config}: job_dir `{name}` does not exist")
            dirs.append(job_dir)
        found[config] = dirs
    return found


def job_dirs() -> list[Path]:
    """Every directory the pipeline generator reads, per the pipeline configs."""
    return sorted({d for dirs in pipelines().values() for d in dirs})


def scan(steps: object, source: Path) -> list[str]:
    """Label of every step in a `steps:` list that declares no key."""
    if not isinstance(steps, list):
        raise SystemExit(f"{source}: `steps:` is not a list")
    keyless: list[str] = []
    for i, step in enumerate(steps):
        # A scalar shorthand such as `- wait` has nowhere to put a key. Every
        # other step does, so there are no other exemptions. A `mirror:` block
        # is not a step here: the generator publishes it as `<hw>-<parent key>`.
        # The jobs of a `devices:` step are keyed from the step's key.
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


def device_key_problems(steps: object) -> list[str]:
    """What is wrong with the job keys of each `devices:` step in a list."""
    problems: list[str] = []
    for step in steps if isinstance(steps, list) else []:
        if not (isinstance(step, dict) and "devices" in step):
            continue
        if not (isinstance(step.get("key"), str) and step["key"].strip()):
            continue  # reported as keyless
        label = step.get("label") or step["key"]
        problems += [
            f"{label} ({job}): {problem}"
            for job, _, problem in device_job_keys(step)
            if problem
        ]
    return problems


def _amd_family(device: str) -> tuple[str, str] | None:
    """(family, pool size) of an AMD device: `mi355_4` -> (`mi355`, `4`),
    `mi300_dpx` and `mi300_dpx_2` -> (`mi300`, `dpx`), `r9700_1` -> (`r9700`,
    `1`)."""
    family, sep, size = device.partition("_")
    letters = family.rstrip("0123456789")
    dpx, _, dpx_count = size.partition("_")
    if dpx == "dpx" and dpx_count.isdigit():
        size = dpx
    if (
        sep
        and letters in ("mi", "r")
        and family[len(letters) :]
        and (size.isdigit() or size == "dpx")
    ):
        return family, size
    return None


def device_vendor(device: str) -> str:
    """`amd`, `cpu` or `nvidia`, as ci-infra's device registry has it."""
    if _amd_family(device):
        return "amd"
    return "cpu" if device.startswith("cpu") else "nvidia"


def device_key_tag(device: str) -> str:
    """What a second device of a vendor in one step adds to the key: the
    hardware family (`mi355_4` -> `mi355`, `mi300_dpx` -> `mi300-dpx`,
    `h200_35gb` -> `h200`, `b200-k8s` -> `b200`, every CPU -> `cpu`). The
    registry in ci-infra is the reference; this follows its naming."""
    amd = _amd_family(device)
    if amd:
        family, size = amd
        return family + ("-dpx" if size == "dpx" else "")
    if device.startswith("cpu"):
        return "cpu"
    device = device.removesuffix("-k8s")
    head, sep, memory = device.rpartition("_")
    if sep and memory.endswith("gb") and memory[:-2].isdigit():
        return head
    return device


def _pinned_key(entry: dict) -> tuple[str | None, str | None]:
    """(key, problem) of the `key:` a device or variant pins, if any."""
    if "key" not in entry:
        return None, None
    pinned = entry["key"]
    if isinstance(pinned, str) and pinned.strip():
        return pinned, None
    return None, "`key:` is empty; drop it or name the job's key"


def device_job_keys(step: dict) -> list[tuple[str, str | None, str | None]]:
    """(job, key, problem) of every job of a `devices:` step, in map order.

    With K the step's key: the first NVIDIA or CPU device is K, any other
    K-<key tag>; the first AMD device is amd-K, any other amd-K-<key tag>; a
    variant is its device job's key plus `-<name>`. A `key:` on a device or
    variant is the whole key. `same_as` copies neither the key nor the
    variants of the device it names. Two unpinned devices with one key tag
    are a problem (reordering them would swap their keys), and so are two
    jobs on one key.
    """
    base = step["key"]
    devices = step.get("devices")
    if not isinstance(devices, dict) or not devices:
        return [("devices", None, "`devices` must be a map of device names")]
    jobs: list[tuple[str, str | None, str | None]] = []
    tags: dict[str, str] = {}
    first: set[str] = set()
    for device, entry in devices.items():
        entry = entry if isinstance(entry, dict) else {}
        vendor = device_vendor(str(device))
        is_amd = vendor == "amd"
        tag = device_key_tag(str(device))
        pinned, problem = _pinned_key(entry)
        if problem is None and pinned is None and tag in tags:
            problem = f"shares the key tag `{tag}` with {tags[tag]}; pin `key:` on one"
        if pinned is None:
            tags.setdefault(tag, str(device))
        prefix = "amd-" if is_amd else ""
        default = f"{prefix}{base}" if is_amd not in first else f"{prefix}{base}-{tag}"
        first.add(is_amd)
        key = pinned or default
        jobs.append((str(device), key, problem))
        variants = entry.get("variants")
        for name, body in (variants if isinstance(variants, dict) else {}).items():
            body = body if isinstance(body, dict) else {}
            variant_key, variant_problem = _pinned_key(body)
            jobs.append(
                (
                    f"{device} variant {name}",
                    variant_key or f"{key}-{name}",
                    variant_problem,
                )
            )
    seen: dict[str, str] = {}
    checked = []
    for job, key, problem in jobs:
        if problem is None and key in seen:
            problem = f"`{key}` is also {seen[key]}'s key"
        if key is not None:
            seen.setdefault(key, job)
        checked.append((job, key, problem))
    return checked


def step_job_keys(step: object) -> list[str]:
    """Every job key a step publishes: its own, `amd-<key>` for a legacy
    `mirror.amd`, or each job's of a `devices:` step."""
    if not isinstance(step, dict) or not isinstance(step.get("key"), str):
        return []
    if "devices" in step:
        return [key for _, key, _ in device_job_keys(step) if key]
    keys = [step["key"]]
    if isinstance(step.get("mirror"), dict) and step["mirror"].get("amd"):
        keys.append(f"amd-{step['key']}")
    return keys


def load_steps(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "steps" not in doc:
        return []
    return doc["steps"]


def keyless_steps(path: Path) -> list[str]:
    steps = load_steps(path)
    return scan(steps, path) if steps else []


def duplicate_keys(files: list[Path]) -> list[str]:
    """Each job key two `devices:` jobs (or one and an older step) share."""
    owners: dict[str, list[tuple[Path, bool]]] = {}
    for path in files:
        for step in load_steps(path) or []:
            device_step = isinstance(step, dict) and "devices" in step
            for key in step_job_keys(step):
                owners.setdefault(key, []).append((path, device_step))
    return [
        f"`{key}` is the key of {len(where)} jobs, in "
        + ", ".join(sorted({str(path) for path, _ in where}))
        for key, where in sorted(owners.items())
        if len(where) > 1 and any(device_step for _, device_step in where)
    ]


def main() -> int:
    files = sorted({p for d in job_dirs() for p in d.rglob(JOB_GLOB)})
    if not files:
        raise SystemExit("no pipeline files under the declared job_dirs")
    keyless = [(path, label) for path in files for label in keyless_steps(path)]
    clashes = [
        f"{path}: {problem}"
        for path in files
        for problem in device_key_problems(load_steps(path))
    ]
    # Keys are unique within a pipeline; the Intel one, say, reuses CUDA's.
    for dirs in pipelines().values():
        pipeline_files = sorted({p for d in dirs for p in d.rglob(JOB_GLOB)})
        clashes += [c for c in duplicate_keys(pipeline_files) if c not in clashes]
    if keyless:
        print(
            f"{len(keyless)} Buildkite step(s) have no explicit key:\n",
            file=sys.stderr,
        )
        for path, label in keyless:
            print(f"  {path}: {label}", file=sys.stderr)
        print(
            "\nAdd a `key:` naming the test each step runs. Without one the pipeline"
            "\ngenerator derives the key from the label, so retitling the step renames"
            "\nits job and breaks retries and build history.",
            file=sys.stderr,
        )
    if clashes:
        print(
            f"\n{len(clashes)} Buildkite job key problem(s) in `devices:` steps:\n",
            file=sys.stderr,
        )
        for line in clashes:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nEach device and variant of a step is a job keyed from the step's"
            "\n`key`; pin a `key:` on the device or variant to give it another.",
            file=sys.stderr,
        )
    return 1 if keyless or clashes else 0


if __name__ == "__main__":
    raise SystemExit(main())
