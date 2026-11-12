# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""List vendored vLLM code that the installed Transformers may make redundant.

Run with the Transformers version under test, from the vLLM repo root:

    uv run --no-project --with transformers==<version> \
        python .agents/skills/transformers-upgrade/scripts/find_upstreamed.py

Imports nothing from vLLM, so no vLLM install is needed. Uses the `gh` CLI, if
available, to check whether referenced Transformers PRs are in this release.
"""

import json
import subprocess
import sys
from pathlib import Path

import regex as re
import transformers
from packaging.version import Version
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FLOOR = Version(transformers.__version__)
UPSTREAM_NAMES = set(dir(transformers))

MODEL_TYPE = re.compile(r'model_type\s*(?::\s*str\s*)?=\s*"([^"]+)"')
REGISTRY_KEY = re.compile(r'^\s+(?:\*\*\{)?"?([\w-]+)"?(?:=|": )"(\w+)"', re.M)
CLASS_KEY = re.compile(r'^\s+"(\w+)": ', re.M)
UPSTREAM_REF = re.compile(
    r"huggingface/transformers(?:/(?:issues|pull)/|#)(\d+)"
    r"|(?:once|until|when) [Tt]ransformers [\d.]+ is the minimum"
)
VERSIONED = re.compile(
    r"(?:min_transformers_version|max_transformers_version|check_version|Version)"
    r'\(?\s*=?\s*"([45]\.\d+(?:\.\d+)?)[^"]*"'
    r"|[Tt]ransformers\s*(?:v|[<>]=?\s*)([45]\.\d+(?:\.\d+)?|4)\b"
)
READS_VERSION = re.compile(
    r"transformers import __version__|transformers\.__version__"
    r"|version\([\"']transformers[\"']\)|TRANSFORMERS_VERSION|check_version\("
)


def section(title: str) -> None:
    print(f"\n## {title}")


def vendored_configs() -> None:
    section(f"Vendored configs whose model_type is upstream in {FLOOR}")
    files = sorted(ROOT.glob("vllm/transformers_utils/configs/*.py"))
    files += sorted(ROOT.glob("vllm/models/*/config*.py"))
    files += sorted(ROOT.glob("vllm/models/*/configs/*.py"))
    for path in files:
        types = sorted(set(MODEL_TYPE.findall(path.read_text())))
        hits = [t for t in types if t in CONFIG_MAPPING_NAMES]
        if hits:
            upstream = ", ".join(f"{t} -> {CONFIG_MAPPING_NAMES[t]}" for t in hits)
            print(f"- {path.relative_to(ROOT)}: {upstream}")

    registry = (ROOT / "vllm/transformers_utils/config.py").read_text()
    start = registry.index("_CONFIG_REGISTRY")
    body = registry[start : registry.index("\n)\n", start)]
    section("_CONFIG_REGISTRY entries that override an upstream model_type")
    for model_type, cls in REGISTRY_KEY.findall(body):
        if model_type in CONFIG_MAPPING_NAMES:
            print(f"- {model_type}: vLLM {cls} over {CONFIG_MAPPING_NAMES[model_type]}")


def vendored_processors() -> None:
    init = (ROOT / "vllm/transformers_utils/processors/__init__.py").read_text()
    names = sorted(set(CLASS_KEY.findall(init)))
    section("Registered processor names that collide with upstream classes")
    for name in names:
        if name in UPSTREAM_NAMES:
            print(f"- {name}")
    section("Processor files named after a model_type with an upstream processor")
    for path in sorted(ROOT.glob("vllm/transformers_utils/processors/*.py")):
        if path.stem in PROCESSOR_MAPPING_NAMES:
            upstream = PROCESSOR_MAPPING_NAMES[path.stem]
            print(f"- {path.relative_to(ROOT)} -> {upstream}")


def version_gates() -> None:
    files = sorted([*ROOT.glob("vllm/**/*.py"), *ROOT.glob("tests/**/*.py")])
    section(f"Version-gated code at or below {FLOOR} (review each)")
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for match in VERSIONED.finditer(line):
                version = Version(match.group(1) or match.group(2))
                if version <= FLOOR:
                    print(f"- {path.relative_to(ROOT)}:{lineno}: {line.strip()}")
                    break
    section("Code that reads the installed Transformers version (review each)")
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if READS_VERSION.search(line):
                print(f"- {path.relative_to(ROOT)}:{lineno}: {line.strip()}")


def upstream_status(number: str) -> str:
    """Whether Transformers issue/PR `number` is resolved in this release."""

    def gh(path: str) -> dict | None:
        try:
            out = subprocess.run(["gh", "api", path], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return json.loads(out.stdout) if out.returncode == 0 else None

    repo = "repos/huggingface/transformers"
    if (issue := gh(f"{repo}/issues/{number}")) is None:
        return "status unknown"
    if "pull_request" not in issue:
        return f"issue {issue['state']}, check which PR fixed it"
    pr = gh(f"{repo}/pulls/{number}")
    if pr is None or not pr["merged_at"]:
        return "PR not merged"
    compare = gh(f"{repo}/compare/v{FLOOR}...{pr['merge_commit_sha']}")
    included = compare and compare["status"] in ("behind", "identical")
    return f"PR merged, {'in' if included else 'not in'} v{FLOOR}"


def upstream_refs() -> None:
    section("Patches referencing Transformers issues/PRs or a future minimum")
    files = sorted([*ROOT.glob("vllm/**/*.py"), *ROOT.glob("tests/**/*.py")])
    statuses: dict[str, str] = {}
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if (match := UPSTREAM_REF.search(line)) is None:
                continue
            note = ""
            if number := match.group(1):
                if number not in statuses:
                    statuses[number] = upstream_status(number)
                note = f" [#{number}: {statuses[number]}]"
            print(f"- {path.relative_to(ROOT)}:{lineno}:{note} {line.strip()}")


if __name__ == "__main__":
    print(f"# Transformers {FLOOR}")
    vendored_configs()
    vendored_processors()
    version_gates()
    upstream_refs()
