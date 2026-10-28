# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report lock changes and known vulnerabilities without blocking by default."""

import argparse
import html
import json
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
LOCKFILES = [
    "requirements/docs.txt",
    "requirements/test/cpu.txt",
    "requirements/test/cuda.txt",
    "requirements/test/rocm.txt",
    "requirements/test/xpu.txt",
]
OSV_URL = "https://api.osv.dev/v1/querybatch"
HASH = re.compile(r"\s+--hash=sha256:([a-f0-9]{64})(?=\s|$)")


@dataclass
class Pin:
    version: str
    hashes: list[str]


def parse_lock(text: str, require_hashes: bool) -> dict[tuple[str, str], Pin]:
    pins = {}
    for line in re.sub(r"\\\r?\n", " ", text).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        hashes = sorted(set(HASH.findall(line)))
        requirement = Requirement(HASH.sub("", line))
        specs = list(requirement.specifier)
        if (
            requirement.url
            or len(specs) != 1
            or specs[0].operator != "=="
            or "*" in specs[0].version
            or (require_hashes and not hashes)
        ):
            raise ValueError(f"Expected an exact hashed registry pin: {line}")
        version = str(Version(specs[0].version))
        key = (canonicalize_name(requirement.name), str(requirement.marker or ""))
        if key in pins:
            raise ValueError(f"Duplicate package/marker: {key}")
        pins[key] = Pin(version, hashes)
    if not pins:
        raise ValueError("Empty dependency lock")
    return pins


def audit(packages: set[tuple[str, str]], url: str) -> Iterator[dict]:
    if not packages:
        raise ValueError("No scannable package pins")
    ordered = sorted(packages)
    for start in range(0, len(ordered), 100):
        batch = ordered[start : start + 100]
        payload = {
            "queries": [
                {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
                for name, version in batch
            ]
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or len(results) != len(batch):
            raise ValueError("OSV returned an incomplete result set")
        for (name, version), result in zip(batch, results):
            if not isinstance(result, dict) or set(result) - {
                "vulns",
                "next_page_token",
            }:
                raise ValueError("OSV returned an invalid result")
            if result.get("next_page_token"):
                raise ValueError("OSV returned paginated results; audit is incomplete")
            vulns = result.get("vulns", [])
            if not isinstance(vulns, list) or any(
                not isinstance(v, dict)
                or not isinstance(v.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v["id"])
                for v in vulns
            ):
                raise ValueError("OSV returned invalid vulnerability identifiers")
            yield {
                "name": name,
                "version": version,
                "vulnerabilities": sorted({v["id"] for v in vulns}),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Git commit before the refresh")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--osv-url", default=OSV_URL, help="OSV querybatch endpoint")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Fail on findings or incomplete checks (default: report only)",
    )
    args = parser.parse_args()
    report = [
        "# Compiled dependency review\n",
        f"Base commit: `{html.escape(args.base)}`\n",
        "Hashes identify archives; this report does not verify their sources or "
        "publisher provenance. Sources are not recorded in these locks.\n",
        "Vulnerability enforcement: "
        + ("**enabled**.\n" if args.enforce else "**disabled (report only)**.\n"),
    ]
    inventory = {}
    packages = set()
    errors = []
    for path in LOCKFILES:
        try:
            previous = subprocess.check_output(
                ["git", "show", f"{args.base}:{path}"], cwd=ROOT, text=True
            )
            before = parse_lock(previous, require_hashes=False)
            after = parse_lock((ROOT / path).read_text(), require_hashes=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            errors.append(f"{path}: {error}")
            inventory[path] = {"error": str(error)}
            report.extend(
                [f"## {path}\n", f"**Report incomplete:** {html.escape(str(error))}\n"]
            )
            continue
        changes = []
        report.extend(
            [
                f"## {path}\n",
                f"{len(after)} pins parsed, including all environment markers.\n",
                "| Package | Before | After | Added / removed hashes | Review |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name, marker in sorted(before.keys() | after.keys()):
            old, new = before.get((name, marker)), after.get((name, marker))
            if new:
                packages.add((name, new.version))
                # Vendor builds may inherit upstream advisories.
                packages.add((name, Version(new.version).public))
            if old == new:
                continue
            old_hashes = set(old.hashes) if old else set()
            new_hashes = set(new.hashes) if new else set()
            label = name + (f"; {marker}" if marker else "")
            reason = "Version changed"
            if old is None:
                reason = "**New dependency**"
            elif new is None:
                reason = "Removed dependency"
            elif old.version == new.version:
                reason = "**Same-version hash change**"
            report.append(
                f"| {html.escape(label)} | {old.version if old else '—'} | "
                f"{new.version if new else '—'} | "
                f"+{len(new_hashes - old_hashes)} / "
                f"-{len(old_hashes - new_hashes)} | {reason} |"
            )
            changes.append(
                {
                    "name": name,
                    "marker": marker,
                    "before": asdict(old) if old else None,
                    "after": asdict(new) if new else None,
                }
            )
        if not changes:
            report.append("| No dependency changes | | | | |")
        report.append("")
        inventory[path] = {"pins_parsed": len(after), "changes": changes}

    report.extend(["## Known vulnerabilities\n", f"Advisory service: {args.osv_url}\n"])
    result = {
        "base": args.base,
        "enforced": args.enforce,
        "locks": inventory,
        "audit": [],
        "errors": errors,
    }
    findings = False
    try:
        for item in audit(packages, args.osv_url):
            result["audit"].append(item)
            for vuln in item["vulnerabilities"]:
                findings = True
                report.append(
                    f"- `{item['name']}=={item['version']}`: "
                    f"[{vuln}](https://osv.dev/vulnerability/{vuln})"
                )
        report.append(
            f"\nChecked {len(packages)} distinct package/version pairs against OSV, "
            "including public versions of vendor builds.\n"
        )
        report.append(
            "\nKnown vulnerabilities found; review the advisories above."
            if findings
            else "No known vulnerabilities returned. This does not establish safety."
        )
    except (OSError, ValueError) as error:
        errors.append(str(error))
        report.append(f"**Advisory check incomplete:** {html.escape(str(error))}")
    result["complete"] = not errors
    if not args.enforce:
        report.append("\nFindings and incomplete checks do not block publication.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dependency-review.md").write_text("\n".join(report) + "\n")
    (args.output_dir / "dependency-review.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return int(args.enforce and (findings or bool(errors)))


if __name__ == "__main__":
    sys.exit(main())
