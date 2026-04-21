#!/usr/bin/env python3
import argparse
import pathlib
import sys
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--requirements-out", required=True)
    parser.add_argument(
        "--forbid-prefix",
        action="append",
        default=[],
        help="Normalized package-name prefixes that must not appear in Requires-Dist.",
    )
    parser.add_argument(
        "--error-on-forbidden",
        action="store_true",
        help="Exit non-zero if a forbidden requirement is found.",
    )
    return parser.parse_args()


def normalize_requirement_name(requirement: str) -> str:
    raw = requirement.split(";", 1)[0].strip()
    for separator in ("[", " ", "<", ">", "=", "!", "~"):
        raw = raw.split(separator, 1)[0]
    return raw.lower().replace("_", "-")


def main() -> int:
    args = parse_args()
    wheel = pathlib.Path(args.wheel)
    requirements_out = pathlib.Path(args.requirements_out)

    forbidden = [item.lower().replace("_", "-") for item in args.forbid_prefix]
    requirements: list[str] = []
    violations: list[str] = []

    with zipfile.ZipFile(wheel) as zf:
      metadata_name = next(
          name for name in zf.namelist() if name.endswith("METADATA"))
      metadata = zf.read(metadata_name).decode()

    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist: "):
            continue
        requirement = line.removeprefix("Requires-Dist: ")
        normalized_name = normalize_requirement_name(requirement)

        if any(normalized_name.startswith(prefix) for prefix in forbidden):
            violations.append(requirement)
            continue

        if "; extra ==" in requirement.lower():
            continue

        requirements.append(requirement)

    requirements_out.write_text("\n".join(requirements) + ("\n" if requirements else ""))

    if violations and args.error_on_forbidden:
        print("Forbidden wheel requirements detected:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
