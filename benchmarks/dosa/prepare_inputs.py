# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct SQLite inputs and the reported model-specific TTFT plans."""

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip"
PREFIXES = {
    "ds": (
        ("128k", 131072, 493014),
        ("256k", 262144, 981111),
        ("384k", 393216, 1401009),
        ("512k", 524288, 1802647),
        ("768k", 786432, 2675019),
        ("1m", 1048569, 3511994),
    ),
    "glm": (
        ("128k", 131072, 537893),
        ("256k", 262144, 1060634),
        ("384k", 393216, 1507846),
        ("512k", 524288, 1955848),
        ("768k", 786432, 2896538),
        ("1m", 1048448, 3765787),
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Reuse SQLite 3.53.4 sqlite3.c")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source is None:
        with (
            urlopen(URL) as response,
            ZipFile(io.BytesIO(response.read())) as archive,
            archive.open("sqlite-amalgamation-3530400/sqlite3.c") as source,
        ):
            text = io.TextIOWrapper(source, encoding="utf-8", newline=None).read()
    else:
        with args.source.open(encoding="utf-8", newline=None) as source:
            text = source.read()
    for model, prefixes in PREFIXES.items():
        directory = args.output_dir.resolve() / model
        directory.mkdir(parents=True, exist_ok=True)
        requests = []
        for name, tokens, end in prefixes:
            path = directory / f"{name}.txt"
            path.write_bytes(text[:end].encode("utf-8"))
            requests.append({"name": name, "input": str(path), "tokens": tokens})
            print(f"{model} {name}: {end} characters, {path}")
        reported = [
            item for item in requests if model != "ds" or item["name"] != "256k"
        ]
        (directory / "requests.json").write_text(json.dumps(reported, indent=2) + "\n")
        middle = [item for item in requests if item["name"] in ("384k", "768k")]
        (directory / "requests-mid.json").write_text(
            json.dumps(middle, indent=2) + "\n"
        )
        only256 = [item for item in requests if item["name"] == "256k"]
        (directory / "requests-256k.json").write_text(
            json.dumps(only256, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
