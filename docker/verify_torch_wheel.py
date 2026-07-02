#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log torch build identifiers for CI verification.

Prints the installed torch version, the exact source commit it was built from
(``torch.version.git_version``), and the sha256 of the torch *wheel file* taken
from the PyTorch package index. The wheel-file hash uniquely identifies the
binary across rebuilds of the same version string -- unlike the package
METADATA, which can be byte-identical for different builds (e.g. RC respins).

Verification must never fail the build, so any lookup error is reported inline
rather than raised.
"""
import argparse
import re
import sys
import urllib.request

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index_url",
        help="Torch package index base, e.g. "
        "https://download.pytorch.org/whl/nightly/cu130",
    )
    args = parser.parse_args()

    version = torch.__version__
    git_version = torch.version.git_version
    # CPython tag of the running interpreter, e.g. cp312.
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    print("=== Torch wheel verification ===")
    print(f"torch=={version}")
    print(f"git_version={git_version}")

    listing_url = args.index_url.rstrip("/") + "/torch/"
    # The local '+cuXXX' build label appears url-encoded ('%2B') on the index.
    version_variants = (version, version.replace("+", "%2B"))
    wheel_file = None
    wheel_sha256 = None
    try:
        html = (
            urllib.request.urlopen(listing_url, timeout=60)
            .read()
            .decode("utf-8", "replace")
        )
        for match in re.finditer(
            r'(torch-[^"#]+\.whl)#sha256=([0-9a-f]{64})', html
        ):
            name, digest = match.group(1), match.group(2)
            if (
                py_tag in name
                and "x86_64" in name
                and any(v in name for v in version_variants)
            ):
                wheel_file, wheel_sha256 = name, digest
                break
    except Exception as exc:  # noqa: BLE001 - never fail the build on lookup
        print(f"wheel_sha256=UNKNOWN (index lookup failed: {exc})")
        print("================================")
        return

    if wheel_sha256:
        print(f"wheel_file={wheel_file}")
        print(f"wheel_sha256={wheel_sha256}")
    else:
        print(
            f"wheel_sha256=UNKNOWN (no {py_tag} x86_64 wheel for {version} "
            f"at {listing_url})"
        )
    print("================================")


if __name__ == "__main__":
    main()
