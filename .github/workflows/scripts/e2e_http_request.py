#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make an HTTP request without depending on the runner's libcurl ABI."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request


def parse_header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition(":")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("headers must use the 'Name: value' form")
    return name.strip(), header_value.lstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--method", choices=("GET", "POST"), default="GET")
    parser.add_argument("--header", action="append", default=[], type=parse_header)
    parser.add_argument("--data")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parsed_url = urllib.parse.urlsplit(args.url)
    if parsed_url.scheme not in {"http", "https"}:
        raise SystemExit("URL scheme must be http or https")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")

    data = args.data.encode() if args.data is not None else None
    request = urllib.request.Request(
        args.url,
        data=data,
        headers=dict(args.header),
        method=args.method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=args.timeout) as response:
            sys.stdout.buffer.write(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"HTTP request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
