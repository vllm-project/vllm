# SPDX-License-Identifier: Apache-2.0
"""
CLI tool to fetch and display the MP cache server status.

Usage:
    python -m lmcache.tools.mp_status_viewer [--url URL]
"""

# Standard
import argparse
import json
import sys
import urllib.error
import urllib.request

HEALTHY = "\033[32mHEALTHY\033[0m"
UNHEALTHY = "\033[31mUNHEALTHY\033[0m"

INDENT = "  "


def _health_str(is_healthy: bool) -> str:
    return HEALTHY if is_healthy else UNHEALTHY


def _fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def _print_section(title: str, data: dict, depth: int = 0) -> None:
    prefix = INDENT * depth
    health = _health_str(data.get("is_healthy", True))
    print(f"{prefix}{title}  [{health}]")

    for key, value in data.items():
        if key == "is_healthy":
            continue
        if isinstance(value, dict):
            _print_section(key, value, depth + 1)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, item in enumerate(value):
                _print_section(f"{key}[{i}]", item, depth + 1)
        else:
            display = value
            if "bytes" in key.lower() and isinstance(value, (int, float)):
                display = _fmt_bytes(int(value))
            elif isinstance(value, float):
                display = f"{value:.4f}"
            print(f"{prefix}{INDENT}{key}: {display}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and display LMCache MP server status"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/status",
        help="URL of the status endpoint",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted view",
    )
    args = parser.parse_args()

    try:
        req = urllib.request.Request(args.url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"Error connecting to {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    print("=" * 60)
    print("  LMCache MP Server Status")
    print("=" * 60)
    _print_section("engine", data)
    print("=" * 60)


if __name__ == "__main__":
    main()
