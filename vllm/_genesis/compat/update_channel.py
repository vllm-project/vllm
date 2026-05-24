# SPDX-License-Identifier: Apache-2.0
"""Genesis update channel — apt-style stable/beta/dev channels.

Phase 3.x scope: channel selection + check-for-updates against the
Genesis GitHub repo. The actual file-update apply pass is DEFERRED
because operators using a git checkout can `git pull` manually based
on this tool's output, which avoids the security + atomicity risks
of building our own pull machinery.

Channels
────────
  stable  — latest tagged release (default; recommended for prod)
  beta    — release candidate branch (testers)
  dev     — main branch HEAD (core developers)

Storage
───────
  $GENESIS_UPDATE_DIR/channel.json   current channel + last_check
  $GENESIS_UPDATE_DIR/cache.json     last GitHub API response

Default: ~/.genesis/update/

Override channel without persisting via GENESIS_UPDATE_CHANNEL env.

CLI
───
  python3 -m vllm._genesis.compat.update_channel status
  python3 -m vllm._genesis.compat.update_channel check
  python3 -m vllm._genesis.compat.update_channel channel set beta
  python3 -m vllm._genesis.compat.update_channel channel get
  python3 -m vllm._genesis.compat.update_channel apply  # prints manual instr

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.compat.update_channel")


# ─── Constants ────────────────────────────────────────────────────────────


KNOWN_CHANNELS = ("stable", "beta", "dev")
DEFAULT_CHANNEL = "stable"
GITHUB_REPO = "Sandermage/genesis-vllm-patches"
CACHE_TTL_SECONDS = 24 * 3600  # 24h cache to avoid rate-limiting

CHANNEL_REFS = {
    # `stable` resolves to whatever GitHub Releases marks as latest. For
    # repos without releases, fall back to `main` tag-stable convention.
    "stable": "main",       # for now (no releases yet); promote to a tag
                             # once Sandermage starts cutting tagged releases
    "beta":   "main",       # beta = main HEAD until a separate beta branch exists
    "dev":    "main",       # dev = main HEAD; will fork to a `dev` branch
                             # if needed
}


# ─── Storage ─────────────────────────────────────────────────────────────


def _resolve_update_dir() -> Path:
    override = os.environ.get("GENESIS_UPDATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path("~/.genesis/update").expanduser()


def _ensure_dir() -> Path:
    base = _resolve_update_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _channel_file() -> Path:
    return _ensure_dir() / "channel.json"


def _cache_file() -> Path:
    return _ensure_dir() / "cache.json"


# ─── Channel get/set ─────────────────────────────────────────────────────


def get_channel() -> str:
    """Return current channel. Env override > persisted > default."""
    env = os.environ.get("GENESIS_UPDATE_CHANNEL")
    if env and env in KNOWN_CHANNELS:
        return env

    f = _channel_file()
    if f.is_file():
        try:
            data = json.loads(f.read_text())
            ch = data.get("channel")
            if ch in KNOWN_CHANNELS:
                return ch
        except Exception as e:
            log.debug("channel.json parse failed: %s", e)
    return DEFAULT_CHANNEL


def set_channel(channel: str) -> None:
    """Persist channel choice. Raises ValueError on unknown channels."""
    if channel not in KNOWN_CHANNELS:
        raise ValueError(
            f"unknown channel {channel!r} — must be one of {KNOWN_CHANNELS}"
        )
    f = _channel_file()
    data = {"channel": channel,
            "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    f.write_text(json.dumps(data, indent=2))


# ─── Local commit detection ─────────────────────────────────────────────


def detect_local_commit() -> str | None:
    """Best-effort: short SHA of the Genesis checkout. Returns None
    if we're not in a git repo or git rev-parse fails."""
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".git").is_dir():
        return None
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception as e:
        log.debug("git rev-parse failed: %s", e)
    return None


# ─── GitHub API ─────────────────────────────────────────────────────────


def _fetch_github_ref(channel: str) -> dict[str, Any]:
    """Query the GitHub API for the tip of the given channel.
    Returns the parsed JSON (sha + commit metadata).

    Designed to be monkey-patchable in tests.
    """
    ref = CHANNEL_REFS.get(channel, "main")
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{ref}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "Genesis-update-checker/1.0",
    })
    # Optional auth token to avoid rate-limiting
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


# ─── Check-for-updates with caching ─────────────────────────────────────


def _read_cache(channel: str) -> dict[str, Any] | None:
    f = _cache_file()
    if not f.is_file():
        return None
    try:
        all_cache = json.loads(f.read_text())
        per = all_cache.get(channel, {})
        if not per:
            return None
        ts = per.get("cached_at_epoch", 0)
        if time.time() - ts > CACHE_TTL_SECONDS:
            return None  # stale
        return per
    except Exception as e:
        log.debug("cache parse failed: %s", e)
        return None


def _write_cache(channel: str, data: dict[str, Any]) -> None:
    f = _cache_file()
    all_cache: dict[str, Any] = {}
    if f.is_file():
        try:
            all_cache = json.loads(f.read_text())
        except Exception:
            all_cache = {}
    data = dict(data)
    data["cached_at_epoch"] = time.time()
    all_cache[channel] = data
    f.write_text(json.dumps(all_cache, indent=2, default=str))


def check_for_updates(force_refresh: bool = False) -> dict[str, Any]:
    """Query upstream + compare to local commit. Returns:
      {
        "channel": "stable" | "beta" | "dev",
        "local_sha": "abc1234" | None,
        "upstream_sha": "def5678",
        "upstream_date": "...",
        "upstream_message": "...",
        "update_available": bool | None,
        "from_cache": bool,
        "cache_age_seconds": float | None,
        "error": str | None,
      }
    """
    channel = get_channel()
    local = detect_local_commit()

    # Try cache first unless force-refresh
    cached = None if force_refresh else _read_cache(channel)
    if cached is not None:
        upstream_sha = cached.get("sha", "")
        upstream_date = cached.get("commit_date", "")
        upstream_msg = cached.get("commit_message", "")
        from_cache = True
        cache_age = time.time() - cached.get("cached_at_epoch", time.time())
        error = None
    else:
        try:
            data = _fetch_github_ref(channel)
            upstream_sha = data.get("sha", "")
            commit = data.get("commit") or {}
            upstream_date = (commit.get("author") or {}).get("date", "")
            upstream_msg = commit.get("message", "")
            _write_cache(channel, {
                "sha": upstream_sha,
                "commit_date": upstream_date,
                "commit_message": upstream_msg,
            })
            from_cache = False
            cache_age = 0
            error = None
        except Exception as e:
            return {
                "channel": channel,
                "local_sha": local,
                "upstream_sha": None,
                "update_available": None,
                "from_cache": False,
                "error": f"{type(e).__name__}: {e}",
            }

    # Decide update_available
    if local is None:
        update_available = None  # can't tell
    else:
        # Compare short SHAs prefix-wise to handle length mismatch
        u = (upstream_sha or "")[:7]
        l = local[:7]
        update_available = bool(u) and (u != l)

    return {
        "channel": channel,
        "local_sha": local,
        "upstream_sha": upstream_sha,
        "upstream_date": upstream_date,
        "upstream_message": upstream_msg.split("\n")[0] if upstream_msg else "",
        "update_available": update_available,
        "from_cache": from_cache,
        "cache_age_seconds": cache_age if from_cache else 0,
        "error": error,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────


def _format_status() -> list[str]:
    L = []
    L.append("=" * 72)
    L.append("Genesis update channel status")
    L.append("=" * 72)
    ch = get_channel()
    L.append(f"  Current channel: {ch}")
    L.append(f"  Storage dir:     {_resolve_update_dir()}")
    L.append(f"  GitHub repo:     {GITHUB_REPO}")
    L.append(f"  Available channels: {', '.join(KNOWN_CHANNELS)}")

    local = detect_local_commit()
    L.append(f"  Local commit: {local or '(not in git checkout)'}")
    L.append("")
    L.append("  Run `update_channel check` to query upstream.")
    L.append("=" * 72)
    return L


def _format_check(result: dict[str, Any]) -> list[str]:
    L = []
    L.append("=" * 72)
    L.append(f"Genesis update check — channel: {result['channel']}")
    L.append("=" * 72)
    if result.get("error"):
        L.append(f"  ✗ ERROR: {result['error']}")
        L.append("")
        L.append("  Network may be down, GitHub may be rate-limiting,")
        L.append("  or the repo may be temporarily unavailable.")
        L.append("=" * 72)
        return L

    L.append(f"  Local commit:    {result.get('local_sha') or '(not detected)'}")
    L.append(f"  Upstream commit: {result.get('upstream_sha', '?')}")
    if result.get("upstream_date"):
        L.append(f"  Upstream date:   {result['upstream_date']}")
    if result.get("upstream_message"):
        L.append(f"  Upstream HEAD:   {result['upstream_message'][:60]}")
    if result.get("from_cache"):
        age = result.get("cache_age_seconds", 0)
        L.append(f"  (cached result, age: {age/60:.1f} min — "
                 f"--force-refresh to re-query)")

    L.append("")
    if result.get("update_available") is True:
        L.append("  🔼 UPDATE AVAILABLE")
        L.append("")
        L.append("  To pull the new version:")
        L.append("    git pull origin main")
        L.append("  Then verify:")
        L.append("    python3 -m vllm._genesis.compat.doctor")
        L.append("    python3 -m vllm._genesis.compat.schema_validator")
    elif result.get("update_available") is False:
        L.append("  ✓ Up-to-date — local commit matches upstream")
    else:
        L.append("  ? Update status unknown (couldn't detect local commit)")
    L.append("=" * 72)
    return L


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.update_channel",
        description="Genesis update channel — check + manage update channel.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show update channel + local state")
    sp_check = sub.add_parser("check", help="Query upstream for updates")
    sp_check.add_argument("--force-refresh", action="store_true",
                          help="Bypass cache + re-query GitHub")
    sp_check.add_argument("--json", action="store_true")

    sp_chan = sub.add_parser("channel", help="Get/set the channel")
    sp_chan_sub = sp_chan.add_subparsers(dest="chan_cmd", required=True)
    sp_chan_sub.add_parser("get", help="Print current channel")
    sp_set = sp_chan_sub.add_parser("set", help="Set channel")
    sp_set.add_argument("name", choices=list(KNOWN_CHANNELS))

    sub.add_parser("apply",
                   help="Print manual update instructions (apply is deferred)")

    args = parser.parse_args(argv)

    if args.cmd == "status":
        for line in _format_status():
            print(line)
        return 0

    if args.cmd == "check":
        result = check_for_updates(force_refresh=args.force_refresh)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            for line in _format_check(result):
                print(line)
        # Exit code: 0 = up to date / unknown, 1 = update available, 2 = error
        if result.get("error"):
            return 2
        if result.get("update_available") is True:
            return 1
        return 0

    if args.cmd == "channel":
        if args.chan_cmd == "get":
            print(get_channel())
            return 0
        if args.chan_cmd == "set":
            try:
                set_channel(args.name)
                print(f"✓ channel set to {args.name!r}")
                return 0
            except ValueError as e:
                print(f"✗ {e}", file=sys.stderr)
                return 2

    if args.cmd == "apply":
        print("Genesis update apply is deferred for safety reasons.")
        print()
        print("To pull the latest version manually:")
        print(f"  git pull origin main")
        print()
        print("Then verify your stack with:")
        print("  python3 -m vllm._genesis.compat.doctor")
        print("  python3 -m vllm._genesis.compat.schema_validator")
        print("  python3 -m vllm._genesis.compat.lifecycle_audit_cli")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
