# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging helper that writes test-retry:<job_id> Buildkite meta-data.

Uploads two Buildkite block steps (tag-wave-1 and tag-wave-2) and polls
Buildkite's own internal meta-data store every 1s so tags are written within
1s of unblocking with zero external API calls or rate limits.
"""

import subprocess
import sys
import time

PIPELINE_YAML = """steps:
  - block: "Staging: tag Wave 1 retries"
    key: "staging-tag-wave-1"
    fields:
      - text: "Wave 1 job IDs"
        key: "tag-wave-1"
  - block: "Staging: tag Wave 2 retries"
    key: "staging-tag-wave-2"
    fields:
      - text: "Wave 2 job IDs"
        key: "tag-wave-2"
"""


def _meta_get(key: str) -> str:
    run = subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        capture_output=True,
        text=True,
        check=False,
    )
    return (run.stdout or "").strip() if run.returncode == 0 else ""


def main() -> int:
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=PIPELINE_YAML,
        text=True,
        check=True,
    )
    print("Uploaded staging-tag-wave-1 and staging-tag-wave-2 block steps.", flush=True)
    tagged: set[str] = set()
    waves_done: set[str] = set()
    deadline = time.time() + 2400
    while time.time() < deadline:
        for key in ("tag-wave-1", "tag-wave-2"):
            if key in waves_done:
                continue
            val = _meta_get(key)
            if not val:
                continue
            for jid in val.split():
                if jid != "DONE" and jid not in tagged:
                    subprocess.run(
                        ["buildkite-agent", "meta-data", "set", f"test-retry:{jid}", "1"],
                        check=True,
                    )
                    print(f"TAGGED test-retry:{jid} (from {key})", flush=True)
                    tagged.add(jid)
            waves_done.add(key)
        if len(waves_done) == 2:
            print("All waves tagged; exiting.", flush=True)
            return 0
        time.sleep(1)
    print("TIMEOUT", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
