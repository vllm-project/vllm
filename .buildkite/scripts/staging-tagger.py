# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throwaway staging helper that tags Wave 1 retry jobs via a block step."""

import subprocess
import sys
import time

PIPELINE = """steps:
  - block: ":label: Tag Wave 1"
    key: staging-tag-wave-1
    fields:
      - text: "Job IDs (comma-separated)"
        key: tag-wave-1
"""


def _upload_block() -> None:
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=PIPELINE,
        text=True,
        check=True,
    )
    print("Uploaded staging-tag-wave-1 block step", flush=True)


def _wait_and_tag(field_key: str, timeout_s: int = 1000) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["buildkite-agent", "meta-data", "get", field_key, "--default", ""],
            capture_output=True,
            text=True,
        )
        raw = proc.stdout.strip()
        if raw:
            for job_id in [x.strip() for x in raw.split(",") if x.strip()]:
                subprocess.run(
                    ["buildkite-agent", "meta-data", "set", f"test-retry:{job_id}", "1"],
                    check=True,
                )
                print(f"TAGGED test-retry:{job_id}=1", flush=True)
            return
        time.sleep(1)
    print(f"Timed out waiting for {field_key}", flush=True)
    sys.exit(1)


def main() -> None:
    _upload_block()
    _wait_and_tag("tag-wave-1")
    print("staging-tagger done", flush=True)


if __name__ == "__main__":
    main()
