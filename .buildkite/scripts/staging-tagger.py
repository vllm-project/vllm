# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging helper that writes test-retry:<job_id> Buildkite meta-data."""

import json
import os
import subprocess
import sys
import time
import urllib.request


def main() -> int:
    pr = os.environ.get("STAGING_PR_NUMBER", "")
    if not pr:
        print("STAGING_PR_NUMBER unset; exiting", flush=True)
        return 0
    url = f"https://api.github.com/repos/vllm-project/vllm/issues/{pr}/comments?per_page=100"
    tagged: set[str] = set()
    deadline = time.time() + 2400
    print(f"Polling PR #{pr} for TAG_RETRY_JOBS ...", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vllm-ci-staging"})
            comments = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            for comment in comments:
                for line in (comment.get("body") or "").splitlines():
                    line = line.strip()
                    if line.startswith("TAG_RETRY_JOBS "):
                        for jid in line.split()[1:]:
                            if jid not in tagged:
                                subprocess.run(
                                    [
                                        "buildkite-agent",
                                        "meta-data",
                                        "set",
                                        f"test-retry:{jid}",
                                        "1",
                                    ],
                                    check=True,
                                )
                                print(f"TAGGED test-retry:{jid}", flush=True)
                                tagged.add(jid)
                    elif line == "TAG_RETRY_DONE":
                        print("DONE", flush=True)
                        return 0
        except Exception as err:
            print(f"poll error: {err}", flush=True)
        time.sleep(2)
    print("TIMEOUT", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
