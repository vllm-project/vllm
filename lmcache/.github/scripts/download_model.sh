#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download (or just verify the local cache of) a HuggingFace model
# repo, with bounded retry + exponential backoff so flaky HF mirrors
# don't fail the whole CI run.
#
# Usage:
#   download_model.sh <repo_id> [<repo_id> ...]
#   MODEL_ID=facebook/opt-125m download_model.sh
#
# Environment:
#   MODEL_ID                  fallback when no positional args given
#   HF_TOKEN                  optional HuggingFace token for authenticated
#                             requests (avoids anonymous rate limits)
#   HF_DOWNLOAD_MAX_RETRIES   default 5
#   HF_DOWNLOAD_RETRY_DELAY   default 30 (seconds, doubled per retry)
#   HF_DOWNLOAD_FAIL_ON_ERROR default 0  (1 -> exit non-zero on failure)

set -euo pipefail

if [ "$#" -eq 0 ]; then
  if [ -z "${MODEL_ID:-}" ]; then
    echo "!! download_model.sh: no model id provided"
    echo "   pass repo ids as args or set MODEL_ID=..."
    exit 2
  fi
  set -- "${MODEL_ID}"
fi

MAX_RETRIES="${HF_DOWNLOAD_MAX_RETRIES:-5}"
RETRY_DELAY="${HF_DOWNLOAD_RETRY_DELAY:-30}"
FAIL_ON_ERROR="${HF_DOWNLOAD_FAIL_ON_ERROR:-0}"

MAX_RETRIES="${MAX_RETRIES}" RETRY_DELAY="${RETRY_DELAY}" \
FAIL_ON_ERROR="${FAIL_ON_ERROR}" HF_TOKEN="${HF_TOKEN:-}" \
  python3 - "$@" <<'PY'
import os
import random
import sys
import time

from huggingface_hub import snapshot_download

max_retries = int(os.environ["MAX_RETRIES"])
base_delay = int(os.environ["RETRY_DELAY"])
fail_on_error = os.environ["FAIL_ON_ERROR"] == "1"
hf_token = os.environ.get("HF_TOKEN") or None

repos = sys.argv[1:]
failures = []

for repo in repos:
    # Try local cache first to avoid unnecessary HF API calls
    # (which can 429 on busy CI runners even when the model is cached).
    try:
        snapshot_download(repo, local_files_only=True, token=hf_token)
        print(f"CACHED: {repo} (local, no network)")
        continue
    except Exception:
        pass

    delay = base_delay
    ok = False
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: {repo}")
            snapshot_download(repo, token=hf_token)
            print(f"OK: {repo}")
            ok = True
            break
        except Exception as exc:
            print(f"Attempt {attempt + 1} failed for {repo}: {exc}")
            if attempt < max_retries - 1:
                # Add ±25% jitter to avoid thundering herd
                jitter = delay * (0.75 + random.random() * 0.5)
                print(
                    "Waiting %.1fs before retry..." % jitter)
                time.sleep(jitter)
                delay *= 2
    if not ok:
        failures.append(repo)

if failures:
    print("All retry attempts failed for: %s" % ", ".join(failures))
    sys.exit(1 if fail_on_error else 0)
PY
