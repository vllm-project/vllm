#!/usr/bin/env bash
# Pin the currently-installed vLLM nightly to the
# `buildkite_latest_tested_vllm` branch.
#
# Runs ONLY when the canary vllm_bench step has succeeded and
# VERIFY_AND_PIN_VLLM=true. Records the just-verified vllm wheel version so
# downstream builds can install the same version deterministically instead
# of resolving "latest nightly" again.
#
# Two files are maintained on the dedicated branch:
#   tested_vllm_versions.csv  -- JSON Lines, append-only history. Each line
#                                is one self-contained record.
#   latest_tested_vllm.txt    -- Plain text, single line: the most recent
#                                verified version. Overwritten every run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

export GIT_TERMINAL_PROMPT=0

# ── Resolve the version that's actually installed in this pod ───────────
VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
if [[ -z "${VLLM_VERSION}" ]]; then
    echo "[ERROR] could not read vllm.__version__ from the live env" >&2
    exit 1
fi
echo "Verified vLLM version: ${VLLM_VERSION}"

# ── Resolve commit SHAs so consumers don't need to call any external API ─
# The PEP 440 local version after `+g` is the short commit SHA, e.g.
# 0.23.1rc1.dev508+gc6dd32a81 -> c6dd32a81. Expand it to the full 40-char
# SHA via the public GitHub commits API; we already have GITHUB_TOKEN in
# the env (5000 req/h). The full SHA gives us the permanent
# wheels.vllm.ai/<full-sha>/<cuda>/ archive URL, which keeps working even
# after the rolling nightly index has dropped the wheel.
VLLM_SHORT_SHA="${VLLM_VERSION##*+g}"
if [[ "${VLLM_SHORT_SHA}" == "${VLLM_VERSION}" \
        || ! "${VLLM_SHORT_SHA}" =~ ^[0-9a-f]+$ ]]; then
    VLLM_SHORT_SHA=""
fi

VLLM_FULL_SHA=""
if [[ -n "${VLLM_SHORT_SHA}" ]]; then
    gh_auth_args=()
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        gh_auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    fi
    for attempt in 1 2 3; do
        VLLM_FULL_SHA="$(curl -fsSL --connect-timeout 5 --max-time 10 \
            -H "Accept: application/vnd.github+json" \
            "${gh_auth_args[@]+"${gh_auth_args[@]}"}" \
            "https://api.github.com/repos/vllm-project/vllm/commits/${VLLM_SHORT_SHA}" \
            2>/dev/null \
            | jq -r '.sha // empty')" || true
        if [[ "${VLLM_FULL_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
            break
        fi
        VLLM_FULL_SHA=""
        echo "[INFO] GitHub commit lookup attempt ${attempt} for" \
             "${VLLM_SHORT_SHA} returned no SHA; retrying..." >&2
        sleep 2
    done
fi

if [[ -n "${VLLM_FULL_SHA}" ]]; then
    VLLM_ARCHIVE_INDEX="https://wheels.vllm.ai/${VLLM_FULL_SHA}/cu130"
    echo "Resolved full SHA: ${VLLM_FULL_SHA}"
    echo "Archive index:     ${VLLM_ARCHIVE_INDEX}"
else
    VLLM_ARCHIVE_INDEX=""
    echo "[WARN] could not resolve full SHA for short SHA" \
         "'${VLLM_SHORT_SHA:-<none>}'; archive_index_url will be empty" \
         "and consumers will fall back to live API lookup" >&2
fi

CI_REPO="LMCache/LMCache"
CI_BRANCH="buildkite_latest_tested_vllm"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    CI_REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${CI_REPO}.git"
else
    echo "[WARN] GITHUB_TOKEN not set — push will likely fail" >&2
    CI_REPO_URL="https://github.com/${CI_REPO}.git"
fi

WORK_DIR="/tmp/pin_vllm_$$"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "--- Preparing ${CI_BRANCH} branch from ${CI_REPO}"
if ! git clone --depth=1 --branch "${CI_BRANCH}" "${CI_REPO_URL}" \
        "${WORK_DIR}" 2>/dev/null; then
    # Branch does not exist yet -- create an orphan with no parent history.
    rm -rf "${WORK_DIR}"
    mkdir -p "${WORK_DIR}"
    git -C "${WORK_DIR}" init -q
    git -C "${WORK_DIR}" remote add origin "${CI_REPO_URL}"
    git -C "${WORK_DIR}" checkout --orphan "${CI_BRANCH}"
    # Drop anything that the orphan checkout might have staged from HEAD.
    git -C "${WORK_DIR}" rm -rf --cached . >/dev/null 2>&1 || true
    find "${WORK_DIR}" -mindepth 1 -maxdepth 1 ! -name ".git" \
        -exec rm -rf {} +
fi

# ── Update files ────────────────────────────────────────────────────────
HISTORY_FILE="${WORK_DIR}/tested_vllm_versions.csv"
LATEST_FILE="${WORK_DIR}/latest_tested_vllm.txt"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUILD_URL="${BUILDKITE_BUILD_URL:-}"
BUILD_NUMBER="${BUILDKITE_BUILD_NUMBER:-}"
COMMIT_SHA="${BUILDKITE_COMMIT:-}"

# Append-only history (JSON Lines). Built via python so quoting is safe.
# Use a quoted heredoc (<<'PY') to disable Bash expansion inside the script,
# and pass variables via the environment to avoid syntax errors from special
# characters in values like BUILD_URL.
TIMESTAMP="${TIMESTAMP}" \
VLLM_VERSION="${VLLM_VERSION}" \
VLLM_SHORT_SHA="${VLLM_SHORT_SHA}" \
VLLM_FULL_SHA="${VLLM_FULL_SHA}" \
VLLM_ARCHIVE_INDEX="${VLLM_ARCHIVE_INDEX}" \
BUILD_NUMBER="${BUILD_NUMBER}" \
BUILD_URL="${BUILD_URL}" \
COMMIT_SHA="${COMMIT_SHA}" \
python - "$HISTORY_FILE" <<'PY'
import json, os, sys
path = sys.argv[1]
record = {
    "timestamp": os.environ.get("TIMESTAMP", ""),
    "vllm_version": os.environ.get("VLLM_VERSION", ""),
    "vllm_short_sha": os.environ.get("VLLM_SHORT_SHA", ""),
    "vllm_full_sha": os.environ.get("VLLM_FULL_SHA", ""),
    "archive_index_url": os.environ.get("VLLM_ARCHIVE_INDEX", ""),
    "build_number": os.environ.get("BUILD_NUMBER", ""),
    "build_url": os.environ.get("BUILD_URL", ""),
    "commit": os.environ.get("COMMIT_SHA", ""),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY

# Latest pointer. The first line is the bare version so older consumers
# that just `head -n1` keep working; the trailing key=value lines let new
# consumers skip the live GitHub API lookup entirely.
{
    printf '%s\n' "${VLLM_VERSION}"
    printf 'short_sha=%s\n' "${VLLM_SHORT_SHA}"
    printf 'full_sha=%s\n' "${VLLM_FULL_SHA}"
    printf 'archive_index_url=%s\n' "${VLLM_ARCHIVE_INDEX}"
} > "${LATEST_FILE}"

# ── Commit + push ───────────────────────────────────────────────────────
cd "${WORK_DIR}"
git add tested_vllm_versions.csv latest_tested_vllm.txt

if git diff --cached --quiet 2>/dev/null; then
    echo "No changes to commit (version unchanged?)."
    exit 0
fi

git -c user.email="ci@lmcache.ai" -c user.name="LMCache CI" \
    commit -m "Pin verified vLLM nightly: ${VLLM_VERSION}"

echo "--- Pushing to ${CI_REPO} ${CI_BRANCH}"
if ! git push origin "HEAD:${CI_BRANCH}" 2>/dev/null; then
    echo "[WARN] Normal push failed, force-pushing..." >&2
    git push origin "+HEAD:${CI_BRANCH}" 2>/dev/null || {
        echo "[ERROR] Failed to push to ${CI_REPO} ${CI_BRANCH}" >&2
        exit 1
    }
fi

echo "--- Pinned vLLM ${VLLM_VERSION} successfully"
git log --oneline -1
