#!/usr/bin/env bash
# Path filter: decide whether a CI build can be skipped based on which files
# changed since the base commit.
#
# Usage:
#   source path-filter.sh
#   if should_skip_ci; then
#       # all changed files are trivial (docs, etc.)
#   fi
#
# Rules:
#   - If ANY changed file lives under .buildkite/, the build runs.
#     (Those PRs are usually fixing the k3 CI itself, so we want to test on
#     the PR instead of waiting for it to land on main.)
#   - Otherwise, if EVERY changed file matches a "trivial" pattern (markdown,
#     LICENSE, anything under docs/, asset/, or .github/, etc.),
#     the build can be
#     skipped. .github/ is trivial for k3 because k3 tests run on Buildkite,
#     not GitHub Actions, so workflow/CODEOWNERS/template changes do not
#     affect what the k3 tests do.
#   - Anything else → build runs.
#
# Opt-out: add a "force-ci" label to the PR on GitHub. Buildkite exposes
# PR labels via BUILDKITE_PULL_REQUEST_LABELS; if "force-ci" is present
# the filter is bypassed and the full pipeline runs.
#
# Detection of "changed files":
#   - PR builds  → diff against the merge-base with BUILDKITE_PULL_REQUEST_BASE_BRANCH.
#   - Push builds → diff HEAD~1..HEAD.
#   - Anything we can't figure out → fall back to "do not skip".

set -uo pipefail

# ── Pattern lists ─────────────────────────────────────────────
# bash `case` patterns: `*` matches any string including `/`, so `docs/*`
# matches `docs/foo/bar.png` as well as `docs/foo`.

_path_filter_is_always_trigger() {
    case "$1" in
        .buildkite/*) return 0 ;;
    esac
    return 1
}

_path_filter_is_trivial() {
    case "$1" in
        *.md) return 0 ;;
        LICENSE|LICENSE.*) return 0 ;;
        NOTICE|NOTICE.*) return 0 ;;
        .gitignore|.gitattributes|.editorconfig|.mailmap) return 0 ;;
        CODEOWNERS) return 0 ;;
        docs/*) return 0 ;;
        asset/*) return 0 ;;
        .github/*) return 0 ;;
    esac
    return 1
}

# ── Changed-files detection ───────────────────────────────────

_path_filter_get_changed_files() {
    local base_branch base merge_base

    # Ephemeral pods may not have GitHub's SSH host key yet.
    # Accept new keys automatically so git-fetch doesn't hang on a prompt.
    export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"

    if [[ -n "${BUILDKITE_PULL_REQUEST:-}" && "${BUILDKITE_PULL_REQUEST:-}" != "false" ]]; then
        base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
        # Buildkite checks out shallow; fetch enough history to find the merge-base.
        git fetch --no-tags --depth=200 origin "$base_branch" 2>/dev/null || \
            git fetch --no-tags origin "$base_branch" 2>/dev/null || true

        if base=$(git rev-parse --verify "origin/${base_branch}" 2>/dev/null); then
            if merge_base=$(git merge-base HEAD "$base" 2>/dev/null); then
                git diff --name-only "$merge_base" HEAD
                return 0
            fi
            # No merge-base (history not deep enough): diff directly.
            git diff --name-only "$base" HEAD
            return 0
        fi
        echo "path-filter: could not resolve origin/${base_branch}" >&2
        return 1
    fi

    # Push build (or unknown context): diff against the previous commit.
    if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        git diff --name-only HEAD~1 HEAD
        return 0
    fi

    echo "path-filter: no parent commit available" >&2
    return 1
}

# ── Public entry point ────────────────────────────────────────

# Returns 0 if the build can be safely skipped, non-zero otherwise.
# Prints a classification of every changed file to stderr for the build log.
should_skip_ci() {
    # PR label opt-out: adding "force-ci" on GitHub forces a full run.
    if [[ ",${BUILDKITE_PULL_REQUEST_LABELS:-}," == *",force-ci,"* ]]; then
        echo "path-filter: PR has 'force-ci' label → not skipping" >&2
        return 1
    fi

    # Never skip scheduled builds (e.g. nightly baselines with NEED_UPLOAD=true).
    if [[ "${BUILDKITE_SOURCE:-}" == "schedule" ]]; then
        echo "path-filter: scheduled build (BUILDKITE_SOURCE=schedule) → not skipping" >&2
        return 1
    fi

    local changed_files
    if ! changed_files=$(_path_filter_get_changed_files); then
        echo "path-filter: could not determine changed files → not skipping" >&2
        return 1
    fi

    if [[ -z "$changed_files" ]]; then
        echo "path-filter: no changed files reported → not skipping (safer default)" >&2
        return 1
    fi

    local has_always_trigger=0
    local has_non_trivial=0
    local trivial_count=0
    local total=0

    echo "path-filter: classifying changed files:" >&2
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        total=$((total + 1))
        if _path_filter_is_always_trigger "$f"; then
            has_always_trigger=1
            echo "  [force-trigger] $f" >&2
        elif _path_filter_is_trivial "$f"; then
            trivial_count=$((trivial_count + 1))
            echo "  [trivial]       $f" >&2
        else
            has_non_trivial=1
            echo "  [non-trivial]   $f" >&2
        fi
    done <<< "$changed_files"

    echo "path-filter: ${total} files changed (${trivial_count} trivial)" >&2

    if [[ "$has_always_trigger" -eq 1 ]]; then
        echo "path-filter: at least one file under .buildkite/ → not skipping" >&2
        return 1
    fi

    if [[ "$has_non_trivial" -eq 1 ]]; then
        echo "path-filter: non-trivial files changed → not skipping" >&2
        return 1
    fi

    echo "path-filter: all changed files are trivial → SKIP" >&2
    return 0
}
