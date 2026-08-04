#!/usr/bin/env bash
set -euo pipefail

# Decide whether comprehensive tests should run for this build.
# Exit code semantics:
#   0 -> run comprehensive tests
#   1 -> skip comprehensive tests (safe paths only)

# If this isn't a PR build, always run comprehensive tests.
if [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
  exit 0
fi

# Determine base ref to diff against.
BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-dev}"
BASE_REF="origin/${BASE_BRANCH}"

# Ensure the base ref exists; log a warning on fetch failure.
if ! git fetch origin "${BASE_BRANCH}" >/dev/null 2>&1; then
  echo "Warning: failed to fetch origin/${BASE_BRANCH}, falling back to local ref for diff." >&2
fi

# Compute changed files between the base and current HEAD.
# If this fails, we will fall back to running the tests (CHANGED_FILES will be empty).
CHANGED_FILES=$(git diff --name-only "${BASE_REF}...HEAD" 2>/dev/null || echo "")

# If we cannot determine changes, be conservative and run tests.
if [[ -z "${CHANGED_FILES}" ]]; then
  exit 0
fi

# Track which safe-path categories were touched.
safe_categories=()

add_safe_category() {
  local cat="$1"
  for existing in "${safe_categories[@]}"; do
    if [[ "$existing" == "$cat" ]]; then
      return
    fi
  done
  safe_categories+=("$cat")
}

# If any changed file is NOT in a safe path, we must run tests.
while IFS= read -r f; do
  case "$f" in
    docs/*)
      add_safe_category "docs/"
      ;;
    *.md|*.rst)
      add_safe_category "*.md/*.rst"
      ;;
    tests/*)
      add_safe_category "tests/"
      ;;
    tools/*)
      add_safe_category "tools/"
      ;;
    lmcache/tools/*)
      add_safe_category "lmcache/tools/"
      ;;
    examples/*)
      add_safe_category "examples/"
      ;;
    asset/*)
      add_safe_category "asset/"
      ;;
    *)
      # Non-safe file touched -> run comprehensive tests.
      exit 0
      ;;
  esac
done <<< "${CHANGED_FILES}"

# If we reach here, all changes are in safe paths, so safe_categories is not empty.
# Join the array of categories for a clean log message.
joined_cats=$(IFS=', '; echo "${safe_categories[*]}")
echo "${joined_cats} change detected; skipping comprehensive tests."

exit 1
