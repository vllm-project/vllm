#!/usr/bin/env bash
# Release GeometricAGI/vllm once per stable upstream vLLM release.
#
# For each stable upstream tag (vX.Y.Z or vX.Y.Z.postN) at or above MIN_TAG that
# has no fork release yet, branch from the upstream tag, cherry-pick the fork's
# own commits on top, tag it vX.Y.Z+geo.N and publish a GitHub release.
#
# Environment:
#   MIN_TAG   oldest upstream tag to release (default v0.31.0)
#   TAG       release only this upstream tag, as a new +geo.N even if one exists
#   DRY_RUN   "true" to stop before anything is pushed or published
#   (uv must be on PATH for the syntax check)
#   GH_TOKEN  token that may push tags, branches and workflow files, and write
#             releases and issues
set -euo pipefail

MIN_TAG="${MIN_TAG:-v0.31.0}"
TAG="${TAG:-}"
DRY_RUN="${DRY_RUN:-false}"
STABLE_RE='^v[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$'

git config user.name "geometric-release-bot"
git config user.email "release-bot@geometric.so"

git remote get-url upstream >/dev/null 2>&1 \
  || git remote add upstream https://github.com/vllm-project/vllm.git
git fetch --quiet upstream main
git fetch --quiet --force upstream 'refs/tags/*:refs/tags/*'
git fetch --quiet origin main

# The fork's own work: every non-merge commit on the fork's main that upstream's
# main does not have, oldest first.
mapfile -t PATCHES < <(git rev-list --reverse --no-merges upstream/main..origin/main)
echo "Fork patches to carry: ${#PATCHES[@]}"

# Highest +geo.N already published for an upstream tag (0 if none).
latest_geo() {
  git ls-remote --tags origin "refs/tags/$1+geo.*" \
    | sed -E 's#.*\+geo\.([0-9]+)(\^\{\})?$#\1#' | sort -n | tail -1 | grep . || echo 0
}

# Open an issue about a failed release unless one is already open.
report_failure() {
  local tag="$1" reason="$2" title
  title="Release sync failed for upstream ${tag}"
  if [ "$(gh issue list --state open --search "\"${title}\" in:title" --json number --jq length)" = "0" ]; then
    gh issue create --title "${title}" --body "${reason}

Resolve it on a branch from \`${tag}\`, then run the **Sync upstream releases** workflow with \`tag=${tag}\`.
Run: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-}"
  fi
}

# Build and publish the release for one upstream tag. Returns non-zero on failure.
release_one() {
  local tag="$1" n new branch
  n=$(( $(latest_geo "$tag") + 1 ))
  new="${tag}+geo.${n}"
  branch="release/${tag}"
  echo "== ${tag} -> ${new}"

  git checkout --quiet -B "${branch}" "refs/tags/${tag}"
  for sha in "${PATCHES[@]}"; do
    if ! git cherry-pick -x --empty=drop "${sha}" >/dev/null 2>&1; then
      git cherry-pick --abort >/dev/null 2>&1 || true
      report_failure "${tag}" "Cherry-picking fork commit \`${sha}\` ($(git log -1 --format=%s "${sha}")) onto \`${tag}\` conflicts."
      return 1
    fi
  done

  # Light checks on what the fork changed: whitespace errors and Python syntax.
  git diff --check "refs/tags/${tag}" HEAD
  git diff --name-only --diff-filter=AM "refs/tags/${tag}" HEAD -- '*.py' \
    | xargs -r uv run --no-project --python 3.12 python -m py_compile

  if [ "${DRY_RUN}" = "true" ]; then
    echo "Dry run: would publish ${new} at $(git rev-parse --short HEAD)"
    return 0
  fi

  git tag -a "${new}" -m "vLLM ${tag} plus GeometricAGI changes"
  git push --force origin "${branch}"
  git push origin "refs/tags/${new}"
  gh release create "${new}" --verify-tag --latest=false --title "${new}" \
    --notes "Upstream vLLM [${tag}](https://github.com/vllm-project/vllm/releases/tag/${tag}) plus ${#PATCHES[@]} GeometricAGI commit(s)."
}

if [ -n "${TAG}" ]; then
  [[ "${TAG}" =~ ${STABLE_RE} ]] || { echo "TAG must look like v1.2.3 or v1.2.3.post1" >&2; exit 2; }
  release_one "${TAG}"
  exit
fi

failed=0
while read -r tag; do
  [ "$(printf '%s\n%s\n' "${MIN_TAG}" "${tag}" | sort -V | head -1)" = "${MIN_TAG}" ] || continue
  [ "$(latest_geo "${tag}")" = "0" ] || continue
  release_one "${tag}" || failed=1
done < <(git tag -l | grep -E "${STABLE_RE}" | sort -V)
exit "${failed}"
