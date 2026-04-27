#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  marin_release.sh resolve-validate-run <repo> <workflow> <branch> [target_sha]
  marin_release.sh download-validate-artifact <repo> <run_id> <artifact_name> <dest>
  marin_release.sh download-release-assets <repo> <tag> <dest> [pattern...]
  marin_release.sh move-tag <repo> <tag> <sha>
  marin_release.sh upsert-release <repo> <tag> <target_sha> <prerelease:true|false> <title> <notes_file> [asset...]
EOF
}

wait_for_release() {
  local repo=$1
  local tag=$2

  for _ in $(seq 1 10); do
    if gh release view "$tag" -R "$repo" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for release $repo:$tag" >&2
  return 1
}

resolve_validate_run() {
  local repo=$1
  local workflow=$2
  local branch=$3
  local target_sha=${4:-}

  local runs_json
  runs_json=$(gh run list \
    -R "$repo" \
    -w "$workflow" \
    -b "$branch" \
    -L 100 \
    --json databaseId,headSha,conclusion)

  local run_id
  if [[ -n "$target_sha" ]]; then
    run_id=$(jq -r \
      --arg sha "$target_sha" \
      'map(select(.conclusion == "success" and .headSha == $sha))[0].databaseId // empty' \
      <<<"$runs_json")
  else
    run_id=$(jq -r \
      'map(select(.conclusion == "success"))[0].databaseId // empty' \
      <<<"$runs_json")
  fi

  if [[ -z "$run_id" ]]; then
    echo "No successful $workflow run found in $repo for branch $branch" >&2
    if [[ -n "$target_sha" ]]; then
      echo "Target SHA: $target_sha" >&2
    fi
    return 1
  fi

  printf '%s\n' "$run_id"
}

download_validate_artifact() {
  local repo=$1
  local run_id=$2
  local artifact_name=$3
  local dest=$4

  rm -rf "$dest"
  mkdir -p "$dest"
  gh run download "$run_id" -R "$repo" -n "$artifact_name" -D "$dest"
}

download_release_assets() {
  local repo=$1
  local tag=$2
  local dest=$3
  shift 3

  rm -rf "$dest"
  mkdir -p "$dest"

  local args=()
  for pattern in "$@"; do
    args+=(--pattern "$pattern")
  done

  gh release download "$tag" -R "$repo" -D "$dest" --clobber "${args[@]}"
}

move_tag() {
  local repo=$1
  local tag=$2
  local sha=$3

  if gh api "repos/$repo/git/ref/tags/$tag" >/dev/null 2>&1; then
    gh api \
      --method PATCH \
      "repos/$repo/git/refs/tags/$tag" \
      -f sha="$sha" \
      -F force=true >/dev/null
  else
    gh api \
      --method POST \
      "repos/$repo/git/refs" \
      -f ref="refs/tags/$tag" \
      -f sha="$sha" >/dev/null
  fi
}

upsert_release() {
  local repo=$1
  local tag=$2
  local target_sha=$3
  local prerelease=$4
  local title=$5
  local notes_file=$6
  shift 6

  local prerelease_json=false
  if [[ "$prerelease" == "true" ]]; then
    prerelease_json=true
  fi

  local payload
  payload=$(
    jq -n \
      --arg tag "$tag" \
      --arg target "$target_sha" \
      --arg name "$title" \
      --arg body "$(cat "$notes_file")" \
      --argjson prerelease "$prerelease_json" \
      '{
        tag_name: $tag,
        target_commitish: $target,
        name: $name,
        body: $body,
        prerelease: $prerelease,
        draft: false
      }'
  )

  local release_json release_id
  if release_json=$(gh api "repos/$repo/releases/tags/$tag" 2>/dev/null); then
    release_id=$(jq -r '.id' <<<"$release_json")
    gh api --method PATCH "repos/$repo/releases/$release_id" --input - <<<"$payload" >/dev/null
  else
    gh api --method POST "repos/$repo/releases" --input - <<<"$payload" >/dev/null
  fi

  wait_for_release "$repo" "$tag"

  if [[ $# -gt 0 ]]; then
    gh api "repos/$repo/releases/tags/$tag" \
      | jq -r '.assets[].id // empty' \
      | while read -r asset_id; do
          [[ -n "$asset_id" ]] || continue
          gh api --method DELETE "repos/$repo/releases/assets/$asset_id" >/dev/null
        done

    gh release upload "$tag" -R "$repo" --clobber "$@"
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd=$1
  shift

  case "$cmd" in
    resolve-validate-run)
      resolve_validate_run "$@"
      ;;
    download-validate-artifact)
      download_validate_artifact "$@"
      ;;
    download-release-assets)
      download_release_assets "$@"
      ;;
    move-tag)
      move_tag "$@"
      ;;
    upsert-release)
      upsert_release "$@"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
