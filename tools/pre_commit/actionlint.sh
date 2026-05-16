#!/usr/bin/env bash

set -euo pipefail

ACTIONLINT_VERSION="1.7.7"

detect_platform() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$os" in
    linux|darwin) ;;
    *)
      echo "Unsupported OS for actionlint bootstrap: $os" >&2
      return 1
      ;;
  esac

  case "$arch" in
    x86_64|amd64)
      arch="amd64"
      ;;
    aarch64|arm64)
      arch="arm64"
      ;;
    *)
      echo "Unsupported architecture for actionlint bootstrap: $arch" >&2
      return 1
      ;;
  esac

  printf '%s_%s' "$os" "$arch"
}

ensure_actionlint() {
  if command -v actionlint >/dev/null 2>&1; then
    command -v actionlint
    return 0
  fi

  local platform cache_root version_dir archive_name archive_url archive_path extracted
  platform="$(detect_platform)"
  cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-hust/tools/actionlint"
  version_dir="$cache_root/$ACTIONLINT_VERSION/$platform"
  extracted="$version_dir/actionlint"

  if [[ -x "$extracted" ]]; then
    printf '%s\n' "$extracted"
    return 0
  fi

  mkdir -p "$version_dir"

  archive_name="actionlint_${ACTIONLINT_VERSION#v}_${platform}.tar.gz"
  archive_url="https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION#v}/$archive_name"
  archive_path="$version_dir/$archive_name"

  curl -fsSL "$archive_url" -o "$archive_path"
  tar -xzf "$archive_path" -C "$version_dir"
  chmod +x "$extracted"

  printf '%s\n' "$extracted"
}

main() {
  local actionlint_bin
  actionlint_bin="$(ensure_actionlint)"
  "$actionlint_bin" -color
}

main "$@"