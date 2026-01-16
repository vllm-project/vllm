#!/bin/bash
set -euo pipefail

scversion="stable"
baseline="tools/pre_commit/shellcheck.baseline"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
# collects warnings as "file:SCcode" pairs for baseline comparison.
collect() {
  find . -path ./.git -prune -o -name "*.sh" \
    -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0 | \
    xargs -0 sh -c 'for f in "$@"; do git check-ignore -q "$f" || shellcheck -s bash -f gcc "$f" || true; done' -- | \
    sed -nE 's|^\./||; s|^([^:]+):[0-9]+:[0-9]+:.*\[(SC[0-9]+)\]$|\1:\2|p' | \
    sort -u
}

if [[ "${1:-}" == "--generate-baseline" ]]; then
  collect > "$baseline"
  echo "Wrote baseline to $baseline"
  exit 0
fi

if [[ ! -f "$baseline" ]]; then
  echo "Baseline not found: $baseline (run: $0 --generate-baseline)"
  exit 1
fi

current="$(mktemp)"
trap 'rm -f "$current"' EXIT
collect > "$current"

# finds new warnings not in baseline
new_errors="$(comm -23 "$current" <(sort -u "$baseline") || true)"
if [ -n "$new_errors" ]; then
  echo "$new_errors" | cut -d: -f1 | sort -u | while IFS= read -r file; do
    if [[ -f "$file" ]]; then
      shellcheck -s bash "$file" 2>&1 || true
    fi
  done
  exit 1
fi
