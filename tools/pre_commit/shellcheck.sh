#!/bin/bash
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        # automatic local install if linux x86_64
        wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
        export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
    else
        # Skip when shellcheck is not installed (e.g. on macOS) so pre-commit doesn't fail.
        # Install manually to lint shell scripts: https://github.com/koalaman/shellcheck#installing
        echo "shellcheck not found, skipping lint (install to lint: brew install shellcheck)"
        exit 0
    fi
fi

# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
find . -path ./.git -prune -o -name "*.sh" \
  -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0 | \
  xargs -0 sh -c "for f in \"\$@\"; do git check-ignore -q \"\$f\" || shellcheck -s bash \"\$f\"; done" --
