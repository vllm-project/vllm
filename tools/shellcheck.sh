#!/bin/bash
set -e

scversion="stable"

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

# TODO - fix warnings in .buildkite/run-amd-test.sh
find . -name "*.sh" -not -path "./.buildkite/run-amd-test.sh" -print0 | xargs -0 -I {} sh -c 'git check-ignore -q "{}" || shellcheck "{}"'
