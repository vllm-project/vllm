#!/usr/bin/env bash
# YAPF formatter, adapted from ray and skypilot.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❓❓$1 is not installed, please run \`pip install -r requirements-lint.txt\`"
        exit 1
    fi
}

check_command yapf
check_command ruff
check_command mypy
check_command codespell
check_command isort
check_command clang-format

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
MYPY_VERSION=$(mypy --version | awk '{print $2}')
CODESPELL_VERSION=$(codespell --version)
ISORT_VERSION=$(isort --vn)
CLANGFORMAT_VERSION=$(clang-format --version | awk '{print $3}')
SPHINX_LINT_VERSION=$(sphinx-lint --version | awk '{print $2}')

# # params: tool name, tool version, required version
tool_version_check() {
    expected=$(grep "$1" requirements-lint.txt | cut -d'=' -f3)
    if [[ "$2" != "$expected" ]]; then
        echo "❓❓Wrong $1 version installed: $expected is required, not $2."
        exit 1
    fi
}

tool_version_check "yapf" "$YAPF_VERSION"
tool_version_check "ruff" "$RUFF_VERSION"
tool_version_check "mypy" "$MYPY_VERSION"
tool_version_check "isort" "$ISORT_VERSION"
tool_version_check "codespell" "$CODESPELL_VERSION"
tool_version_check "clang-format" "$CLANGFORMAT_VERSION"
tool_version_check "sphinx-lint" "$SPHINX_LINT_VERSION"

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi

}

# Format all files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'vLLM yapf: Done'

# Run mypy
echo 'vLLM mypy:'
tools/mypy.sh
echo 'vLLM mypy: Done'


# If git diff returns a file that is in the skip list, the file may be checked anyway:
# https://github.com/codespell-project/codespell/issues/1915
# Avoiding the "./" prefix and using "/**" globs for directories appears to solve the problem
CODESPELL_EXCLUDES=(
    '--skip' 'tests/prompts/**,./benchmarks/sonnet.txt,*tests/lora/data/**,build/**'
)

# check spelling of specified files
spell_check() {
    codespell "$@"
}

spell_check_all(){
  codespell --toml pyproject.toml "${CODESPELL_EXCLUDES[@]}"
}

# Spelling check of files that differ from main branch.
spell_check_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
            codespell "${CODESPELL_EXCLUDES[@]}"
    fi
}

# Run Codespell
## This flag runs spell check of individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   spell_check "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   spell_check_all
else
   # Check spelling only of the files that changed in last commit.
   spell_check_changed
fi
echo 'vLLM codespell: Done'


# Lint specified files
lint() {
    ruff check "$@"
}

# Lint files that differ from main branch. Ignores dirs that are not slated
# for autolint yet.
lint_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check
    fi

}

# Run Ruff
### This flag lints individual files. --files *must* be the first command line
### arg to use this option.
if [[ "$1" == '--files' ]]; then
   lint "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   lint vllm tests
else
   # Format only the files that changed in last commit.
   lint_changed
fi
echo 'vLLM ruff: Done'

# check spelling of specified files
isort_check() {
    isort "$@"
}

isort_check_all(){
  isort .
}

# Spelling  check of files that differ from main branch.
isort_check_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             isort
    fi
}

# Run Isort
# This flag runs spell check of individual files. --files *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
   isort_check "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   isort_check_all
else
   # Check spelling only of the files that changed in last commit.
   isort_check_changed
fi
echo 'vLLM isort: Done'

# Clang-format section
# Exclude some files for formatting because they are vendored
# NOTE: Keep up to date with .github/workflows/clang-format.yml
CLANG_FORMAT_EXCLUDES=(
    'csrc/moe/topk_softmax_kernels.cu'
    'csrc/quantization/gguf/ggml-common.h'
    'csrc/quantization/gguf/dequantize.cuh'
    'csrc/quantization/gguf/vecdotq.cuh'
    'csrc/quantization/gguf/mmq.cuh'
    'csrc/quantization/gguf/mmvq.cuh'
)

# Format specified files with clang-format
clang_format() {
    clang-format -i "$@"
}

# Format files that differ from main branch with clang-format.
clang_format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause clang-format to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    # Get the list of changed files, excluding the specified ones
    changed_files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.h' '*.cpp' '*.cu' '*.cuh' | (grep -vFf <(printf "%s\n" "${CLANG_FORMAT_EXCLUDES[@]}") || echo -e))
    if [ -n "$changed_files" ]; then
        echo "$changed_files" | xargs -P 5 clang-format -i
    fi
}

# Format all files with clang-format
clang_format_all() {
    find csrc/ \( -name '*.h' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print \
        | grep -vFf <(printf "%s\n" "${CLANG_FORMAT_EXCLUDES[@]}") \
        | xargs clang-format -i
}

# Run clang-format
if [[ "$1" == '--files' ]]; then
   clang_format "${@:2}"
elif [[ "$1" == '--all' ]]; then
   clang_format_all
else
   clang_format_changed
fi
echo 'vLLM clang-format: Done'

echo 'vLLM actionlint:'
tools/actionlint.sh -color
echo 'vLLM actionlint: Done'

echo 'vLLM shellcheck:'
tools/shellcheck.sh
echo 'vLLM shellcheck: Done'

echo 'excalidraw png check:'
tools/png-lint.sh
echo 'excalidraw png check: Done'

if ! git diff --quiet &>/dev/null; then
    echo 
    echo "🔍🔍There are files changed by the format checker or by you that are not added and committed:"
    git --no-pager diff --name-only
    echo "🔍🔍Format checker passed, but please add, commit and push all the files above to include changes made by the format checker."

    exit 1
else
    echo "✨🎉 Format check passed! Congratulations! 🎉✨"
fi

echo 'vLLM sphinx-lint:'
tools/sphinx-lint.sh
echo 'vLLM sphinx-lint: Done'
