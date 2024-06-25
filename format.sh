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

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
MYPY_VERSION=$(mypy --version | awk '{print $2}')
CODESPELL_VERSION=$(codespell --version)
ISORT_VERSION=$(isort --vn)
CLANGFORMAT_VERSION=$(clang-format --version | awk '{print $3}')

# # params: tool name, tool version, required version
tool_version_check() {
    if [[ $2 != $3 ]]; then
        echo "Wrong $1 version installed: $3 is required, not $2."
        exit 1
    fi
}

tool_version_check "yapf" $YAPF_VERSION "$(grep yapf requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "ruff" $RUFF_VERSION "$(grep "ruff==" requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "mypy" "$MYPY_VERSION" "$(grep mypy requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "isort" "$ISORT_VERSION" "$(grep isort requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "codespell" "$CODESPELL_VERSION" "$(grep codespell requirements-dev.txt | cut -d'=' -f3)"
tool_version_check "clang-format" "$CLANGFORMAT_VERSION" "$(grep clang-format requirements-dev.txt | cut -d'=' -f3)"

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
mypy vllm/attention --config-file pyproject.toml
mypy vllm/core --config-file pyproject.toml
mypy vllm/distributed --config-file pyproject.toml
mypy vllm/entrypoints --config-file pyproject.toml
mypy vllm/executor --config-file pyproject.toml
mypy vllm/multimodal --config-file pyproject.toml
mypy vllm/usage --config-file pyproject.toml
mypy vllm/*.py --config-file pyproject.toml
mypy vllm/transformers_utils --config-file pyproject.toml
mypy vllm/engine  --config-file pyproject.toml
mypy vllm/worker --config-file pyproject.toml
mypy vllm/spec_decode --config-file pyproject.toml
mypy vllm/model_executor  --config-file pyproject.toml
mypy vllm/lora --config-file pyproject.toml
mypy vllm/logging --config-file pyproject.toml
mypy vllm/model_executor --config-file pyproject.toml


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

# Spelling  check of files that differ from main branch.
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
    ruff "$@"
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
             ruff
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
    'csrc/punica/bgmv/bgmv_bf16_bf16_bf16.cu'
    'csrc/punica/bgmv/bgmv_config.h'
    'csrc/punica/bgmv/bgmv_impl.cuh'
    'csrc/punica/bgmv/vec_dtypes.cuh'
    'csrc/punica/punica_ops.cu'
    'csrc/punica/type_convert.h'
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
    changed_files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.h' '*.cpp' '*.cu' '*.cuh' | grep -vFf <(printf "%s\n" "${CLANG_FORMAT_EXCLUDES[@]}"))
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


if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi
