
#!/bin/bash

set -euo pipefail

if [[ -z "${RUN_ALL:-}" ]]; then
    RUN_ALL=0
fi

if [[ -z "${NIGHTLY:-}" ]]; then
    NIGHTLY=0
fi

if [[ -z "${VLLM_CI_BRANCH:-}" ]]; then
    VLLM_CI_BRANCH="main"
fi

if [[ -z "${AMD_MIRROR_HW:-}" ]]; then
    AMD_MIRROR_HW="amdproduction"
fi

if [[ -z "${DOCS_ONLY_DISABLE:-}" ]]; then
    DOCS_ONLY_DISABLE=0
fi

if [[ -z "${COV_ENABLED:-}" ]]; then
    COV_ENABLED=0
fi

# ---------------------------------------------------------------------------
# Git setup: ensure origin/main is available and compute merge base once.
# On K8s (blobless clones with --filter=blob:none), origin/main may not be
# fetched yet if the agent only fetched the PR branch. Also mark the repo
# as safe in case the checkout uid differs from the running uid.
# ---------------------------------------------------------------------------
git config --global --add safe.directory "$(pwd)" 2>/dev/null || true

if ! git rev-parse --verify origin/main >/dev/null 2>&1; then
    echo "origin/main not found, fetching..."
    git fetch origin main --depth=1 2>/dev/null || git fetch origin main || true
fi

if [[ -z "${MERGE_BASE_COMMIT:-}" ]]; then
    MERGE_BASE_COMMIT=$(git merge-base origin/main HEAD 2>/dev/null || echo "")
    if [[ -z "$MERGE_BASE_COMMIT" ]]; then
        echo "WARNING: Could not compute merge base, falling back to run_all=1"
        RUN_ALL=1
        MERGE_BASE_COMMIT="HEAD"
    fi
fi

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

fail_fast() {
    DISABLE_LABEL="ci-no-fail-fast"
    # If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
    if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
        PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')
        if [[ $PR_LABELS == *"$DISABLE_LABEL"* ]]; then
            echo false
        else
            echo true
        fi
    else
        echo false  # not a PR or BUILDKITE_PULL_REQUEST not set
    fi
}

check_run_all_label() {
    RUN_ALL_LABEL="ready-run-all-tests"
    # If BUILDKITE_PULL_REQUEST != "false", then we check the PR labels using curl and jq
    if [ "$BUILDKITE_PULL_REQUEST" != "false" ]; then
        PR_LABELS=$(curl -s "https://api.github.com/repos/vllm-project/vllm/pulls/$BUILDKITE_PULL_REQUEST" | jq -r '.labels[].name')
        if [[ $PR_LABELS == *"$RUN_ALL_LABEL"* ]]; then
            echo true
        else
            echo false
        fi
    else
        echo false  # not a PR or BUILDKITE_PULL_REQUEST not set
    fi
}

# ---------------------------------------------------------------------------
# get_diff: compute changed files between commits only (no index staging).
#
# IMPORTANT: Do NOT use "git add ." here. On K8s blobless clones
# (--filter=blob:none), git add . fetches and stages the entire repo,
# producing a diff with every file and eventually exceeding ARG_MAX.
# We only need committed changes between refs.
# ---------------------------------------------------------------------------
get_diff() {
    git diff --name-only --diff-filter=ACMDR "$MERGE_BASE_COMMIT" HEAD 2>/dev/null || echo ""
}

get_diff_main() {
    git diff --name-only --diff-filter=ACMDR HEAD~1 HEAD 2>/dev/null || echo ""
}

# ---------------------------------------------------------------------------
# upload_pipeline: render and upload the Buildkite pipeline YAML
# ---------------------------------------------------------------------------
upload_pipeline() {
    echo "Uploading pipeline..."
    # Install minijinja
    ls .buildkite || buildkite-agent annotate --style error 'Please merge upstream main branch for buildkite CI'
    curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
    source "$HOME/.cargo/env"

    if [[ $BUILDKITE_PIPELINE_SLUG == "fastcheck" ]]; then
        AMD_MIRROR_HW="amdtentative"
        curl -o .buildkite/test-template.j2 \
            "https://raw.githubusercontent.com/vllm-project/ci-infra/$VLLM_CI_BRANCH/buildkite/test-template-amd.j2?$(date +%s)"
    else
        curl -o .buildkite/test-template.j2 \
            "https://raw.githubusercontent.com/vllm-project/ci-infra/$VLLM_CI_BRANCH/buildkite/test-template-amd.j2?$(date +%s)"
    fi


    # (WIP) Use pipeline generator instead of jinja template
    if [ -e ".buildkite/pipeline_generator/pipeline_generator.py" ]; then
        python -m pip install click pydantic
        python .buildkite/pipeline_generator/pipeline_generator.py --run_all=$RUN_ALL --list_file_diff="$LIST_FILE_DIFF" --nightly="$NIGHTLY" --mirror_hw="$AMD_MIRROR_HW"
        buildkite-agent pipeline upload .buildkite/pipeline.yaml
        exit 0
    fi
    echo "List file diff: $LIST_FILE_DIFF"
    echo "Run all: $RUN_ALL"
    echo "Nightly: $NIGHTLY"
    echo "AMD Mirror HW: $AMD_MIRROR_HW"

    FAIL_FAST=$(fail_fast)

    cd .buildkite
    (
        set -x
        # Output pipeline.yaml with all blank lines removed
        minijinja-cli test-template.j2 test-amd.yaml \
            -D branch="$BUILDKITE_BRANCH" \
            -D list_file_diff="$LIST_FILE_DIFF" \
            -D run_all="$RUN_ALL" \
            -D nightly="$NIGHTLY" \
            -D mirror_hw="$AMD_MIRROR_HW" \
            -D fail_fast="$FAIL_FAST" \
            -D vllm_use_precompiled="$VLLM_USE_PRECOMPILED" \
            -D vllm_merge_base_commit="$MERGE_BASE_COMMIT" \
            -D cov_enabled="$COV_ENABLED" \
            -D vllm_ci_branch="$VLLM_CI_BRANCH" \
            | sed '/^[[:space:]]*$/d' \
            > pipeline.yaml
    )
    cat pipeline.yaml
    buildkite-agent artifact upload pipeline.yaml
    buildkite-agent pipeline upload pipeline.yaml
    exit 0
}

# ---------------------------------------------------------------------------
# Compute file diff
# ---------------------------------------------------------------------------
file_diff=$(get_diff)
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    file_diff=$(get_diff_main)
fi

# ----------------------------------------------------------------------
# Early exit start: skip pipeline if conditions are met
# ----------------------------------------------------------------------

# skip pipeline if *every* changed file is docs/** OR **/*.md OR mkdocs.yaml
if [[ "${DOCS_ONLY_DISABLE}" != "1" ]]; then
  if [[ -n "${file_diff:-}" ]]; then
    docs_only=1
    # Iterate robustly over newline-separated paths
    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      # Match any of: docs/**  OR  **/*.md  OR  mkdocs.yaml
      if [[ "${f#docs/}" != "$f" || "$f" == *.md || "$f" == "mkdocs.yaml" ]]; then
        continue
      else
        docs_only=0
        break
      fi
    done < <(printf '%s\n' "$file_diff" | tr ' ' '\n' | tr -d '\r')

    if [[ "$docs_only" -eq 1 ]]; then
      buildkite-agent annotate ":memo: CI skipped — docs/Markdown/mkdocs-only changes detected

\`\`\`
$(printf '%s\n' "$file_diff" | tr ' ' '\n')
\`\`\`" --style "info" || true
      echo "[docs-only] All changes are docs/**, *.md, or mkdocs.yaml. Exiting before pipeline upload."
      exit 0
    fi
  fi
fi

# ----------------------------------------------------------------------
# Early exit end
# ----------------------------------------------------------------------

patterns=(
    "docker/Dockerfile.rocm"
    "docker/Dockerfile.rocm_base"
    "CMakeLists.txt"
    "requirements/common.txt"
    "requirements/cuda.txt"
    "requirements/build.txt"
    "requirements/test.txt"
    "requirements/rocm.txt"
    "requirements/rocm-build.txt"
    "requirements/rocm-test.txt"
    "setup.py"
    "csrc/"
    "cmake/"
)

ignore_patterns=(
    "csrc/cpu"
    "csrc/rocm"
    "cmake/hipify.py"
    "cmake/cpu_extension.cmake"
)

for file in $file_diff; do
    # First check if file matches any pattern
    matches_pattern=0
    for pattern in "${patterns[@]}"; do
        if [[ $file == $pattern* ]] || [[ $file == $pattern ]]; then
            matches_pattern=1
            break
        fi
    done

    # If file matches pattern, check it's not in ignore patterns
    if [[ $matches_pattern -eq 1 ]]; then
        matches_ignore=0
        for ignore in "${ignore_patterns[@]}"; do
            if [[ $file == $ignore* ]] || [[ $file == $ignore ]]; then
                matches_ignore=1
                break
            fi
        done

        if [[ $matches_ignore -eq 0 ]]; then
            RUN_ALL=1
            echo "Found changes: $file. Run all tests"
            break
        fi
    fi
done

# Check for ready-run-all-tests label
LABEL_RUN_ALL=$(check_run_all_label)
if [[ $LABEL_RUN_ALL == true ]]; then
    RUN_ALL=1
    NIGHTLY=1
    echo "Found 'ready-run-all-tests' label. Running all tests including optional tests."
fi

# Decide whether to use precompiled wheels
# Relies on existing patterns array as a basis.
if [[ -n "${VLLM_USE_PRECOMPILED:-}" ]]; then
    echo "VLLM_USE_PRECOMPILED is already set to: $VLLM_USE_PRECOMPILED"
elif [[ $RUN_ALL -eq 1 ]]; then
    export VLLM_USE_PRECOMPILED=0
    echo "Detected critical changes, building wheels from source"
else
    export VLLM_USE_PRECOMPILED=1
    echo "No critical changes, using precompiled wheels"
fi

# Build LIST_FILE_DIFF from the already-computed file_diff.
# When run_all=1, the jinja template ignores list_file_diff (all tests are
# unblocked unconditionally), so use a short sentinel to avoid exceeding
# ARG_MAX when the diff is large (K8s fresh clones).
if [[ $RUN_ALL -eq 1 ]]; then
    LIST_FILE_DIFF="run_all"
else
    LIST_FILE_DIFF=$(echo "$file_diff" | tr ' ' '|')
fi

upload_pipeline
