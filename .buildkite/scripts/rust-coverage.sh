#!/bin/sh

RUST_CODECOV_VERSION="v11.3.1"
RUST_CODECOV_SHA256="ca1d64196d2d34771084afe76ea657d581bf628e31d993ff8e52ea09cc88a56d"

rust_coverage_repo_root() {
    if [ -f /vllm-workspace/.buildkite/scripts/rust-coverage.sh ]; then
        printf '%s\n' /vllm-workspace
    elif [ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ] \
        && [ -d "$BUILDKITE_BUILD_CHECKOUT_PATH" ]; then
        printf '%s\n' "$BUILDKITE_BUILD_CHECKOUT_PATH"
    else
        git rev-parse --show-toplevel
    fi
}

rust_coverage_start() {
    RUST_COVERAGE_FLAG=${1:?coverage flag is required}
    RUST_COVERAGE_DIR="/tmp/vllm-rust-coverage/${BUILDKITE_JOB_ID:-local}"
    export RUST_COVERAGE_FLAG RUST_COVERAGE_DIR
    mkdir -p "$RUST_COVERAGE_DIR"
    LLVM_PROFILE_FILE="$RUST_COVERAGE_DIR/rust-%4m.profraw"
    export LLVM_PROFILE_FILE
    trap rust_coverage_finalize 0
}

rust_coverage_objects() {
    rust_cov_objects_manifest="$(dirname "$(command -v llvm-cov)")/../objects"
    python3 - "$rust_cov_objects_manifest" <<'PY'
from pathlib import Path
import sys

for relative in Path(sys.argv[1]).read_text().splitlines():
    for entry in sys.path:
        path = Path(entry or ".").resolve() / relative
        if path.is_file():
            print(path)
            break
    else:
        raise RuntimeError(f"installed Rust coverage object was not found: {relative}")
PY
}

rust_coverage_collect() {
    rust_cov_collect_flag=${1:?coverage flag is required}
    rust_cov_collect_lcov="$RUST_COVERAGE_DIR/$rust_cov_collect_flag.lcov"

    rust_cov_collect_objects=$(rust_coverage_objects) || return 1
    rust_cov_collect_primary=
    set --
    while IFS= read -r rust_cov_collect_object; do
        if [ -z "$rust_cov_collect_primary" ]; then
            rust_cov_collect_primary=$rust_cov_collect_object
        else
            set -- "$@" "--object=$rust_cov_collect_object"
        fi
    done <<EOF
$rust_cov_collect_objects
EOF

    llvm-profdata merge \
        -sparse \
        "$RUST_COVERAGE_DIR"/*.profraw \
        -o "$RUST_COVERAGE_DIR/merged.profdata" || return 1
    llvm-cov export \
        "$rust_cov_collect_primary" \
        "$@" \
        --format=lcov \
        --instr-profile="$RUST_COVERAGE_DIR/merged.profdata" \
        --ignore-filename-regex='/\.cargo/(registry|git)/|/rustc/|/target/' \
        > "$rust_cov_collect_lcov" || return 1
    RUST_COVERAGE_LCOV=$rust_cov_collect_lcov
    export RUST_COVERAGE_LCOV
}

rust_coverage_upload() {
    rust_cov_upload_lcov=${1:?LCOV path is required}
    rust_cov_upload_flag=${2:?coverage flag is required}
    rust_cov_upload_repo_root=$(rust_coverage_repo_root) || return 1

    if [ "$(uname -m)" != "x86_64" ]; then
        echo "Rust coverage upload currently supports x86_64 CI agents" >&2
        return 1
    fi

    rust_cov_upload_codecov_dir=$(mktemp -d /tmp/codecov-bin.XXXXXX) \
        || return 1
    curl -fsSL \
        "https://github.com/codecov/codecov-cli/releases/download/${RUST_CODECOV_VERSION}/codecovcli_linux" \
        -o "$rust_cov_upload_codecov_dir/codecov" || return 1
    echo "$RUST_CODECOV_SHA256  $rust_cov_upload_codecov_dir/codecov" \
        | sha256sum -c - || return 1
    chmod +x "$rust_cov_upload_codecov_dir/codecov" || return 1

    rust_cov_upload_slug="vllm-project/vllm"
    if [ -n "${BUILDKITE_PULL_REQUEST:-}" ] \
        && [ "${BUILDKITE_PULL_REQUEST}" != "false" ] \
        && [ -n "${BUILDKITE_PULL_REQUEST_REPO:-}" ]; then
        rust_cov_upload_slug=$(echo "$BUILDKITE_PULL_REQUEST_REPO" \
            | sed -E 's#(git@|https?://)([^/:]+)[:/]([^/]+/[^/.]+)(\.git)?$#\3#')
        case "$rust_cov_upload_slug" in
            */*) ;;
            *) rust_cov_upload_slug="vllm-project/vllm" ;;
        esac
    fi

    rust_cov_upload_branch=${BUILDKITE_BRANCH:?BUILDKITE_BRANCH is required}
    if [ -z "${CODECOV_TOKEN:-}" ]; then
        # Codecov accepts tokenless public uploads on unprotected branch names.
        # A colon-separated prefix keeps feature-branch and fork uploads from
        # requiring a repository secret.
        if [ -n "${BUILDKITE_PULL_REQUEST:-}" ] \
            && [ "${BUILDKITE_PULL_REQUEST}" != "false" ]; then
            rust_cov_upload_branch="pr${BUILDKITE_PULL_REQUEST}:$rust_cov_upload_branch"
        else
            rust_cov_upload_branch="buildkite:$rust_cov_upload_branch"
        fi
    fi

    set --
    set -- "$@" upload-process
    set -- "$@" --file "$rust_cov_upload_lcov"
    # LCOV paths are mapped server-side by codecov.yml. Skip the CLI's local
    # source-line fix scanning, which is unrelated to path mapping.
    set -- "$@" --disable-search --disable-file-fixes
    set -- "$@" --fail-on-error --git-service github
    set -- "$@" --build "${BUILDKITE_BUILD_NUMBER:?BUILDKITE_BUILD_NUMBER is required}"
    set -- "$@" --branch "$rust_cov_upload_branch"
    set -- "$@" --sha "${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is required}"
    set -- "$@" --slug "$rust_cov_upload_slug"
    set -- "$@" --flag "$rust_cov_upload_flag"
    set -- "$@" --name "${rust_cov_upload_flag}-${BUILDKITE_JOB_ID:?BUILDKITE_JOB_ID is required}"
    set -- "$@" --dir "$rust_cov_upload_repo_root"
    set -- "$@" --network-root-folder "$rust_cov_upload_repo_root"
    if [ -n "${BUILDKITE_PULL_REQUEST:-}" ] \
        && [ "${BUILDKITE_PULL_REQUEST}" != "false" ]; then
        set -- "$@" --pr "$BUILDKITE_PULL_REQUEST"
    fi

    rust_cov_upload_log="$rust_cov_upload_codecov_dir/codecov.log"
    # E2E steps run from tests/, so execute from the repository root to resolve
    # codecov.yml and repository paths consistently.
    (
        cd "$rust_cov_upload_repo_root" || exit 1
        "$rust_cov_upload_codecov_dir/codecov" "$@"
    ) >"$rust_cov_upload_log" 2>&1
    rust_cov_upload_rc=$?
    cat "$rust_cov_upload_log"
    # v11.3.1 can log API failures while returning zero even with
    # --fail-on-error. Preserve the strict CI contract explicitly.
    if grep -aEq 'error.* -- ' "$rust_cov_upload_log"; then
        echo "Codecov CLI reported an upload error" >&2
        rust_cov_upload_rc=1
    fi
    rm -rf "$rust_cov_upload_codecov_dir"
    return "$rust_cov_upload_rc"
}

rust_coverage_finalize() {
    rust_cov_finalize_test_rc=$?
    trap - 0
    set +e

    rust_coverage_collect "$RUST_COVERAGE_FLAG"
    rust_cov_finalize_collect_rc=$?

    rust_cov_finalize_upload_rc=0
    if [ "$rust_cov_finalize_collect_rc" -eq 0 ]; then
        rust_coverage_upload "$RUST_COVERAGE_LCOV" "$RUST_COVERAGE_FLAG"
        rust_cov_finalize_upload_rc=$?
    fi

    find "$RUST_COVERAGE_DIR" -type f -name '*.profraw' -delete

    if [ "$rust_cov_finalize_test_rc" -ne 0 ]; then
        exit "$rust_cov_finalize_test_rc"
    fi
    if [ "$rust_cov_finalize_collect_rc" -ne 0 ]; then
        exit "$rust_cov_finalize_collect_rc"
    fi
    exit "$rust_cov_finalize_upload_rc"
}
