#!/bin/sh

RUST_CODECOV_VERSION="v11.3.1"
RUST_CODECOV_SHA256="ca1d64196d2d34771084afe76ea657d581bf628e31d993ff8e52ea09cc88a56d"

rust_coverage_repo_root() {
    if [ -f /vllm-workspace/.buildkite/scripts/normalize-rust-lcov.py ]; then
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
    LLVM_PROFILE_FILE="$RUST_COVERAGE_DIR/vllm-rust-%4m.profraw"
    export LLVM_PROFILE_FILE
    trap rust_coverage_finalize 0
}

rust_coverage_objects() {
    python3 - <<'PY'
from pathlib import Path
import sys

for entry in sys.path:
    package_dir = Path(entry or ".").resolve() / "vllm"
    binary = package_dir / "vllm-rs"
    parsers = sorted(package_dir.glob("_rust_tool_parser*.so"))
    if binary.is_file() and len(parsers) == 1:
        print(binary)
        print(parsers[0])
        break
else:
    raise RuntimeError("installed vLLM Rust coverage objects were not found")
PY
}

rust_coverage_collect() {
    rust_cov_collect_flag=${1:?coverage flag is required}
    rust_cov_collect_repo_root=$(rust_coverage_repo_root) || return 1
    rust_cov_collect_raw_lcov="$RUST_COVERAGE_DIR/$rust_cov_collect_flag.raw.lcov"
    rust_cov_collect_lcov="$RUST_COVERAGE_DIR/$rust_cov_collect_flag.lcov"

    command -v llvm-profdata >/dev/null || return 1
    command -v llvm-cov >/dev/null || return 1

    set --
    for rust_cov_collect_profile in "$RUST_COVERAGE_DIR"/*.profraw; do
        if [ -s "$rust_cov_collect_profile" ]; then
            set -- "$@" "$rust_cov_collect_profile"
        fi
    done
    if [ "$#" -eq 0 ]; then
        echo "Rust coverage profile is empty" >&2
        return 1
    fi

    rust_cov_collect_objects=$(rust_coverage_objects) || return 1
    rust_cov_collect_object_count=$(
        printf '%s\n' "$rust_cov_collect_objects" \
            | awk 'NF { count++ } END { print count + 0 }'
    )
    rust_cov_collect_vllm_rs=$(
        printf '%s\n' "$rust_cov_collect_objects" | sed -n '1p'
    )
    rust_cov_collect_parser=$(
        printf '%s\n' "$rust_cov_collect_objects" | sed -n '2p'
    )
    if [ "$rust_cov_collect_object_count" -ne 2 ] \
        || [ ! -f "$rust_cov_collect_vllm_rs" ] \
        || [ ! -f "$rust_cov_collect_parser" ]; then
        echo "Rust coverage objects were not found" >&2
        return 1
    fi

    llvm-profdata merge \
        -sparse \
        "$@" \
        -o "$RUST_COVERAGE_DIR/merged.profdata" || return 1
    llvm-cov export \
        "$rust_cov_collect_vllm_rs" \
        --object="$rust_cov_collect_parser" \
        --format=lcov \
        --instr-profile="$RUST_COVERAGE_DIR/merged.profdata" \
        --ignore-filename-regex='/\.cargo/(registry|git)/|/rustc/|/target/' \
        > "$rust_cov_collect_raw_lcov" || return 1
    python3 "$rust_cov_collect_repo_root/.buildkite/scripts/normalize-rust-lcov.py" \
        --input "$rust_cov_collect_raw_lcov" \
        --output "$rust_cov_collect_lcov" \
        --repo-root "$rust_cov_collect_repo_root" || return 1
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
        -o "$rust_cov_upload_codecov_dir/codecov" || {
            rm -rf "$rust_cov_upload_codecov_dir"
            return 1
        }
    echo "$RUST_CODECOV_SHA256  $rust_cov_upload_codecov_dir/codecov" \
        | sha256sum -c - || {
            rm -rf "$rust_cov_upload_codecov_dir"
            return 1
        }
    chmod +x "$rust_cov_upload_codecov_dir/codecov" || {
        rm -rf "$rust_cov_upload_codecov_dir"
        return 1
    }

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
    # The normalizer already validates every source path. The test image moves
    # vllm/ to src/vllm/, so generic repository file fixes cannot resolve all
    # tracked paths inside that image.
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
    # The CLI still resolves codecov.yml file-fix paths against its process
    # working directory even when --dir and --network-root-folder are set.
    # E2E steps run from tests/, so execute it from the repository root.
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
