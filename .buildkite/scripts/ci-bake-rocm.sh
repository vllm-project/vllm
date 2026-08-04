#!/bin/bash
# ci-bake-rocm.sh - Docker buildx bake wrapper for ROCm CI builds.
#
# The wrapper keeps three build concerns separate:
#   * ci_base builds are content-addressed by vllm.ci_base.content_hash.
#   * test image builds are commit-addressed by org.opencontainers.image.revision.
#   * ROCm install artifacts are uploaded only for targets that export wheels.
#
# Usage:
#   ci-bake-rocm.sh [TARGET]
#
# Set BAKE_PRINT_ONLY=1 to stop after docker buildx bake --print.

set -euo pipefail

DEFAULT_REPO_SLUG="vllm-project/vllm"
DEFAULT_CI_HCL_SOURCE="docker/ci-rocm.hcl"
DEFAULT_CI_BASE_CONTENT_FILES=".dockerignore requirements/common.txt requirements/rocm.txt requirements/test/rocm.txt docker/ci-rocm.hcl docker/docker-bake-rocm.hcl tools/install_torchcodec_rocm.sh tools/install_protoc.sh rust-toolchain.toml tests/vllm_test_utils"
DEFAULT_CI_BASE_DOCKERFILE="docker/Dockerfile.rocm"
DEFAULT_CI_BASE_DOCKERFILE_STAGES="base rust_toolchain_input_0 rust-toolchain-input rust-toolchain build_nixl build_rocshmem build_deepep mori_base ci_base"
DEFAULT_CI_BASE_METADATA_VERSION="3"
DEFAULT_ROCM_CACHE_NAMESPACE="v2"
DEFAULT_ROCM_CSRC_CONTENT_FILES=".dockerignore requirements/common.txt requirements/rocm.txt pyproject.toml setup.py CMakeLists.txt cmake csrc vllm/envs.py vllm/__init__.py tools/build_rust.py"
DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES="base fetch_vllm_0 fetch_vllm_1 fetch_vllm build_vllm_dependencies csrc-build"
DEFAULT_ROCM_RUST_CONTENT_FILES=".dockerignore requirements/build/rust.txt rust/Cargo.lock rust/Cargo.toml rust/proto rust/src rust-toolchain.toml tools/build_rust.py tools/install_protoc.sh build_rust.sh"
IMAGE_EXISTED_BEFORE_BUILD=0
ROCM_CACHE_WRITE_SUFFIX=""

TARGET=""
CI_HCL_SOURCE="${CI_HCL_SOURCE:-}"
CI_HCL_PATH=""
CI_BASE_LABEL_OVERRIDE_PATH=""
CSRC_CACHE_OVERRIDE_PATH=""
ROCM_ARG_OVERRIDE_PATH=""
SCRIPT_TMP_DIR=""
BAKE_CONFIG_FILE=""
BAKE_FILES=()
BAKE_TARGETS=()
DEPENDENCY_CACHE_TARGETS=()

cleanup() {
    if [[ -n "${SCRIPT_TMP_DIR}" && -d "${SCRIPT_TMP_DIR}" ]]; then
        rm -rf "${SCRIPT_TMP_DIR}"
    fi
}
trap cleanup EXIT

clean_docker_tag() {
    local input="$1"
    echo "${input}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-128
}

is_url_like() {
    local value="${1:-}"
    [[ "${value}" =~ ^[a-zA-Z][a-zA-Z0-9+.-]*:// || "${value}" == git@*:* ]]
}

is_full_git_sha() {
    local value="${1:-}"
    [[ "${value}" =~ ^[0-9a-fA-F]{40}$ ]]
}

select_cache_branch_name() {
    local candidate=""
    local var=""

    for var in \
        ROCM_CACHE_BRANCH_NAME \
        BUILDKITE_PULL_REQUEST_HEAD_BRANCH \
        BUILDKITE_HEAD_BRANCH \
        BUILDKITE_BRANCH \
        VLLM_BRANCH; do
        candidate="${!var:-}"
        [[ -n "${candidate}" ]] || continue
        is_url_like "${candidate}" && continue
        is_full_git_sha "${candidate}" && continue
        printf '%s\n' "${candidate}"
        return 0
    done
}

cache_scope_suffix() {
    local arch_hash=""
    arch_hash=$(printf '%s' "${PYTORCH_ROCM_ARCH:-default}" | sha256sum | cut -c1-12)
    printf 'arch-%s\n' "${arch_hash}"
}

compose_cache_branch_tag() {
    local repo_slug="$1"
    local branch="$2"
    local suffix=""
    local prefix=""
    local max_prefix_len=0
    local max_tag_len=60

    suffix="$(cache_scope_suffix)"
    prefix="$(clean_docker_tag "${repo_slug}")-$(clean_docker_tag "${branch}")"
    max_prefix_len=$((max_tag_len - ${#suffix} - 1))
    if (( max_prefix_len < 1 )); then
        max_prefix_len=1
    fi
    printf '%s-%s\n' "${prefix:0:${max_prefix_len}}" "${suffix}"
}

parse_repo_slug() {
    local repo_url="${1:-}"
    local repo_slug=""

    if [[ -z "${repo_url}" ]]; then
        printf '%s\n' "${DEFAULT_REPO_SLUG}"
        return 0
    fi

    repo_slug=$(echo "${repo_url}" | sed -E 's#(git@|https?://)([^/:]+)[:/]([^/]+/[^/.]+)(\.git)?$#\3#')
    if [[ "${repo_slug}" != */* ]]; then
        repo_slug="${DEFAULT_REPO_SLUG}"
    fi
    printf '%s\n' "${repo_slug}"
}

normalize_repo_slug() {
    local repo_slug="${1:-}"

    repo_slug="${repo_slug%/}"
    repo_slug="${repo_slug%.git}"
    repo_slug="${repo_slug#https://github.com/}"
    repo_slug="${repo_slug#http://github.com/}"
    repo_slug="${repo_slug#ssh://git@github.com/}"
    repo_slug="${repo_slug#git@github.com:}"
    repo_slug="${repo_slug#github.com/}"
    printf '%s\n' "${repo_slug}"
}

is_trusted_ci_cache_writer() {
    local actual_repo=""
    local trusted_repo=""

    [[ "${BUILDKITE:-false}" == "true" ]] || return 1
    [[ "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]] || return 1
    [[ "${BUILDKITE_BRANCH:-}" == "${CI_BASE_STABLE_BRANCH:-main}" ]] || return 1
    actual_repo=$(normalize_repo_slug "${BUILDKITE_REPO:-}")
    trusted_repo=$(normalize_repo_slug \
        "${CI_BASE_STABLE_REPO_SLUG:-${DEFAULT_REPO_SLUG}}")
    [[ -n "${actual_repo}" && "${actual_repo}" == "${trusted_repo}" ]]
}

ci_cache_write_scope() {
    local identity=""
    local pull_request="${BUILDKITE_PULL_REQUEST:-false}"
    local branch="${BUILDKITE_PULL_REQUEST_HEAD_BRANCH:-${BUILDKITE_BRANCH:-local}}"
    local source_repo="${BUILDKITE_PULL_REQUEST_REPO:-${BUILDKITE_REPO:-local}}"

    if is_trusted_ci_cache_writer; then
        return 0
    fi

    identity=$(printf '%s\n' "${source_repo}" | sha256sum | cut -c1-12)
    if [[ "${pull_request}" != "false" && -n "${pull_request}" ]]; then
        printf 'pr-%s-%s\n' \
            "$(clean_docker_tag "${pull_request}" | cut -c1-12)" "${identity}"
        return 0
    fi

    printf 'preview-%s-%s\n' \
        "$(clean_docker_tag "${branch}" | cut -c1-12)" "${identity}"
}

configure_cache_write_scope() {
    local scope=""

    if [[ ! "${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}" \
        =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,12}$ ]]; then
        echo "Invalid ROCm cache namespace: ${ROCM_CACHE_NAMESPACE:-}" >&2
        return 1
    fi

    scope=$(ci_cache_write_scope)
    if [[ -n "${scope}" ]]; then
        ROCM_CACHE_WRITE_SUFFIX="-$(clean_docker_tag "${scope}")"
        echo "Untrusted cache writes are scoped with: ${ROCM_CACHE_WRITE_SUFFIX#-}"
    else
        ROCM_CACHE_WRITE_SUFFIX=""
        echo "Trusted main build: publishing canonical shared cache refs"
    fi
    export ROCM_CACHE_WRITE_SUFFIX
}

get_buildkite_repo_slug() {
    parse_repo_slug "${BUILDKITE_PULL_REQUEST_REPO:-${BUILDKITE_REPO:-}}"
}

get_buildkite_target_repo_slug() {
    parse_repo_slug "${BUILDKITE_REPO:-}"
}

get_buildkite_target_repo_url() {
    local repo_url="${BUILDKITE_REPO:-}"

    if [[ -n "${repo_url}" ]] && is_url_like "${repo_url}"; then
        printf '%s\n' "${repo_url}"
        return 0
    fi

    printf 'https://github.com/%s.git\n' "${DEFAULT_REPO_SLUG}"
}

git_fetch_for_cache() {
    local timeout_secs="${ROCM_CACHE_GIT_FETCH_TIMEOUT:-60}"

    if command -v timeout >/dev/null 2>&1; then
        timeout "${timeout_secs}s" git fetch "$@" 2>/dev/null
    else
        git fetch "$@" 2>/dev/null
    fi
}

hash_string_short() {
    printf '%s' "$1" | sha256sum | cut -c1-16
}

list_content_files() {
    git ls-files -z -- "$1" | LC_ALL=C sort -z
}

hash_content_file() {
    local file="$1"
    local mode="644"
    local raw_mode=""

    if [[ -L "${file}" ]]; then
        printf 'symlink:%s\ntarget:%s\n' "${file}" "$(readlink "${file}")"
        return
    fi
    raw_mode=$(stat -c '%a' "${file}")
    if (((8#${raw_mode} & 0111) != 0)); then
        mode="755"
    fi
    printf 'file:%s\nmode:%s\n' "${file}" "${mode}"
    sha256sum "${file}"
}

hash_content_directory() {
    if ! list_content_files "$1" | while IFS= read -r -d '' file; do
        hash_content_file "${file}" || exit $?
    done; then
        echo "Error: failed to hash content under $1" >&2
        return 1
    fi
}

compute_content_hash() {
    local path=""

    for path in "$@"; do
        if [[ -L "${path}" || -f "${path}" ]]; then
            hash_content_file "${path}" || return $?
        elif [[ -d "${path}" ]]; then
            hash_content_directory "${path}" || return $?
        else
            printf 'missing:%s\n' "${path}"
        fi
    done | sha256sum | cut -d' ' -f1
}

normalize_ci_worktree_modes() {
    local entry=""
    local metadata=""
    local mode=""
    local stage=""
    local path=""
    local index_modes_file=""
    local -a regular_files=()
    local -a executable_files=()

    [[ "${BUILDKITE:-false}" == "true" ]] || return 0
    [[ "${REMOTE_VLLM:-0}" == "0" ]] || return 0
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "A Git worktree is required to normalize the CI Docker context" >&2
        return 1
    fi
    if [[ -z "${SCRIPT_TMP_DIR}" || ! -d "${SCRIPT_TMP_DIR}" ]]; then
        echo "A script temporary directory is required to normalize the CI Docker context" >&2
        return 1
    fi
    index_modes_file="${SCRIPT_TMP_DIR}/git-index-modes"
    if ! git ls-files --stage -z > "${index_modes_file}"; then
        echo "Failed to read Git index modes for the CI Docker context" >&2
        return 1
    fi

    # Some CI workspaces expose every checked-out file as executable. Restore
    # the Git index modes before hashing or sending the local Docker context so
    # equivalent commits produce equivalent cache keys and image layers.
    while IFS= read -r -d '' entry; do
        if [[ "${entry}" != *$'\t'* ]]; then
            echo "Malformed git index entry while normalizing Docker context" >&2
            return 1
        fi
        metadata="${entry%%$'\t'*}"
        mode="${metadata%% *}"
        stage="${metadata##* }"
        path="${entry#*$'\t'}"
        if [[ "${stage}" != "0" ]]; then
            echo "Unmerged git index entry cannot be used as Docker context: ${path}" >&2
            return 1
        fi

        case "${mode}" in
            100644|100755)
                if [[ ! -f "${path}" || -L "${path}" ]]; then
                    echo "Tracked Docker context file is missing or invalid: ${path}" >&2
                    return 1
                fi
                if [[ "${mode}" == "100755" ]]; then
                    executable_files+=("${path}")
                else
                    regular_files+=("${path}")
                fi
                ;;
            120000|160000)
                ;;
            *)
                echo "Unsupported git index mode ${mode} for ${path}" >&2
                return 1
                ;;
        esac
    done < "${index_modes_file}"

    if [[ ${#regular_files[@]} -gt 0 ]]; then
        chmod 0644 -- "${regular_files[@]}"
    fi
    if [[ ${#executable_files[@]} -gt 0 ]]; then
        chmod 0755 -- "${executable_files[@]}"
    fi
    echo "Normalized $((${#regular_files[@]} + ${#executable_files[@]})) Git-tracked file modes for the Docker context"
}

compose_dependency_cache_key() {
    local prefix="$1"
    local material="$2"
    local cleaned_prefix=""

    # Leave room for the cache family, schema, and untrusted write scope in a
    # Docker tag (128 chars maximum). The terminal hash preserves uniqueness.
    cleaned_prefix=$(clean_docker_tag "${prefix}" | cut -c1-46)
    printf '%s-%s\n' "${cleaned_prefix}" "$(hash_string_short "${material}")"
}

hash_dockerfile_stages() {
    local dockerfile="$1"
    local stages="$2"

    awk -v wanted_stages="${stages}" '
        BEGIN {
            split(wanted_stages, stage_list, /[[:space:]]+/)
            for (idx in stage_list) {
                if (stage_list[idx] != "") {
                    wanted[stage_list[idx]] = 1
                }
            }
            emit = 1
        }
        $1 == "FROM" {
            stage = ""
            for (idx = 1; idx <= NF; idx++) {
                if (tolower($idx) == "as" && idx < NF) {
                    stage = $(idx + 1)
                }
            }
            emit = (stage in wanted)
        }
        emit {
            print
        }
    ' "${dockerfile}"
}

discover_dockerfile_stage_args() {
    local dockerfile="$1"
    local stages="$2"

    [[ -f "${dockerfile}" ]] || return 0

    awk -v wanted_stages="${stages}" '
        function add_arg(name) {
            if (name != "" && !(name in seen)) {
                seen[name] = 1
                args[++arg_count] = name
            }
        }
        BEGIN {
            split(wanted_stages, stage_list, /[[:space:]]+/)
            for (idx in stage_list) {
                if (stage_list[idx] != "") {
                    wanted[stage_list[idx]] = 1
                }
            }
            emit = 1
        }
        {
            line = $0
            if ($1 == "FROM") {
                stage = ""
                for (idx = 1; idx <= NF; idx++) {
                    if (tolower($idx) == "as" && idx < NF) {
                        stage = $(idx + 1)
                    }
                }
                emit = (stage in wanted)
            }
            if (emit) {
                lines[++line_count] = line
            }
        }
        END {
            for (idx = 1; idx <= line_count; idx++) {
                line = lines[idx]
                arg_name = line
                sub(/^[[:space:]]*ARG[[:space:]]+/, "", arg_name)
                if (arg_name != line) {
                    sub(/[=[:space:]].*/, "", arg_name)
                    if (arg_name ~ /^[A-Za-z_][A-Za-z0-9_]*$/) {
                        add_arg(arg_name)
                    }
                }
            }

            for (idx = 1; idx <= line_count; idx++) {
                line = lines[idx]
                for (arg_idx = 1; arg_idx <= arg_count; arg_idx++) {
                    name = args[arg_idx]
                    if (line ~ "\\$\\{" name "([}:][^}]*)?\\}" \
                        || line ~ "\\$" name "([^A-Za-z0-9_]|$)") {
                        used[name] = 1
                    }
                }
            }

            for (arg_idx = 1; arg_idx <= arg_count; arg_idx++) {
                name = args[arg_idx]
                if (used[name]) {
                    print name
                }
            }
        }
    ' "${dockerfile}"
}

get_content_arg_names() {
    local dockerfile="$1"
    local stages="$2"
    local explicit_args="${3:-}"

    if [[ -n "${explicit_args}" ]]; then
        tr ' ' '\n' <<< "${explicit_args}"
    else
        discover_dockerfile_stage_args "${dockerfile}" "${stages}"
    fi | awk 'NF && !seen[$0]++'
}

compute_ci_base_content_hash_once() {
    local -a content_paths=()
    local -a content_args=()
    local content_files_hash=""
    local dockerfile="${CI_BASE_DOCKERFILE:-}"
    local stages="${CI_BASE_DOCKERFILE_STAGES:-}"

    read -r -a content_paths <<< "${CI_BASE_CONTENT_FILES}"
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile}" "${stages}" "${CI_BASE_CONTENT_ARGS:-}"
    )
    if ! content_files_hash=$(compute_content_hash "${content_paths[@]}"); then
        echo "Failed to hash ci_base content files" >&2
        return 1
    fi

    {
        printf 'content-files-hash:%s\n' "${content_files_hash}"
        if [[ -n "${dockerfile}" ]]; then
            printf 'dockerfile:%s\n' "${dockerfile}"
            printf 'resolved-build-args:\n'
            hash_dockerfile_arg_values "${dockerfile}" "${content_args[@]}" \
                || return 1
            if [[ -n "${stages}" ]]; then
                printf 'dockerfile-stages:%s\n' "${stages}"
                if [[ -f "${dockerfile}" ]]; then
                    hash_dockerfile_stages "${dockerfile}" "${stages}"
                else
                    printf 'missing:%s\n' "${dockerfile}"
                fi
            fi
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_ci_base_content_hash() {
    local attempts="${CI_BASE_HASH_ATTEMPTS:-1}"
    local delay_secs="${CI_BASE_HASH_RETRY_DELAY:-5}"
    local attempt=0
    local hash=""
    local failed=0
    local -a hashes=()

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid CI_BASE_HASH_ATTEMPTS: ${attempts}" >&2
        return 1
    fi
    if [[ ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid CI_BASE_HASH_RETRY_DELAY: ${delay_secs}" >&2
        return 1
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if ! hash=$(compute_ci_base_content_hash_once); then
            echo "ci_base content hash calculation ${attempt}/${attempts} failed" >&2
            failed=1
        else
            hashes+=("${hash}")
            echo "ci_base content hash calculation ${attempt}/${attempts}: ${hash}" >&2
        fi

        if ((attempt < attempts)); then
            sleep "${delay_secs}"
        fi
    done

    if ((failed)) || ((${#hashes[@]} != attempts)); then
        echo "Could not calculate a reliable ci_base content hash" >&2
        return 1
    fi

    for hash in "${hashes[@]:1}"; do
        if [[ "${hash}" != "${hashes[0]}" ]]; then
            echo "ci_base content hash changed between calculations" >&2
            printf '  observed: %s\n' "${hashes[@]}" >&2
            return 1
        fi
    done

    printf '%s\n' "${hashes[0]}"
}

extract_dockerfile_arg_default() {
    local dockerfile="$1"
    local arg_name="$2"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${dockerfile}" | head -1
}

resolve_image_digest() {
    local image_ref="$1"
    local attempts="${ROCM_IMAGE_DIGEST_ATTEMPTS:-4}"
    local delay_secs="${ROCM_IMAGE_DIGEST_RETRY_DELAY:-2}"
    local attempt=0
    local digest=""
    local output=""
    local status=0

    if [[ "${image_ref}" =~ @(sha256:[0-9a-f]{64})$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return
    fi
    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid image digest retry configuration" >&2
        return 1
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) || status=$?
        digest=$(awk '$1 == "Digest:" { print $2; exit }' <<< "${output}")
        if ((status == 0)) && [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            printf '%s\n' "${digest}"
            return
        fi
        if ((attempt < attempts)); then
            printf \
                'Image digest lookup failed for %s (%d/%d, status %d); retrying\n' \
                "${image_ref}" "${attempt}" "${attempts}" "${status}" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Failed to resolve digest for %s (status %d)\n%s\n' \
        "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 1
}

resolve_dockerfile_arg_value() {
    local dockerfile="$1"
    local arg_name="$2"
    local env_name="${arg_name}"
    local value=""

    case "${arg_name}" in
        ARG_PYTORCH_ROCM_ARCH)
            env_name="PYTORCH_ROCM_ARCH"
            ;;
        max_jobs)
            env_name="CI_MAX_JOBS"
            ;;
    esac

    value="${!env_name:-}"
    if [[ -z "${value}" && "${env_name}" != "${arg_name}" ]]; then
        value="${!arg_name:-}"
    fi
    if [[ -z "${value}" && -f "${dockerfile}" ]]; then
        value=$(extract_dockerfile_arg_default "${dockerfile}" "${arg_name}")
    fi

    printf '%s\n' "${value}"
}

hash_dockerfile_arg_values() {
    local dockerfile="$1"
    local arg_name=""
    local arg_value=""
    local digest=""
    shift || true

    for arg_name in "$@"; do
        [[ -n "${arg_name}" ]] || continue
        arg_value=$(resolve_dockerfile_arg_value "${dockerfile}" "${arg_name}")
        if [[ "${arg_name}" == "BASE_IMAGE" && -n "${arg_value}" ]]; then
            if ! digest=$(resolve_image_digest "${arg_value}"); then
                echo "Failed to resolve digest for BASE_IMAGE=${arg_value}" >&2
                return 1
            fi
            printf 'arg:%s.digest=%s\n' "${arg_name}" "${digest}"
        else
            printf 'arg:%s=%s\n' "${arg_name}" "${arg_value:-<empty>}"
        fi
    done
}

pin_base_image() {
    local dockerfile="${CI_BASE_DOCKERFILE:-${DEFAULT_CI_BASE_DOCKERFILE}}"
    local base_image=""
    local digest=""

    base_image=$(resolve_dockerfile_arg_value "${dockerfile}" "BASE_IMAGE")
    [[ -n "${base_image}" ]] || return 0
    if ! digest=$(resolve_image_digest "${base_image}"); then
        echo "Error: could not resolve base image digest for ${base_image}" >&2
        echo "Refusing to compute content-addressed cache keys from a mutable tag." >&2
        return 1
    fi
    BASE_IMAGE="${base_image%@*}@${digest}"
    export BASE_IMAGE
    echo "Pinned base image for this build: ${BASE_IMAGE}"
}

publish_ci_base_parent_ref() {
    [[ -n "${BASE_IMAGE:-}" ]] || {
        echo "Cannot publish ci_base parent handoff without a pinned BASE_IMAGE" >&2
        return 1
    }

    if command -v buildkite-agent >/dev/null 2>&1; then
        if ! buildkite-agent meta-data set "rocm-ci-base-parent-image" "${BASE_IMAGE}"; then
            echo "Could not publish required ci_base parent-image metadata" >&2
            return 1
        fi
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish ci_base parent-image metadata" >&2
        return 1
    fi
    echo "Published pinned ci_base parent image: ${BASE_IMAGE}"
}

is_ci_base_target() {
    [[ "${TARGET}" == *"ci-base-rocm"* ]]
}

is_commit_image_target() {
    [[ -n "${IMAGE_TAG:-}" && -n "${BUILDKITE_COMMIT:-}" ]] || return 1
    is_ci_base_target && return 1
    return 0
}

image_tag_is_commit_scoped() {
    [[ -n "${IMAGE_TAG:-}" && -n "${BUILDKITE_COMMIT:-}" ]] || return 1
    [[ "${IMAGE_TAG}" == *"${BUILDKITE_COMMIT}"* ]]
}

should_upload_wheel_artifacts() {
    [[ "${UPLOAD_ROCM_WHEEL_ARTIFACTS:-0}" == "1" ]] && return 0
    [[ "${TARGET}" == *"with-wheel"* \
        || "${TARGET}" == *"export-wheel"* \
        || "${TARGET}" == *"artifact"* ]]
}

set_buildkite_metadata() {
    local key="$1"
    local value="$2"

    [[ -n "${value}" ]] || return 0
    if command -v buildkite-agent >/dev/null 2>&1; then
        if ! buildkite-agent meta-data set "${key}" "${value}"; then
            echo "Could not publish required Buildkite metadata: ${key}" >&2
            return 1
        fi
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish ${key}" >&2
        return 1
    fi
}

get_remote_image_label() {
    local image_ref="$1"
    local label_key="$2"

    docker buildx imagetools inspect "${image_ref}" --raw 2>/dev/null \
        | python3 -c '
import json
import subprocess
import sys
import urllib.parse
import urllib.request

image_ref = sys.argv[1]
label_key = sys.argv[2]


def docker_hub_repo(image_name):
    image_name = image_name.split("@", 1)[0]
    last_component = image_name.rsplit("/", 1)[-1]
    if ":" in last_component:
        image_name = image_name.rsplit(":", 1)[0]

    parts = image_name.split("/")
    if len(parts) > 1 and (
        "." in parts[0] or ":" in parts[0] or parts[0] == "localhost"
    ):
        registry = parts[0]
        if registry not in {
            "docker.io",
            "index.docker.io",
            "registry-1.docker.io",
        }:
            return None
        image_name = "/".join(parts[1:])
    elif len(parts) == 1:
        image_name = f"library/{image_name}"

    return image_name


try:
    data = json.load(sys.stdin)
    if data.get("manifests"):
        manifest = next(
            (
                entry
                for entry in data["manifests"]
                if entry.get("platform", {}).get("os") != "unknown"
                and entry.get("platform", {}).get("architecture") != "unknown"
            ),
            data["manifests"][0],
        )
        digest = manifest["digest"]
        result = subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                image_ref.split("@", 1)[0] + "@" + digest,
                "--raw",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError("digest inspect failed")
        data = json.loads(result.stdout)

    annotations = data.get("annotations", {})
    if label_key in annotations:
        print(annotations[label_key])
        raise SystemExit(0)

    config_digest = data.get("config", {}).get("digest")
    if not config_digest:
        print("")
        raise SystemExit(0)

    image_name = docker_hub_repo(image_ref)
    if not image_name:
        print("")
        raise SystemExit(0)

    token_url = (
        "https://auth.docker.io/token?"
        + urllib.parse.urlencode(
            {
                "service": "registry.docker.io",
                "scope": f"repository:{image_name}:pull",
            }
        )
    )
    with urllib.request.urlopen(token_url, timeout=30) as response:
        token = json.load(response)["token"]

    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{image_name}/blobs/{config_digest}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        config_blob = json.load(response)

    labels = config_blob.get("config", {}).get("Labels", {})
    print(labels.get(label_key, ""))
except Exception:
    print("")
' "${image_ref}" "${label_key}" 2>/dev/null || echo ""
}

get_remote_image_label_with_retry() {
    local image_ref="$1"
    local label_key="$2"
    local attempts="${3:-6}"
    local delay_secs="${4:-5}"
    local label_value=""
    local attempt

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        label_value=$(get_remote_image_label "${image_ref}" "${label_key}")
        if [[ -n "${label_value}" ]]; then
            printf '%s\n' "${label_value}"
            return 0
        fi
        if [[ ${attempt} -lt ${attempts} ]]; then
            sleep "${delay_secs}"
        fi
    done

    return 0
}

remote_ci_base_identity_is_current_with_retry() {
    local image_ref="$1"
    local attempts="${CI_BASE_LABEL_ATTEMPTS:-4}"
    local delay_secs="${CI_BASE_LABEL_RETRY_DELAY:-2}"
    local identity=""
    local remote_hash=""
    local remote_version=""
    local content_files_hash=""
    local base_digest=""
    local inspect_status=0
    local saw_complete_identity=0
    local attempt=0
    local expected_base_digest=""
    local expected_version="${CI_BASE_METADATA_VERSION:-${DEFAULT_CI_BASE_METADATA_VERSION}}"
    local format='{{ index .Image.Config.Labels "vllm.ci_base.content_hash" }}|{{ index .Image.Config.Labels "vllm.ci_base.metadata_version" }}|{{ index .Image.Config.Labels "vllm.ci_base.content_files_hash" }}|{{ index .Image.Config.Labels "vllm.rocm.base_image_digest" }}'

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ci_base label retry configuration" >&2
        return 2
    fi
    if [[ ! "${BASE_IMAGE:-}" =~ @sha256:[0-9a-f]{64}$ ]]; then
        echo "Cannot validate ci_base identity without a pinned BASE_IMAGE" >&2
        return 2
    fi
    expected_base_digest="${BASE_IMAGE##*@}"

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        inspect_status=0
        identity=$(docker buildx imagetools inspect "${image_ref}" \
            --format "${format}" 2>/dev/null) || inspect_status=$?
        if ((inspect_status != 0)); then
            ((attempt == attempts)) || sleep "${delay_secs}"
            continue
        fi
        remote_hash=""
        remote_version=""
        content_files_hash=""
        base_digest=""
        IFS='|' read -r \
            remote_hash remote_version content_files_hash base_digest <<< "${identity}"
        if [[ "${remote_hash}" =~ ^[0-9a-f]{64}$ \
            && "${remote_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ \
            && "${content_files_hash}" =~ ^[0-9a-f]{64}$ \
            && "${base_digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            saw_complete_identity=1
            if [[ "${remote_hash}" == "${CI_BASE_CONTENT_HASH:-}" \
                && "${remote_version}" == "${expected_version}" \
                && "${base_digest}" == "${expected_base_digest}" ]]; then
                return 0
            fi
        fi
        ((attempt == attempts)) || sleep "${delay_secs}"
    done

    if ((saw_complete_identity == 1)); then
        echo "ci_base identity does not match: ${image_ref}" >&2
        echo "  expected hash/version: ${CI_BASE_CONTENT_HASH:-<missing>}/${expected_version}" >&2
        echo "  observed hash/version: ${remote_hash:-<missing>}/${remote_version:-<missing>}" >&2
        return 1
    fi
    echo "Failed to inspect ci_base identity after ${attempts} attempts: ${image_ref}" >&2
    return 2
}

registry_ref_exists_with_retry() {
    local inspect_kind="$1"
    local image_ref="$2"
    local attempts="${ROCM_REGISTRY_PROBE_ATTEMPTS:-2}"
    local delay_secs="${ROCM_REGISTRY_PROBE_RETRY_DELAY:-1}"
    local output=""
    local status=0
    local attempt=0

    if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_secs}" =~ ^[0-9]+$ ]]; then
        echo "Invalid registry probe retry configuration" >&2
        return 2
    fi

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        status=0
        if [[ "${inspect_kind}" == "manifest" ]]; then
            output=$(docker manifest inspect "${image_ref}" 2>&1) || status=$?
        else
            output=$(docker buildx imagetools inspect "${image_ref}" 2>&1) \
                || status=$?
        fi
        if ((status == 0)); then
            return 0
        fi
        # A definitive registry miss is normal for a new content key. Retry
        # transport/rate-limit failures, but do not add latency to a real 404.
        if grep -Eiq \
            'manifest unknown|name unknown|no such manifest|(^|[^0-9])404([^0-9]|$)|(^|[[:space:]])[^[:space:]]+/[^[:space:]]+:[^[:space:]]+:[[:space:]]+not found([[:space:]]|$)' \
            <<< "${output}"; then
            return 1
        fi
        if ((attempt < attempts)); then
            echo "Registry probe failed for ${image_ref} (${attempt}/${attempts}); retrying" >&2
            sleep "${delay_secs}"
        fi
    done

    printf 'Registry probe failed after %s attempts: %s (status %s)\n%s\n' \
        "${attempts}" "${image_ref}" "${status}" "${output:-<no output>}" >&2
    return 2
}

remote_image_exists() {
    registry_ref_exists_with_retry manifest "$1"
}

use_existing_builder() {
    echo "Using existing builder: ${BUILDER_NAME}"
    docker buildx use "${BUILDER_NAME}"
    docker buildx inspect --bootstrap
}

buildx_driver() {
    local builder="${1:-}"

    if [[ -n "${builder}" ]]; then
        docker buildx inspect "${builder}" 2>/dev/null
    else
        docker buildx inspect 2>/dev/null
    fi | awk -F': *' '$1 == "Driver" { print $2; exit }'
}

builder_supports_registry_cache() {
    local driver="$1"

    [[ -n "${driver}" && "${driver}" != "docker" ]]
}

create_and_bootstrap_builder() {
    local driver="$1"
    local endpoint="${2:-}"

    echo "Creating builder '${BUILDER_NAME}' with ${driver} driver"
    if [[ -n "${endpoint}" ]]; then
        docker buildx create \
            --name "${BUILDER_NAME}" \
            --driver "${driver}" \
            --use \
            "${endpoint}"
    else
        docker buildx create --name "${BUILDER_NAME}" --driver "${driver}" --use
    fi
    docker buildx inspect --bootstrap
}

init_config() {
    TARGET="${1:-test-ci}"
    BAKE_TARGETS=("${TARGET}")
    DEPENDENCY_CACHE_TARGETS=()
    CI_HCL_SOURCE="${CI_HCL_SOURCE:-${CI_HCL_FILE:-${DEFAULT_CI_HCL_SOURCE}}}"
    VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake-rocm.hcl}"
    BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
    BUILDKIT_SOCKET="${BUILDKIT_SOCKET:-/run/buildkit/buildkitd.sock}"
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
    CI_BASE_CONTENT_FILES="${CI_BASE_CONTENT_FILES:-${DEFAULT_CI_BASE_CONTENT_FILES}}"
    CI_BASE_DOCKERFILE="${CI_BASE_DOCKERFILE:-${DEFAULT_CI_BASE_DOCKERFILE}}"
    CI_BASE_DOCKERFILE_STAGES="${CI_BASE_DOCKERFILE_STAGES:-${DEFAULT_CI_BASE_DOCKERFILE_STAGES}}"
    CI_BASE_METADATA_VERSION="${CI_BASE_METADATA_VERSION:-${DEFAULT_CI_BASE_METADATA_VERSION}}"
    ROCM_CACHE_NAMESPACE="${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}"
    CI_BASE_IMAGE_TAG="${CI_BASE_IMAGE_TAG:-rocm/vllm-dev:ci_base}"
    export PYTORCH_ROCM_ARCH
    export ROCM_CACHE_NAMESPACE

    SCRIPT_TMP_DIR=$(mktemp -d -t ci-bake-rocm.XXXXXX)
    CI_HCL_PATH="${SCRIPT_TMP_DIR}/ci.hcl"
    CI_BASE_LABEL_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/ci-base-label-override.hcl"
    CSRC_CACHE_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/rocm-csrc-cache-override.hcl"
    ROCM_ARG_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/rocm-arg-override.hcl"
    BAKE_CONFIG_FILE="bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
}

print_header() {
    echo "--- :docker: Setting up Docker buildx bake"
    echo "Target: ${TARGET}"
    echo "CI HCL source: ${CI_HCL_SOURCE}"
    echo "vLLM bake file: ${VLLM_BAKE_FILE}"
    if is_ci_base_target; then
        echo "Build mode: ci_base"
    elif is_commit_image_target; then
        echo "Build mode: commit image"
    else
        echo "Build mode: generic"
    fi
    if [[ "${USE_SCCACHE:-0}" == "1" ]]; then
        echo "Compiler cache: sccache enabled"
    fi
}

validate_inputs() {
    if [[ ! -f "${VLLM_BAKE_FILE}" ]]; then
        echo "Error: vLLM bake file not found at ${VLLM_BAKE_FILE}"
        echo "Make sure you're running from the vLLM repository root"
        exit 1
    fi

    if [[ -n "${CI_HCL_SOURCE:-}" ]] && is_url_like "${CI_HCL_SOURCE}"; then
        echo "Error: remote CI HCL sources are not supported: ${CI_HCL_SOURCE}"
        echo "Use the vLLM-owned docker/ci-rocm.hcl or set CI_HCL_SOURCE to a local file."
        exit 1
    fi

    if [[ -n "${CI_HCL_SOURCE:-}" && ! -f "${CI_HCL_SOURCE}" ]]; then
        echo "Error: CI HCL file not found at ${CI_HCL_SOURCE}"
        echo "Set CI_HCL_SOURCE to a local file if you need an override."
        exit 1
    fi
}

load_ci_hcl() {
    echo "--- :page_facing_up: Loading ci.hcl"
    cp "${CI_HCL_SOURCE}" "${CI_HCL_PATH}"
    echo "Copied ${CI_HCL_SOURCE} to ${CI_HCL_PATH}"
}

init_bake_files() {
    BAKE_FILES=(-f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}")
}

compute_ci_base_hash_if_needed() {
    if [[ -z "${CI_BASE_CONTENT_FILES:-}" ]]; then
        return 0
    fi
    pin_base_image
    if ! is_ci_base_target; then
        return 0
    fi
    publish_ci_base_parent_ref
    if [[ "${REMOTE_VLLM:-0}" != "0" ]]; then
        echo "Error: content-addressed ci_base builds require REMOTE_VLLM=0" >&2
        return 1
    fi

    CI_BASE_CONTENT_HASH=$(compute_ci_base_content_hash)
    export CI_BASE_CONTENT_HASH
    echo "ci_base content hash: ${CI_BASE_CONTENT_HASH:0:16}..."
}

wants_stable_ci_base_tag() {
    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        return 1
    fi

    if [[ "${CI_BASE_PUSH_STABLE_TAG:-}" == "1" ]]; then
        return 0
    fi
    if [[ "${CI_BASE_PUSH_STABLE_TAG:-}" == "0" ]]; then
        return 1
    fi

    [[ "${NIGHTLY:-0}" == "1" && "${BUILDKITE_BRANCH:-}" == "${CI_BASE_STABLE_BRANCH:-main}" ]]
}

trusted_ci_base_tip_matches_build() {
    local branch="${CI_BASE_STABLE_BRANCH:-main}"
    local build_commit="${BUILDKITE_COMMIT:-}"
    local remote_tip=""

    is_trusted_ci_cache_writer || return 1
    if [[ ! "${build_commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Skipping ci_base stable tag: Buildkite commit is missing or invalid" >&2
        return 1
    fi
    remote_tip=$(git ls-remote --exit-code "${BUILDKITE_REPO}" \
        "refs/heads/${branch}" 2>/dev/null | awk 'NR == 1 { print $1 }')
    if [[ ! "${remote_tip}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Skipping ci_base stable tag: could not resolve remote ${branch} tip" >&2
        return 1
    fi
    if [[ "${remote_tip,,}" != "${build_commit,,}" ]]; then
        echo "Skipping ci_base stable tag: ${branch} advanced from ${build_commit} to ${remote_tip}" >&2
        return 1
    fi
}

should_push_stable_ci_base_tag() {
    wants_stable_ci_base_tag \
        && is_trusted_ci_cache_writer \
        && trusted_ci_base_tip_matches_build
}

ci_base_tag_with_suffix() {
    local base_tag="$1"
    local suffix="$2"

    printf '%s-%s\n' "${base_tag}" "$(clean_docker_tag "${suffix}")"
}

configure_ci_base_image_refs() {
    local stable_tag="${CI_BASE_IMAGE_TAG:-rocm/vllm-dev:ci_base}"
    local metadata_version="${CI_BASE_METADATA_VERSION:-${DEFAULT_CI_BASE_METADATA_VERSION}}"
    local scope="${ROCM_CACHE_WRITE_SUFFIX#-}"
    local trusted_content_tag=""
    local content_tag=""
    local commit_tag=""
    local build_tag=""
    local build_tag_name=""

    if [[ ! "${metadata_version}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,15}$ ]]; then
        echo "Invalid ci_base metadata version: ${metadata_version}" >&2
        return 1
    fi

    # Do not import the historical globally PR-writable stable cache. v3 starts
    # a clean trusted-only discovery alias while the legacy stable runtime tag
    # remains available to existing consumers.
    CI_BASE_STABLE_CACHE_REF="${CI_BASE_STABLE_CACHE_REF:-${stable_tag}-v${metadata_version}}"
    CI_BASE_STABLE_PROMOTION_REF="${stable_tag}"
    export CI_BASE_STABLE_CACHE_REF
    export CI_BASE_STABLE_PROMOTION_REF

    if [[ "${BUILDKITE:-false}" == "true" ]] \
        && ! is_full_git_sha "${BUILDKITE_COMMIT:-}"; then
        echo "Invalid Buildkite commit for ci_base handoff: ${BUILDKITE_COMMIT:-<empty>}" >&2
        return 1
    fi
    if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
        # The external AMD pod template consumes ci_base-$BUILDKITE_COMMIT
        # before repository code can run. Keep this exact compatibility
        # handoff while content/cache writes remain trust-scoped.
        commit_tag=$(ci_base_tag_with_suffix "${stable_tag}" "${BUILDKITE_COMMIT}")
    fi
    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        if [[ ! "${BUILDKITE_BUILD_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
            echo "Invalid Buildkite build ID for ci_base handoff: ${BUILDKITE_BUILD_ID:-<empty>}" >&2
            return 1
        fi
        build_tag=$(ci_base_tag_with_suffix \
            "${stable_tag}" "build-${BUILDKITE_BUILD_ID}")
        build_tag_name="${build_tag##*:}"
        if ((${#build_tag_name} > 128)); then
            echo "Build-scoped ci_base tag exceeds Docker's 128-character limit" >&2
            return 1
        fi
    fi
    CI_BASE_IMAGE_TAG_COMMIT_REF="${commit_tag}"
    CI_BASE_IMAGE_TAG_BUILD_REF="${build_tag}"
    export CI_BASE_IMAGE_TAG_COMMIT_REF
    export CI_BASE_IMAGE_TAG_BUILD_REF

    if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
        if is_ci_base_target; then
            echo "Error: ci_base builds require a content hash" >&2
            return 1
        fi
        CI_BASE_IMAGE="${CI_BASE_IMAGE:-${stable_tag}}"
        export CI_BASE_IMAGE
        return 0
    fi

    trusted_content_tag=$(ci_base_tag_with_suffix \
        "${stable_tag}" "v${metadata_version}-${CI_BASE_CONTENT_HASH}")
    if [[ -n "${scope}" ]]; then
        content_tag=$(ci_base_tag_with_suffix \
            "${stable_tag}" "v${metadata_version}-${scope}-${CI_BASE_CONTENT_HASH}")
    else
        content_tag="${trusted_content_tag}"
    fi
    CI_BASE_IMAGE_TAG_CONTENT_REF="${content_tag}"
    CI_BASE_TRUSTED_CONTENT_REF="${trusted_content_tag}"

    # Content tags are canonical. Untrusted jobs write content under scoped
    # refs while still importing the trusted canonical content ref. The exact
    # commit handoff cannot affect another commit's content discovery.
    # Stable aliases are promoted centrally only after the test image is smoked.
    CI_BASE_IMAGE_TAG="${content_tag}"
    if [[ -n "${commit_tag}" && "${commit_tag}" != "${content_tag}" ]]; then
        CI_BASE_IMAGE_TAG_COMMIT_EXTRA="${commit_tag}"
    else
        CI_BASE_IMAGE_TAG_COMMIT_EXTRA=""
    fi
    if [[ -n "${build_tag}" && "${build_tag}" != "${content_tag}" ]]; then
        CI_BASE_IMAGE_TAG_BUILD_EXTRA="${build_tag}"
    else
        CI_BASE_IMAGE_TAG_BUILD_EXTRA=""
    fi
    export CI_BASE_IMAGE_TAG
    export CI_BASE_IMAGE_TAG_COMMIT_EXTRA
    export CI_BASE_IMAGE_TAG_BUILD_EXTRA
    export CI_BASE_IMAGE_TAG_CONTENT_REF
    export CI_BASE_TRUSTED_CONTENT_REF

    if is_ci_base_target; then
        IMAGE_TAG="${content_tag}"
        CI_BASE_IMAGE="${content_tag}"
        export CI_BASE_IMAGE
        export IMAGE_TAG

        echo "ci_base primary image tag: ${CI_BASE_IMAGE_TAG}"
        if [[ -n "${commit_tag}" ]]; then
            echo "ci_base commit image tag: ${commit_tag}"
        fi
        if [[ -n "${build_tag}" ]]; then
            echo "ci_base build image tag: ${build_tag}"
        fi
        echo "ci_base content image tag: ${content_tag}"
        echo "ci_base stable cache source: ${CI_BASE_STABLE_CACHE_REF}"
        echo "ci_base stable promotion is deferred until the test image passes smoke"
        set_buildkite_metadata "rocm-ci-base-image-content" "${content_tag}" \
            || return 1
        set_buildkite_metadata "rocm-ci-base-image-build" "${build_tag}" \
            || return 1
        return 0
    fi

    if [[ -z "${CI_BASE_IMAGE:-}" || "${CI_BASE_IMAGE}" == "${stable_tag}" ]]; then
        CI_BASE_IMAGE="${content_tag}"
        export CI_BASE_IMAGE
        echo "Using ci_base image: ${CI_BASE_IMAGE}"
    else
        echo "Using provided CI_BASE_IMAGE override: ${CI_BASE_IMAGE}"
    fi
}

publish_ci_base_handoff_ref() {
    local source_ref="${1:-${CI_BASE_IMAGE_TAG_BUILD_REF:-${CI_BASE_IMAGE_TAG_CONTENT_REF:-}}}"
    local content_ref="${CI_BASE_IMAGE_TAG_CONTENT_REF:-}"
    local digest=""
    local handoff_ref=""

    is_ci_base_target || return 0
    if [[ -z "${source_ref}" || -z "${content_ref}" ]]; then
        echo "Cannot publish ci_base handoff without source and content refs" >&2
        return 1
    fi
    if ! digest=$(resolve_image_digest "${source_ref}"); then
        echo "Could not resolve immutable ci_base handoff: ${source_ref}" >&2
        return 1
    fi

    handoff_ref="${content_ref%@*}@${digest}"
    if ! confirm_remote_image_push "${handoff_ref}"; then
        echo "Could not validate immutable ci_base handoff: ${handoff_ref}" >&2
        return 1
    fi
    if command -v buildkite-agent >/dev/null 2>&1; then
        if ! buildkite-agent meta-data set "rocm-ci-base-image" "${handoff_ref}"; then
            echo "Could not publish required ci_base handoff metadata" >&2
            return 1
        fi
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot publish ci_base handoff" >&2
        return 1
    fi
    echo "Published immutable ci_base handoff: ${handoff_ref}"
}

ci_base_output_refs() {
    printf '%s\n' \
        "${IMAGE_TAG:-}" \
        "${CI_BASE_IMAGE_TAG:-}" \
        "${CI_BASE_IMAGE_TAG_COMMIT_EXTRA:-}" \
        "${CI_BASE_IMAGE_TAG_BUILD_EXTRA:-}" \
        | awk 'NF && !seen[$0]++'
}

ci_base_candidate_refs() {
    printf '%s\n' \
        "${CI_BASE_TRUSTED_CONTENT_REF:-}" \
        "${CI_BASE_IMAGE_TAG_CONTENT_REF:-}" \
        "${CI_BASE_STABLE_CACHE_REF:-}" \
        | awk 'NF && !seen[$0]++'
}

find_matching_ci_base_ref() {
    local candidate=""
    local candidate_digest=""
    local immutable_candidate=""
    local probe_status=0
    local identity_status=0

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        # Missing content refs are normal after an input change. Avoid paying
        # the full digest retry budget for each definitely absent candidate.
        probe_status=0
        remote_image_exists "${candidate}" || probe_status=$?
        case "${probe_status}" in
            0) ;;
            1) continue ;;
            *) return "${probe_status}" ;;
        esac
        if ! candidate_digest=$(resolve_image_digest "${candidate}"); then
            echo "Could not pin ci_base candidate: ${candidate}" >&2
            return 2
        fi
        immutable_candidate="${candidate%@*}@${candidate_digest}"
        identity_status=0
        remote_ci_base_identity_is_current_with_retry "${immutable_candidate}" \
            || identity_status=$?
        case "${identity_status}" in
            0)
                printf '%s\n' "${immutable_candidate}"
                return 0
                ;;
            1) ;;
            *) return "${identity_status}" ;;
        esac
    done < <(ci_base_candidate_refs)

    return 1
}

refresh_ci_base_tags_from_ref() {
    local source_ref="$1"
    local source_digest=""
    local tag=""
    local tag_digest=""
    local probe_status=0

    if ! source_digest=$(resolve_image_digest "${source_ref}"); then
        echo "Could not resolve ci_base source ref: ${source_ref}" >&2
        return 2
    fi

    while IFS= read -r tag; do
        [[ -n "${tag}" ]] || continue
        [[ "${tag}" != "${source_ref}" ]] || continue
        probe_status=0
        remote_image_exists "${tag}" || probe_status=$?
        if ((probe_status == 0)); then
            if ! tag_digest=$(resolve_image_digest "${tag}"); then
                echo "Could not resolve existing ci_base tag: ${tag}" >&2
                return 2
            fi
            if [[ "${tag_digest}" == "${source_digest}" ]]; then
                echo "ci_base tag is already current: ${tag}"
                continue
            fi
        elif ((probe_status != 1)); then
            return "${probe_status}"
        fi
        echo "Updating ci_base tag ${tag} -> ${source_ref}"
        if ! docker buildx imagetools create --prefer-index=false \
            -t "${tag}" "${source_ref}"; then
            echo "Failed to update ci_base tag ${tag} from ${source_ref}" >&2
            return 2
        fi
        if ! confirm_remote_image_push "${tag}"; then
            echo "Updated ci_base tag did not become visible: ${tag}" >&2
            return 2
        fi
        if ! tag_digest=$(resolve_image_digest "${tag}") \
            || [[ "${tag_digest}" != "${source_digest}" ]]; then
            echo "Updated ci_base tag does not match its source digest: ${tag}" >&2
            return 2
        fi
    done < <(ci_base_output_refs)
}

validate_ci_base_output_refs() {
    local source_ref="${CI_BASE_IMAGE_TAG_BUILD_REF:-${CI_BASE_IMAGE_TAG_CONTENT_REF:-}}"
    local expected_digest=""
    local image_ref=""
    local image_digest=""

    is_ci_base_target || return 0
    if [[ -z "${source_ref}" ]] \
        || ! expected_digest=$(resolve_image_digest "${source_ref}"); then
        echo "Required ci_base source ref is missing or stale: ${source_ref:-<empty>}" >&2
        return 2
    fi
    while IFS= read -r image_ref; do
        [[ -n "${image_ref}" ]] || continue
        if ! confirm_remote_image_push "${image_ref}"; then
            echo "Required ci_base output ref is missing or stale: ${image_ref}" >&2
            return 2
        fi
        if ! image_digest=$(resolve_image_digest "${image_ref}") \
            || [[ "${image_digest}" != "${expected_digest}" ]]; then
            echo "Required ci_base output ref has the wrong digest: ${image_ref}" >&2
            return 2
        fi
    done < <(ci_base_output_refs)
}

promote_stable_ci_base_tag() {
    local source_ref="${1:-${CI_BASE_IMAGE_TAG_CONTENT_REF:-}}"
    local stable_ref="${CI_BASE_STABLE_PROMOTION_REF:-}"
    local stable_cache_ref="${CI_BASE_STABLE_CACHE_REF:-}"
    local promoted_ref=""
    local digest=""
    local immutable_source=""

    is_ci_base_target || return 0
    wants_stable_ci_base_tag || return 0
    if ! should_push_stable_ci_base_tag; then
        echo "Skipping ci_base stable promotion: build is not the trusted current main tip"
        return 0
    fi
    if [[ -z "${source_ref}" || -z "${stable_ref}" \
        || -z "${stable_cache_ref}" ]]; then
        echo "Cannot promote ci_base stable tag without source and destination refs" >&2
        return 1
    fi
    if ! digest=$(resolve_image_digest "${source_ref}"); then
        echo "Could not pin ci_base source before stable promotion: ${source_ref}" >&2
        return 1
    fi
    immutable_source="${source_ref%@*}@${digest}"
    echo "Promoting ci_base stable tags from ${immutable_source}"
    if ! docker buildx imagetools create --prefer-index=false \
        -t "${stable_ref}" -t "${stable_cache_ref}" "${immutable_source}"; then
        echo "Failed to promote ci_base stable tags" >&2
        return 1
    fi
    for promoted_ref in "${stable_ref}" "${stable_cache_ref}"; do
        if ! confirm_remote_image_push "${promoted_ref}"; then
            echo "Promoted ci_base stable tag did not become visible: ${promoted_ref}" >&2
            return 1
        fi
    done
    set_buildkite_metadata "rocm-ci-base-image-stable" "${stable_ref}"
}

maybe_reuse_matching_ci_base_ref() {
    local matching_ref=""
    local lookup_status=0

    matching_ref=$(find_matching_ci_base_ref) || lookup_status=$?
    if ((lookup_status != 0)); then
        return "${lookup_status}"
    fi
    [[ -n "${matching_ref}" ]] || return 1

    echo "Found existing ci_base image with matching content hash: ${matching_ref}"
    if ! refresh_ci_base_tags_from_ref "${matching_ref}"; then
        echo "ci_base tag refresh failed; refusing to rebuild over a registry error" >&2
        return 2
    fi
    if ! validate_ci_base_output_refs; then
        echo "ci_base alias validation failed; refusing to rebuild over a registry error" >&2
        return 2
    fi
    if ! publish_ci_base_handoff_ref "${matching_ref}"; then
        echo "ci_base handoff validation failed; refusing to rebuild over a registry error" >&2
        return 2
    fi
    set_buildkite_metadata "rocm-ci-base-build-required" "0" || return 2
    echo "Content hashes match -- ci_base is current"
    echo "Skipping build"
    exit 0
}

maybe_skip_existing_image() {
    local remote_revision=""
    local probe_status=0
    local reuse_status=0

    if [[ -z "${IMAGE_TAG:-}" ]]; then
        return 0
    fi

    if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
        echo "FORCE_BUILD=1 set; skipping existing-image check"
        return 0
    fi

    echo "--- :mag: Checking image tag"
    echo "Image tag: ${IMAGE_TAG}"

    remote_image_exists "${IMAGE_TAG}" || probe_status=$?
    if ((probe_status != 0)); then
        if ((probe_status != 1)); then
            return "${probe_status}"
        fi
        if is_ci_base_target && [[ -n "${CI_BASE_CONTENT_HASH:-}" ]]; then
            reuse_status=0
            maybe_reuse_matching_ci_base_ref || reuse_status=$?
            ((reuse_status <= 1)) || return "${reuse_status}"
        fi
        echo "Image not found, proceeding with build"
        return 0
    fi

    IMAGE_EXISTED_BEFORE_BUILD=1

    if is_ci_base_target; then
        if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
            echo "ci_base image already exists and no content hash was configured"
            echo "Skipping build"
            exit 0
        fi

        reuse_status=0
        maybe_reuse_matching_ci_base_ref || reuse_status=$?
        ((reuse_status <= 1)) || return "${reuse_status}"
        echo "No current ci_base image matched the expected content hash"
        echo "Proceeding with build"
        return 0
    fi

    if is_commit_image_target; then
        remote_revision=$(get_remote_image_label "${IMAGE_TAG}" "org.opencontainers.image.revision")
        if [[ -n "${remote_revision}" && "${remote_revision}" != "${BUILDKITE_COMMIT}" ]]; then
            echo "Existing image revision does not match ${BUILDKITE_COMMIT}"
            echo "  found revision: ${remote_revision}"
            echo "Rebuilding image"
            return 0
        fi

        if should_upload_wheel_artifacts; then
            echo "Commit image already exists: ${IMAGE_TAG}"
            echo "Continuing build because this target uploads per-build ROCm artifacts"
            return 0
        fi

        echo "Commit image already exists: ${IMAGE_TAG}"
        echo "Skipping build"
        exit 0
    fi

    echo "Image already exists: ${IMAGE_TAG}"
    echo "Skipping build"
    exit 0
}

setup_builder() {
    echo "--- :buildkite: Setting up buildx builder"

    local setup_mode="${ROCM_SETUP_BUILDX_BUILDER:-auto}"
    local current_driver=""
    local named_driver=""

    if [[ "${setup_mode}" == "0" || "${setup_mode}" == "false" ]]; then
        echo "Using current Docker buildx builder"
        echo "ROCM_SETUP_BUILDX_BUILDER=${setup_mode}; cache exporters may fail if the driver is docker"
        docker buildx inspect --bootstrap
        echo "Active builder:"
        docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
        return 0
    fi

    current_driver=$(buildx_driver || true)
    if [[ "${setup_mode}" != "1" ]] && builder_supports_registry_cache "${current_driver}"; then
        echo "Using current Docker buildx builder with ${current_driver} driver"
        docker buildx inspect --bootstrap
        echo "Active builder:"
        docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
        return 0
    fi

    if [[ "${setup_mode}" != "1" ]]; then
        echo "Current buildx driver '${current_driver:-unknown}' cannot export registry caches"
        echo "Creating or using a cache-capable builder: ${BUILDER_NAME}"
    fi

    if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        named_driver=$(buildx_driver "${BUILDER_NAME}" || true)
        if ! builder_supports_registry_cache "${named_driver}"; then
            echo "Builder '${BUILDER_NAME}' uses ${named_driver:-unknown} driver; using ${BUILDER_NAME}-cache instead"
            BUILDER_NAME="${BUILDER_NAME}-cache"
        fi
    fi

    if [[ -S "${BUILDKIT_SOCKET}" ]]; then
        echo "Found local buildkitd socket at ${BUILDKIT_SOCKET}"
        echo "Using remote driver to connect to buildkitd"

        if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
            use_existing_builder
        else
            create_and_bootstrap_builder remote "unix://${BUILDKIT_SOCKET}"
        fi
    elif docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        use_existing_builder
    else
        echo "No local buildkitd found, using docker-container driver"
        create_and_bootstrap_builder docker-container
    fi

    echo "Active builder:"
    docker buildx ls | grep -E '^\*|^NAME' || docker buildx ls
}

validate_cache_branch_tag() {
    local name="$1"
    local value="$2"

    if [[ -n "${value}" \
        && ! "${value}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,59}$ ]]; then
        echo "Invalid ${name}; expected a Docker tag component of at most 60 characters" >&2
        return 1
    fi
}

prepare_git_cache_metadata() {
    local cache_branch_name=""
    local cache_base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    local target_repo_slug=""
    local target_repo_url=""
    local merge_base_ref=""

    if is_ci_base_target; then
        echo "Skipping commit-cache ancestry lookup for content-addressed ci_base"
        return 0
    fi

    if [[ -z "${PARENT_COMMIT:-}" || -z "${VLLM_MERGE_BASE_COMMIT:-}" ]] \
        && git rev-parse --is-shallow-repository 2>/dev/null | grep -q "true"; then
        echo "Shallow clone detected - deepening for cache key computation"
        git_fetch_for_cache --deepen=1 origin || true
    fi

    if [[ -z "${PARENT_COMMIT:-}" ]]; then
        PARENT_COMMIT=$(git rev-parse HEAD~1 2>/dev/null || echo "")
        if [[ -n "${PARENT_COMMIT}" ]]; then
            export PARENT_COMMIT
            echo "Computed parent commit for cache fallback: ${PARENT_COMMIT}"
        else
            echo "Could not determine parent commit"
        fi
    else
        echo "Using provided PARENT_COMMIT: ${PARENT_COMMIT}"
    fi

    if [[ -z "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        cache_branch_name=$(select_cache_branch_name)
        if [[ -z "${cache_branch_name}" && "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
            cache_branch_name="pr-${BUILDKITE_PULL_REQUEST}"
            echo "Using pull request number for ROCm branch cache tag: ${cache_branch_name}"
        fi
    fi

    if [[ -z "${ROCM_CACHE_BRANCH_TAG:-}" && -n "${cache_branch_name}" ]]; then
        ROCM_CACHE_BRANCH_TAG=$(
            compose_cache_branch_tag "$(get_buildkite_repo_slug)" "${cache_branch_name}"
        )
        export ROCM_CACHE_BRANCH_TAG
        echo "Computed ROCm branch cache tag: ${ROCM_CACHE_BRANCH_TAG} (from ${cache_branch_name})"
    elif [[ -n "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        echo "Using provided ROCM_CACHE_BRANCH_TAG: ${ROCM_CACHE_BRANCH_TAG}"
    elif [[ -n "${BUILDKITE_BRANCH:-}" ]]; then
        echo "Skipping ROCm branch cache tag: no usable branch name found"
        echo "  BUILDKITE_BRANCH=${BUILDKITE_BRANCH}"
    fi

    if [[ -z "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}" \
          && -n "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}" \
          && "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
        target_repo_slug=$(get_buildkite_target_repo_slug)
        ROCM_CACHE_UPSTREAM_BRANCH_TAG=$(
            compose_cache_branch_tag "${target_repo_slug}" "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}"
        )
        export ROCM_CACHE_UPSTREAM_BRANCH_TAG
        echo "Computed ROCm upstream branch cache tag: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
    elif [[ -n "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}" ]]; then
        echo "Using provided ROCM_CACHE_UPSTREAM_BRANCH_TAG: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
    fi

    validate_cache_branch_tag \
        ROCM_CACHE_BRANCH_TAG "${ROCM_CACHE_BRANCH_TAG:-}"
    validate_cache_branch_tag \
        ROCM_CACHE_UPSTREAM_BRANCH_TAG "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}"

    if [[ -z "${VLLM_MERGE_BASE_COMMIT:-}" ]]; then
        target_repo_url=$(get_buildkite_target_repo_url)
        merge_base_ref="refs/remotes/vllm-cache-upstream/${cache_base_branch}"
        git_fetch_for_cache --no-tags --depth=200 "${target_repo_url}" \
            "+refs/heads/${cache_base_branch}:${merge_base_ref}" 2>/dev/null || true
        VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD "${merge_base_ref}" 2>/dev/null || echo "")
        if [[ -z "${VLLM_MERGE_BASE_COMMIT}" ]]; then
            git_fetch_for_cache --no-tags --deepen=1000 "${target_repo_url}" \
                "+refs/heads/${cache_base_branch}:${merge_base_ref}" 2>/dev/null || true
            VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD "${merge_base_ref}" 2>/dev/null || echo "")
        fi
        if [[ -n "${VLLM_MERGE_BASE_COMMIT}" ]]; then
            export VLLM_MERGE_BASE_COMMIT
            echo "Computed merge base commit for cache fallback: ${VLLM_MERGE_BASE_COMMIT}"
        else
            echo "Could not determine merge base with ${cache_base_branch}"
        fi
    else
        echo "Using provided VLLM_MERGE_BASE_COMMIT: ${VLLM_MERGE_BASE_COMMIT}"
    fi
}

join_words() {
    local IFS=" "
    printf '%s' "$*"
}

metadata_pair() {
    local key="$1"
    local value="${2:-}"

    printf '%s\t%s\n' "${key}" "${value}"
}

ci_base_metadata_pairs() {
    local dockerfile="${CI_BASE_DOCKERFILE:-${DEFAULT_CI_BASE_DOCKERFILE}}"
    local stages="${CI_BASE_DOCKERFILE_STAGES:-${DEFAULT_CI_BASE_DOCKERFILE_STAGES}}"
    local content_files="${CI_BASE_CONTENT_FILES:-${DEFAULT_CI_BASE_CONTENT_FILES}}"
    local content_files_hash=""
    local base_image=""
    local base_image_digest=""
    local -a content_paths=()
    local -a content_args=()

    read -r -a content_paths <<< "${content_files}"
    if [[ ${#content_paths[@]} -gt 0 ]]; then
        if ! content_files_hash=$(compute_content_hash "${content_paths[@]}"); then
            echo "Failed to hash ci_base metadata content files" >&2
            return 1
        fi
    fi
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile}" "${stages}" "${CI_BASE_CONTENT_ARGS:-}"
    )

    base_image=$(resolve_dockerfile_arg_value "${dockerfile}" "BASE_IMAGE")
    if [[ -n "${base_image}" ]]; then
        if ! base_image_digest=$(resolve_image_digest "${base_image}"); then
            echo "Failed to resolve ci_base metadata digest for ${base_image}" >&2
            return 1
        fi
    fi

    metadata_pair "vllm.ci_base.metadata_version" "${CI_BASE_METADATA_VERSION:-${DEFAULT_CI_BASE_METADATA_VERSION}}"
    metadata_pair "vllm.ci_base.content_hash" "${CI_BASE_CONTENT_HASH:-}"
    metadata_pair "vllm.ci_base.content_files_hash" "${content_files_hash}"
    metadata_pair "vllm.ci_base.content_files" "${content_files}"
    metadata_pair "vllm.ci_base.content_args" "$(join_words "${content_args[@]}")"
    metadata_pair "vllm.ci_base.dockerfile" "${dockerfile}"
    metadata_pair "vllm.ci_base.dockerfile_stages" "${stages}"

    # The parent identity is digest-only: mutable aliases that resolve to the
    # same image must produce byte-identical canonical metadata.
    metadata_pair "vllm.rocm.base_image" "${base_image_digest}"
    metadata_pair "vllm.rocm.base_image_digest" "${base_image_digest}"
    metadata_pair "vllm.rocm.pytorch_rocm_arch" "${PYTORCH_ROCM_ARCH:-}"
    metadata_pair "vllm.rocm.nic_backend" "$(resolve_dockerfile_arg_value "${dockerfile}" "NIC_BACKEND")"
    metadata_pair "vllm.rocm.ainic_version" "$(resolve_dockerfile_arg_value "${dockerfile}" "AINIC_VERSION")"
    metadata_pair "vllm.rocm.ubuntu_codename" "$(resolve_dockerfile_arg_value "${dockerfile}" "UBUNTU_CODENAME")"
    metadata_pair "vllm.rocm.max_jobs" "$(resolve_dockerfile_arg_value "${dockerfile}" "max_jobs")"
    metadata_pair "vllm.rocm.nixl_repo" "$(resolve_dockerfile_arg_value "${dockerfile}" "NIXL_REPO")"
    metadata_pair "vllm.rocm.nixl_commit" "${NIXL_BRANCH:-$(resolve_dockerfile_arg_value "${dockerfile}" "NIXL_BRANCH")}"
    metadata_pair "vllm.rocm.ucx_repo" "$(resolve_dockerfile_arg_value "${dockerfile}" "UCX_REPO")"
    metadata_pair "vllm.rocm.ucx_commit" "${UCX_BRANCH:-$(resolve_dockerfile_arg_value "${dockerfile}" "UCX_BRANCH")}"
    metadata_pair "vllm.rocm.rocshmem_repo" "$(resolve_dockerfile_arg_value "${dockerfile}" "ROCSHMEM_REPO")"
    metadata_pair "vllm.rocm.rocshmem_commit" "${ROCSHMEM_BRANCH:-$(resolve_dockerfile_arg_value "${dockerfile}" "ROCSHMEM_BRANCH")}"
    metadata_pair "vllm.rocm.deepep_repo" "$(resolve_dockerfile_arg_value "${dockerfile}" "DEEPEP_REPO")"
    metadata_pair "vllm.rocm.deepep_commit" "${DEEPEP_BRANCH:-$(resolve_dockerfile_arg_value "${dockerfile}" "DEEPEP_BRANCH")}"
    metadata_pair "vllm.rocm.deepep_nic" "$(resolve_dockerfile_arg_value "${dockerfile}" "DEEPEP_NIC")"
    metadata_pair "vllm.rocm.deepep_rocm_arch" "$(resolve_dockerfile_arg_value "${dockerfile}" "DEEPEP_ROCM_ARCH")"
    metadata_pair "vllm.rocm.nixl_cache_key" "${NIXL_CACHE_KEY:-}"
    metadata_pair "vllm.rocm.rocshmem_cache_key" "${ROCSHMEM_CACHE_KEY:-}"
    metadata_pair "vllm.rocm.deepep_cache_key" "${DEEPEP_CACHE_KEY:-}"
}

write_ci_base_metadata_annotations() {
    local metadata="$1"
    local key=""
    local value=""

    [[ -n "${metadata}" ]] || return 0
    while IFS=$'\t' read -r key value; do
        [[ -n "${key}" && -n "${value}" ]] || continue
        printf '    "%s",\n' \
            "$(hcl_escape_string "manifest:${key}=${value}")"
    done <<< "${metadata}"
}

write_ci_base_metadata_labels() {
    local metadata="$1"
    local key=""
    local value=""

    [[ -n "${metadata}" ]] || return 0
    while IFS=$'\t' read -r key value; do
        [[ -n "${key}" && -n "${value}" ]] || continue
        printf '    "%s" = "%s"\n' \
            "$(hcl_escape_string "${key}")" \
            "$(hcl_escape_string "${value}")"
    done <<< "${metadata}"
}

write_ci_base_label_override() {
    local target_name=""
    local metadata=""
    local -a ci_base_targets=()

    if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
        return 0
    fi

    mapfile -t ci_base_targets < <(
        {
            printf '%s\n' "ci-base-rocm"
            sed -n -E 's/^target "(ci-base-rocm[^"]+)".*/\1/p' "${CI_HCL_PATH}" 2>/dev/null || true
        } | awk '!seen[$0]++'
    )

    if [[ ${#ci_base_targets[@]} -eq 0 ]]; then
        return 0
    fi

    metadata=$(ci_base_metadata_pairs)

    : > "${CI_BASE_LABEL_OVERRIDE_PATH}"
    for target_name in "${ci_base_targets[@]}"; do
        cat >> "${CI_BASE_LABEL_OVERRIDE_PATH}" <<EOF
target "${target_name}" {
  annotations = [
    "manifest:org.opencontainers.image.revision=",
EOF
        write_ci_base_metadata_annotations "${metadata}" >> "${CI_BASE_LABEL_OVERRIDE_PATH}"
        cat >> "${CI_BASE_LABEL_OVERRIDE_PATH}" <<EOF
  ]
  labels = {
    "org.opencontainers.image.revision" = ""
EOF
        write_ci_base_metadata_labels "${metadata}" >> "${CI_BASE_LABEL_OVERRIDE_PATH}"
        cat >> "${CI_BASE_LABEL_OVERRIDE_PATH}" <<EOF
  }
}

EOF
    done

    BAKE_FILES+=(-f "${CI_BASE_LABEL_OVERRIDE_PATH}")
    echo "Appended ci_base metadata label override for targets: ${ci_base_targets[*]}"
}

uses_rocm_csrc_cache() {
    case "${TARGET}" in
        csrc-rocm-ci|test-rocm-ci|test-rocm-ci-with-wheel|test-rocm-ci-with-artifacts|export-wheel-rocm)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

uses_rocm_rust_cache() {
    case "${TARGET}" in
        rust-rocm-ci|test-rocm-ci|test-rocm-ci-with-wheel|test-rocm-ci-with-artifacts|export-wheel-rocm)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

compute_rocm_csrc_content_hash() {
    local bake_dir=""
    local dockerfile_rocm=""
    local content_files="${ROCM_CSRC_CONTENT_FILES:-${DEFAULT_ROCM_CSRC_CONTENT_FILES}}"
    local stages="${ROCM_CSRC_DOCKERFILE_STAGES:-${DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES}}"
    local -a content_paths=()
    local -a content_args=()

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    read -r -a content_paths <<< "${content_files}"
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile_rocm}" "${stages}" "${ROCM_CSRC_CONTENT_ARGS:-}"
    )

    {
        printf 'csrc-input-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${dockerfile_rocm}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile_rocm}" "${content_args[@]}"
        printf 'dockerfile-stages:%s\n' "${stages}"
        if [[ -f "${dockerfile_rocm}" ]]; then
            hash_dockerfile_stages "${dockerfile_rocm}" "${stages}"
        else
            printf 'missing:%s\n' "${dockerfile_rocm}"
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_rocm_csrc_content_hash_if_needed() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local namespace="${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}"

    if [[ "${ROCM_CSRC_CONTENT_CACHE:-1}" == "0" ]] || ! uses_rocm_csrc_cache; then
        return 0
    fi

    ROCM_CSRC_CONTENT_HASH=$(compute_rocm_csrc_content_hash)
    ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF="${cache_repo}:csrc-rocm-${namespace}-input-${ROCM_CSRC_CONTENT_HASH}"
    ROCM_CSRC_CONTENT_CACHE_REF="${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}${ROCM_CACHE_WRITE_SUFFIX}"
    export ROCM_CSRC_CONTENT_HASH
    export ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF
    export ROCM_CSRC_CONTENT_CACHE_REF
    echo "ROCm csrc content cache ref: ${ROCM_CSRC_CONTENT_CACHE_REF}"
}

compute_rocm_rust_content_hash() {
    local bake_dir=""
    local dockerfile_rocm=""
    local content_files="${ROCM_RUST_CONTENT_FILES:-${DEFAULT_ROCM_RUST_CONTENT_FILES}}"
    local stages="base rust_toolchain_input_0 rust_toolchain_input_1 rust-toolchain-input rust_input_0 rust_input_1 rust-input rust-toolchain rust-build"
    local remote_vllm=""
    local -a content_paths=()
    local -a content_args=()

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    read -r -a content_paths <<< "${content_files}"
    remote_vllm=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" REMOTE_VLLM)
    if [[ "${remote_vllm}" == "0" ]]; then
        mapfile -t content_args < <(
            get_content_arg_names \
                "${dockerfile_rocm}" "${stages}" "${ROCM_RUST_CONTENT_ARGS:-}" \
                | grep -Ev '^(VLLM_REPO|VLLM_BRANCH)$'
        )
    else
        mapfile -t content_args < <(
            get_content_arg_names \
                "${dockerfile_rocm}" "${stages}" "${ROCM_RUST_CONTENT_ARGS:-}"
        )
    fi

    {
        printf 'rust-input-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${dockerfile_rocm}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile_rocm}" "${content_args[@]}"
        printf 'dockerfile-stages:%s\n' "${stages}"
        if [[ -f "${dockerfile_rocm}" ]]; then
            hash_dockerfile_stages "${dockerfile_rocm}" "${stages}"
        else
            printf 'missing:%s\n' "${dockerfile_rocm}"
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_rocm_rust_content_hash_if_needed() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local namespace="${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}"

    if [[ "${ROCM_RUST_CONTENT_CACHE:-1}" == "0" ]] || ! uses_rocm_rust_cache; then
        return 0
    fi

    ROCM_RUST_CONTENT_HASH=$(compute_rocm_rust_content_hash)
    ROCM_RUST_TRUSTED_CONTENT_CACHE_REF="${cache_repo}:rust-rocm-${namespace}-input-${ROCM_RUST_CONTENT_HASH}"
    ROCM_RUST_CONTENT_CACHE_REF="${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}${ROCM_CACHE_WRITE_SUFFIX}"
    export ROCM_RUST_CONTENT_HASH
    export ROCM_RUST_TRUSTED_CONTENT_CACHE_REF
    export ROCM_RUST_CONTENT_CACHE_REF
    echo "ROCm Rust content cache ref: ${ROCM_RUST_CONTENT_CACHE_REF}"
}

write_hcl_string_list_entries() {
    local indent="$1"
    local value=""
    shift

    for value in "$@"; do
        value="${value//\\/\\\\}"
        value="${value//\"/\\\"}"
        printf '%s"%s",\n' "${indent}" "${value}"
    done
}

hcl_escape_string() {
    local value="$1"

    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "${value}"
}

write_hcl_string_list() {
    local indent="$1"
    shift

    printf '%s[\n' "${indent}"
    write_hcl_string_list_entries "${indent}  " "$@"
    printf '%s]\n' "${indent}"
}

write_rocm_build_arg_override() {
    local bake_dir=""
    local dockerfile_rocm=""
    local -a arg_names=()
    local arg_name=""
    local arg_value=""

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    mapfile -t arg_names < <(
        {
            get_content_arg_names \
                "${dockerfile_rocm}" \
                "${CI_BASE_DOCKERFILE_STAGES:-${DEFAULT_CI_BASE_DOCKERFILE_STAGES}}" \
                "${CI_BASE_CONTENT_ARGS:-}"
            get_content_arg_names \
                "${dockerfile_rocm}" \
                "${ROCM_CSRC_DOCKERFILE_STAGES:-${DEFAULT_ROCM_CSRC_DOCKERFILE_STAGES}}" \
                "${ROCM_CSRC_CONTENT_ARGS:-}"
            get_content_arg_names "${dockerfile_rocm}" "base rust_toolchain_input_0 rust_toolchain_input_1 rust-toolchain-input rust_input_0 rust_input_1 rust-input rust-toolchain rust-build" "${ROCM_RUST_CONTENT_ARGS:-}"
        } | awk 'NF && !seen[$0]++'
    )

    {
        cat <<EOF
target "_common-rocm" {
  args = {
EOF
        for arg_name in "${arg_names[@]}"; do
            [[ -n "${arg_name}" ]] || continue
            arg_value=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "${arg_name}")
            [[ -n "${arg_value}" ]] || continue
            printf '    %s = "%s"\n' "${arg_name}" "$(hcl_escape_string "${arg_value}")"
        done
        cat <<EOF
  }
}
EOF
    } > "${ROCM_ARG_OVERRIDE_PATH}"

    BAKE_FILES+=(-f "${ROCM_ARG_OVERRIDE_PATH}")
    echo "Appended resolved ROCm Docker ARG override"
}

write_hcl_string_list_attr() {
    local indent="$1"
    local attr="$2"
    shift 2

    printf '%s%s = [\n' "${indent}" "${attr}"
    write_hcl_string_list_entries "${indent}  " "$@"
    printf '%s]\n' "${indent}"
}

validate_cache_export_mode() {
    local mode="$1"
    local env_name="$2"

    case "${mode}" in
        min|max)
            ;;
        *)
            echo "Error: ${env_name} must be one of: min, max"
            exit 1
            ;;
    esac
}

validate_content_cache_export_mode() {
    local mode="$1"
    local env_name="$2"

    case "${mode}" in
        missing|always|never)
            ;;
        *)
            echo "Error: ${env_name} must be one of: missing, always, never"
            exit 1
            ;;
    esac
}

should_export_content_cache_ref() {
    local cache_ref="$1"
    local cache_name="$2"
    local trusted_ref="${3:-${cache_ref}}"
    local mode="${ROCM_CONTENT_CACHE_EXPORT_MODE:-missing}"
    local existing_ref=""

    case "${mode}" in
        always)
            echo "${cache_name} content cache export mode is always; exporting ${cache_ref}"
            return 0
            ;;
        never)
            echo "${cache_name} content cache export mode is never; not exporting ${cache_ref}"
            return 1
            ;;
        missing|"")
            if docker buildx imagetools inspect "${trusted_ref}" >/dev/null 2>&1; then
                existing_ref="${trusted_ref}"
            elif [[ "${cache_ref}" != "${trusted_ref}" ]] \
                && docker buildx imagetools inspect "${cache_ref}" >/dev/null 2>&1; then
                existing_ref="${cache_ref}"
            fi
            if [[ -n "${existing_ref}" ]]; then
                echo "${cache_name} content cache exists at ${existing_ref}; not re-exporting ${cache_ref}"
                return 1
            fi
            echo "${cache_name} content cache missing; will export ${cache_ref}"
            return 0
            ;;
        *)
            echo "Error: ROCM_CONTENT_CACHE_EXPORT_MODE must be one of: missing, always, never"
            exit 1
            ;;
    esac
}

write_rocm_cache_override() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local namespace="${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}"
    local write_suffix="${ROCM_CACHE_WRITE_SUFFIX:-}"
    local content_cache_export_mode="${ROCM_CONTENT_CACHE_EXPORT_MODE:-missing}"
    local csrc_cache_to_mode="${ROCM_CSRC_CACHE_TO_MODE:-max}"
    local rust_cache_to_mode="${ROCM_RUST_CACHE_TO_MODE:-max}"
    local rocm_cache_to_mode="${ROCM_FINAL_CACHE_TO_MODE:-min}"
    local -a csrc_content_cache_from=()
    local -a rust_content_cache_from=()
    local -a combined_content_cache_from=()
    local -a csrc_cache_to=()
    local -a rust_cache_to=()
    local -a rocm_cache_to=()
    local -a export_wheel_cache_to=()

    if ! uses_rocm_csrc_cache && ! uses_rocm_rust_cache; then
        return 0
    fi

    validate_content_cache_export_mode \
        "${content_cache_export_mode}" \
        "ROCM_CONTENT_CACHE_EXPORT_MODE"
    validate_cache_export_mode "${csrc_cache_to_mode}" "ROCM_CSRC_CACHE_TO_MODE"
    validate_cache_export_mode "${rust_cache_to_mode}" "ROCM_RUST_CACHE_TO_MODE"
    validate_cache_export_mode "${rocm_cache_to_mode}" "ROCM_FINAL_CACHE_TO_MODE"
    echo "ROCm content cache export mode: ${content_cache_export_mode}"
    echo "ROCm csrc cache export mode: ${csrc_cache_to_mode}"
    echo "ROCm Rust cache export mode: ${rust_cache_to_mode}"
    echo "ROCm final image cache export mode: ${rocm_cache_to_mode}"

    if [[ -n "${ROCM_CSRC_CONTENT_CACHE_REF:-}" ]]; then
        csrc_content_cache_from+=(
            "type=registry,ref=${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}"
        )
        if [[ "${ROCM_CSRC_CONTENT_CACHE_REF}" != \
            "${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}" ]]; then
            csrc_content_cache_from+=(
                "type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF}"
            )
        fi
        if should_export_content_cache_ref \
            "${ROCM_CSRC_CONTENT_CACHE_REF}" \
            "ROCm csrc" \
            "${ROCM_CSRC_TRUSTED_CONTENT_CACHE_REF}"; then
            csrc_cache_to+=(
                "type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF},mode=${csrc_cache_to_mode},ignore-error=true"
            )
        fi
    fi

    if [[ -n "${ROCM_RUST_CONTENT_CACHE_REF:-}" ]]; then
        rust_content_cache_from+=(
            "type=registry,ref=${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}"
        )
        if [[ "${ROCM_RUST_CONTENT_CACHE_REF}" != \
            "${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}" ]]; then
            rust_content_cache_from+=(
                "type=registry,ref=${ROCM_RUST_CONTENT_CACHE_REF}"
            )
        fi
        if should_export_content_cache_ref \
            "${ROCM_RUST_CONTENT_CACHE_REF}" \
            "ROCm Rust" \
            "${ROCM_RUST_TRUSTED_CONTENT_CACHE_REF}"; then
            rust_cache_to+=(
                "type=registry,ref=${ROCM_RUST_CONTENT_CACHE_REF},mode=${rust_cache_to_mode},ignore-error=true"
            )
        fi
    fi

    combined_content_cache_from=("${csrc_content_cache_from[@]}" "${rust_content_cache_from[@]}")

    # Docker Hub cache exports are best-effort. A cache-only target failure can
    # otherwise cancel the sibling image target before its manifest is pushed.
    if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-${namespace}-${BUILDKITE_COMMIT}${write_suffix},mode=${rocm_cache_to_mode},ignore-error=true"
        )
    fi

    if [[ -n "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        csrc_cache_to+=(
            "type=registry,ref=${cache_repo}:csrc-rocm-${namespace}-branch-${ROCM_CACHE_BRANCH_TAG}${write_suffix},mode=${csrc_cache_to_mode},ignore-error=true"
        )
        rust_cache_to+=(
            "type=registry,ref=${cache_repo}:rust-rocm-${namespace}-branch-${ROCM_CACHE_BRANCH_TAG}${write_suffix},mode=${rust_cache_to_mode},ignore-error=true"
        )
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-${namespace}-branch-${ROCM_CACHE_BRANCH_TAG}${write_suffix},mode=${rocm_cache_to_mode},ignore-error=true"
        )
    fi

    if [[ "${TARGET}" == "test-rocm-ci-with-wheel" ]]; then
        export_wheel_cache_to=()
    else
        export_wheel_cache_to=("${rocm_cache_to[@]}")
    fi

    {
        cat <<EOF
target "csrc-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm_csrc(),
EOF
        write_hcl_string_list "    " "${csrc_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${csrc_cache_to[@]}"
        cat <<EOF
}

target "rust-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm_rust(),
EOF
        write_hcl_string_list "    " "${rust_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${rust_cache_to[@]}"
        cat <<EOF
}

target "test-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${combined_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${rocm_cache_to[@]}"
        cat <<EOF
}

target "export-wheel-rocm" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${combined_content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${export_wheel_cache_to[@]}"
        cat <<EOF
}
EOF
    } > "${CSRC_CACHE_OVERRIDE_PATH}"

    BAKE_FILES+=(-f "${CSRC_CACHE_OVERRIDE_PATH}")
    echo "Appended ROCm cache override with non-fatal registry exports"
}

extract_dependency_pins() {
    local bake_dir=""
    local dockerfile_rocm=""
    local var=""
    local val=""

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    if [[ ! -f "${dockerfile_rocm}" ]]; then
        return 0
    fi

    for var in NIXL_BRANCH UCX_BRANCH ROCSHMEM_BRANCH DEEPEP_BRANCH; do
        if [[ -n "${!var:-}" ]]; then
            echo "Using provided ${var}: ${!var}"
            continue
        fi

        val=$(
            sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${var}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
                "${dockerfile_rocm}" | head -1
        )
        if [[ -n "${val}" ]]; then
            export "${var}=${val}"
            echo "Extracted ${var}=${val} from Dockerfile.rocm"
        fi
    done
}

compute_dependency_cache_keys() {
    local bake_dir=""
    local dockerfile_rocm=""
    local nixl_branch=""
    local ucx_branch=""
    local rocshmem_branch=""
    local deepep_branch=""
    local nixl_material=""
    local rocshmem_material=""
    local deepep_material=""

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    nixl_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "NIXL_BRANCH")
    ucx_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "UCX_BRANCH")
    rocshmem_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "ROCSHMEM_BRANCH")
    deepep_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "DEEPEP_BRANCH")

    if [[ -n "${nixl_branch}" && -n "${ucx_branch}" ]]; then
        nixl_material=$(compose_stage_cache_material "${dockerfile_rocm}" "base build_nixl")
        NIXL_CACHE_KEY=$(
            compose_dependency_cache_key \
                "${nixl_branch}-ucx-${ucx_branch}" \
                "${nixl_material}"
        )
        export NIXL_CACHE_KEY
        echo "NIXL dependency cache key: ${NIXL_CACHE_KEY}"
    fi

    if [[ -n "${rocshmem_branch}" ]]; then
        rocshmem_material=$(compose_stage_cache_material "${dockerfile_rocm}" "base build_rocshmem")
        ROCSHMEM_CACHE_KEY=$(
            compose_dependency_cache_key \
                "${rocshmem_branch}" \
                "${rocshmem_material}"
        )
        export ROCSHMEM_CACHE_KEY
        echo "ROCShmem dependency cache key: ${ROCSHMEM_CACHE_KEY}"
    fi

    if [[ -n "${deepep_branch}" && -n "${rocshmem_branch}" ]]; then
        deepep_material=$(compose_stage_cache_material "${dockerfile_rocm}" "base build_rocshmem build_deepep")
        DEEPEP_CACHE_KEY=$(
            compose_dependency_cache_key \
                "${deepep_branch}-rocshmem-${rocshmem_branch}" \
                "${deepep_material}"
        )
        export DEEPEP_CACHE_KEY
        echo "DeepEP dependency cache key: ${DEEPEP_CACHE_KEY}"
    fi
}

compose_stage_cache_material() {
    local dockerfile="$1"
    local stages="$2"
    local -a content_args=()

    mapfile -t content_args < <(get_content_arg_names "${dockerfile}" "${stages}" "")
    {
        printf 'dockerfile:%s\n' "${dockerfile}"
        printf 'dockerfile-stages:%s\n' "${stages}"
        hash_dockerfile_stages "${dockerfile}" "${stages}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile}" "${content_args[@]}"
    }
}

dependency_cache_ref_exists() {
    local cache_ref="$1"
    registry_ref_exists_with_retry imagetools "${cache_ref}"
}

dependency_cache_ref_for_target() {
    local target="$1"
    local scope="${2:-write}"
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local namespace="${ROCM_CACHE_NAMESPACE:-${DEFAULT_ROCM_CACHE_NAMESPACE}}"
    local suffix="${ROCM_CACHE_WRITE_SUFFIX:-}"

    if [[ "${scope}" == "trusted" ]]; then
        suffix=""
    fi

    case "${target}" in
        nixl-rocm-ci)
            if [[ -n "${NIXL_CACHE_KEY:-}" ]]; then
                printf '%s\n' "${cache_repo}:nixl-rocm-${namespace}-${NIXL_CACHE_KEY}${suffix}"
            elif [[ -n "${NIXL_BRANCH:-}" ]]; then
                printf '%s\n' "${cache_repo}:nixl-rocm-${namespace}-${NIXL_BRANCH}-ucx-${UCX_BRANCH:-}${suffix}"
            fi
            ;;
        rocshmem-rocm-ci)
            if [[ -n "${ROCSHMEM_CACHE_KEY:-}" ]]; then
                printf '%s\n' "${cache_repo}:rocshmem-rocm-${namespace}-${ROCSHMEM_CACHE_KEY}${suffix}"
            elif [[ -n "${ROCSHMEM_BRANCH:-}" ]]; then
                printf '%s\n' "${cache_repo}:rocshmem-rocm-${namespace}-${ROCSHMEM_BRANCH}${suffix}"
            fi
            ;;
        deepep-rocm-ci)
            if [[ -n "${DEEPEP_CACHE_KEY:-}" ]]; then
                printf '%s\n' "${cache_repo}:deepep-rocm-${namespace}-${DEEPEP_CACHE_KEY}${suffix}"
            elif [[ -n "${DEEPEP_BRANCH:-}" ]]; then
                printf '%s\n' "${cache_repo}:deepep-rocm-${namespace}-${DEEPEP_BRANCH}-rocshmem-${ROCSHMEM_BRANCH:-}${suffix}"
            fi
            ;;
    esac
}

dependency_cache_source_exists() {
    local target="$1"
    local write_ref=""
    local trusted_ref=""
    local probe_status=0

    write_ref=$(dependency_cache_ref_for_target "${target}")
    trusted_ref=$(dependency_cache_ref_for_target "${target}" trusted)
    if [[ -n "${trusted_ref}" ]]; then
        dependency_cache_ref_exists "${trusted_ref}" || probe_status=$?
        case "${probe_status}" in
            0)
                echo "Trusted dependency cache exists: ${trusted_ref}"
                return 0
                ;;
            1) ;;
            *) return "${probe_status}" ;;
        esac
    fi
    if [[ "${write_ref}" != "${trusted_ref}" && -n "${write_ref}" ]]; then
        probe_status=0
        dependency_cache_ref_exists "${write_ref}" || probe_status=$?
        case "${probe_status}" in
            0)
                echo "Scoped dependency cache exists: ${write_ref}"
                return 0
                ;;
            1) ;;
            *) return "${probe_status}" ;;
        esac
    fi
    return 1
}

add_dependency_cache_target() {
    local target="$1"

    if printf '%s\n' "${DEPENDENCY_CACHE_TARGETS[@]}" | grep -qx "${target}"; then
        return 0
    fi
    DEPENDENCY_CACHE_TARGETS+=("${target}")
}

add_dependency_cache_target_if_missing() {
    local target="$1"
    local label="$2"
    local cache_ref=""
    local lookup_status=0

    cache_ref=$(dependency_cache_ref_for_target "${target}")
    dependency_cache_source_exists "${target}" || lookup_status=$?
    case "${lookup_status}" in
        0) return 0 ;;
        1)
            echo "${label} dependency cache missing; will seed: ${cache_ref}"
            add_dependency_cache_target "${target}"
            ;;
        *)
            echo "Could not determine whether ${label} dependency cache exists" >&2
            return "${lookup_status}"
            ;;
    esac
}

resolve_ci_base_dependency_targets() {
    local mode="${ROCM_DEP_CACHE_EXPORT_MODE:-missing}"

    [[ "${TARGET}" == "ci-base-rocm-ci-with-deps" ]] || return 0

    case "${mode}" in
        always)
            echo "ROCM_DEP_CACHE_EXPORT_MODE=always; exporting all dependency caches serially"
            for target in nixl-rocm-ci rocshmem-rocm-ci deepep-rocm-ci; do
                if [[ -n "$(dependency_cache_ref_for_target "${target}")" ]]; then
                    add_dependency_cache_target "${target}"
                fi
            done
            ;;
        never)
            BAKE_TARGETS=("ci-base-rocm-ci")
            DEPENDENCY_CACHE_TARGETS=()
            echo "ROCM_DEP_CACHE_EXPORT_MODE=never; building ci_base without dependency cache exports"
            return 0
            ;;
        missing|"")
            ;;
        *)
            echo "Error: ROCM_DEP_CACHE_EXPORT_MODE must be one of: missing, always, never"
            exit 1
            ;;
    esac

    if [[ "${mode}" != "always" && -n "${NIXL_CACHE_KEY:-}" ]]; then
        add_dependency_cache_target_if_missing "nixl-rocm-ci" "NIXL" \
            || return $?
    fi

    if [[ "${mode}" != "always" && -n "${ROCSHMEM_CACHE_KEY:-}" ]]; then
        add_dependency_cache_target_if_missing "rocshmem-rocm-ci" "ROCShmem" \
            || return $?
    fi

    if [[ "${mode}" != "always" && -n "${DEEPEP_CACHE_KEY:-}" ]]; then
        add_dependency_cache_target_if_missing "deepep-rocm-ci" "DeepEP" \
            || return $?
    fi

    # DeepEP inherits from ROCShmem. If ROCShmem is being seeded, seed DeepEP too
    # so the pair stays consistent for future ci_base rebuilds.
    if printf '%s\n' "${DEPENDENCY_CACHE_TARGETS[@]}" | grep -qx "rocshmem-rocm-ci" \
        && ! printf '%s\n' "${DEPENDENCY_CACHE_TARGETS[@]}" | grep -qx "deepep-rocm-ci" \
        && [[ -n "${DEEPEP_BRANCH:-}" ]]; then
        echo "ROCShmem cache is missing; also seeding DeepEP cache"
        add_dependency_cache_target "deepep-rocm-ci"
    fi

    BAKE_TARGETS=("ci-base-rocm-ci")
    if [[ ${#DEPENDENCY_CACHE_TARGETS[@]} -eq 0 ]]; then
        echo "All dependency caches exist; building ci_base without dependency cache exports"
    else
        echo "Resolved dependency cache seed targets: ${DEPENDENCY_CACHE_TARGETS[*]}"
        echo "Resolved ci_base bake targets: ${BAKE_TARGETS[*]}"
    fi
}

bake_config_targets() {
    printf '%s\n' "${DEPENDENCY_CACHE_TARGETS[@]}" "${BAKE_TARGETS[@]}" \
        | awk 'NF && !seen[$0]++'
}

print_bake_config() {
    local -a print_targets=()

    echo "--- :page_facing_up: Resolved bake configuration"
    mapfile -t print_targets < <(bake_config_targets)
    docker buildx bake "${BAKE_FILES[@]}" --print "${print_targets[@]}" | tee "${BAKE_CONFIG_FILE}"

    if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        buildkite-agent artifact upload "${BAKE_CONFIG_FILE}" || true
        echo "Uploaded ${BAKE_CONFIG_FILE} as Buildkite artifact"
    else
        echo "Saved bake config to ${BAKE_CONFIG_FILE} (not in Buildkite, skipping upload)"
    fi
}

confirm_remote_image_push() {
    local image_ref="$1"
    local remote_revision=""

    if is_ci_base_target; then
        if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
            remote_image_exists "${image_ref}"
            return
        fi

        if remote_ci_base_identity_is_current_with_retry "${image_ref}"; then
            return 0
        fi

        echo "Remote image does not have the expected complete ci_base identity."
        return 1
    fi

    if ! remote_image_exists "${image_ref}"; then
        return 1
    fi

    if is_commit_image_target; then
        remote_revision=$(get_remote_image_label_with_retry "${image_ref}" "org.opencontainers.image.revision")
        if [[ -n "${remote_revision}" && "${remote_revision}" == "${BUILDKITE_COMMIT}" ]]; then
            return 0
        fi

        if [[ -z "${remote_revision}" \
              && ${IMAGE_EXISTED_BEFORE_BUILD} -eq 0 ]] \
            && image_tag_is_commit_scoped; then
            echo "Remote image exists under a commit-scoped tag; accepting push despite missing revision label."
            return 0
        fi

        echo "Remote image exists but revision label does not match ${BUILDKITE_COMMIT}."
        echo "  found revision: ${remote_revision:-<missing>}"
        return 1
    fi

    return 0
}

verify_dependency_cache_ref() {
    local cache_ref="$1"
    local attempts="${ROCM_DEP_CACHE_VERIFY_ATTEMPTS:-6}"
    local delay_secs="${ROCM_DEP_CACHE_VERIFY_DELAY:-5}"
    local attempt

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if dependency_cache_ref_exists "${cache_ref}"; then
            echo "Dependency cache confirmed: ${cache_ref}"
            return 0
        fi
        if [[ ${attempt} -lt ${attempts} ]]; then
            echo "Dependency cache not visible yet (${attempt}/${attempts}): ${cache_ref}"
            sleep "${delay_secs}"
        fi
    done

    echo "ERROR: dependency cache was not confirmed after upload: ${cache_ref}"
    return 1
}

seed_dependency_caches_if_needed() {
    local target=""
    local cache_ref=""

    if [[ "${TARGET}" != "ci-base-rocm-ci-with-deps" ]]; then
        return 0
    fi
    if [[ ${#DEPENDENCY_CACHE_TARGETS[@]} -eq 0 ]]; then
        return 0
    fi

    echo "--- :docker: Seeding ROCm dependency caches"
    echo "Dependency cache uploads are required for this build."
    echo "Seeding serially to avoid concurrent Docker Hub cache exporters."

    for target in "${DEPENDENCY_CACHE_TARGETS[@]}"; do
        cache_ref=$(dependency_cache_ref_for_target "${target}")
        if [[ -z "${cache_ref}" ]]; then
            echo "ERROR: could not resolve dependency cache ref for ${target}"
            return 1
        fi

        echo "--- :docker: Seeding ${target}"
        echo "Expected cache ref: ${cache_ref}"
        if ! docker buildx bake \
            "${BAKE_FILES[@]}" \
            --progress "${BUILDKIT_PROGRESS:-plain}" \
            "${target}"; then
            return 1
        fi
        if ! verify_dependency_cache_ref "${cache_ref}"; then
            return 1
        fi
    done
}

annotate_cache_export_warning() {
    local build_rc="$1"
    local image_ref="${2:-${IMAGE_TAG:-}}"

    if ! command -v buildkite-agent >/dev/null 2>&1; then
        return 0
    fi

    buildkite-agent annotate \
        --style warning \
        --context "cache-export-warning" \
        "### :warning: Docker cache export failed (non-fatal)

Image was pushed successfully: \`${image_ref}\`

The BuildKit build returned exit code ${build_rc}, but the expected image
is present in the registry. Treating this as a registry cache export failure
so tests can continue with the pushed image." 2>/dev/null || true
}

run_bake() {
    local build_rc=0
    local confirmation_ref="${IMAGE_TAG:-}"

    if is_ci_base_target && [[ -n "${CI_BASE_IMAGE_TAG_BUILD_REF:-}" ]]; then
        confirmation_ref="${CI_BASE_IMAGE_TAG_BUILD_REF}"
    fi

    echo "--- :docker: Building ${TARGET}"
    docker buildx bake \
        "${BAKE_FILES[@]}" \
        --progress "${BUILDKIT_PROGRESS:-plain}" \
        "${BAKE_TARGETS[@]}" || build_rc=$?

    if [[ ${build_rc} -eq 0 ]]; then
        if ! validate_ci_base_output_refs; then
            return 1
        fi
        echo "--- :white_check_mark: Build complete"
        return 0
    fi

    echo ""
    echo "WARNING: docker buildx bake exited with code ${build_rc}"

    if [[ -n "${confirmation_ref}" ]]; then
        echo "Checking if image was pushed successfully..."
        if confirm_remote_image_push "${confirmation_ref}"; then
            if is_ci_base_target; then
                if ! refresh_ci_base_tags_from_ref "${confirmation_ref}" \
                    || ! validate_ci_base_output_refs; then
                    echo "ERROR: ci_base build left one or more required aliases missing" >&2
                    return "${build_rc}"
                fi
            fi
            echo ""
            echo "WARNING: Build reported failure (rc=${build_rc}) but the"
            echo "         image was pushed successfully: ${confirmation_ref}"
            echo ""
            echo "         Treating this as a non-fatal registry cache export failure."
            echo "         The image is usable, but registry cache may be cold on the next build."
            echo ""
            annotate_cache_export_warning "${build_rc}" "${confirmation_ref}"
            echo "--- :white_check_mark: Build complete"
            return 0
        fi

        echo ""
        echo "ERROR: Build failed and image was NOT confirmed: ${confirmation_ref}"
        echo "       This is a real build failure, not a cache export warning."
        echo ""
    fi

    return "${build_rc}"
}

upload_wheel_artifacts_if_present() {
    local wheel_dir="./wheel-export"
    local artifact_dir="artifacts/vllm-rocm-install"
    local archive_name="vllm-rocm-install.tar.gz"
    local metadata_dir="${wheel_dir}/.vllm-ci-artifact"
    local build_base_digest=""
    local build_base_image=""
    local expected_build_base_image=""
    local expected_native_base_image=""
    local native_base_image=""
    local native_base_digest=""
    local base_alias=""
    local whl=""
    local whl_name=""
    local -a wheels=()

    if ! should_upload_wheel_artifacts; then
        return 0
    fi

    if [[ -d "${wheel_dir}" ]]; then
        mapfile -t wheels < <(find "${wheel_dir}" -maxdepth 1 -type f -name '*.whl' -print)
    fi
    if [[ ${#wheels[@]} -ne 1 ]]; then
        echo "Expected exactly one ROCm wheel in ${wheel_dir}; found ${#wheels[@]}" >&2
        return 1
    fi
    whl="${wheels[0]}"
    whl_name=$(basename "${whl}")
    native_base_image="${CI_BASE_IMAGE_TAG_COMMIT_REF:-${CI_BASE_IMAGE:-}}"
    build_base_image="${CI_BASE_IMAGE_TAG_BUILD_REF:-}"
    if [[ -z "${native_base_image}" ]]; then
        echo "Native ROCm artifact requires a ci_base image reference" >&2
        return 1
    fi
    if [[ "${BUILDKITE:-false}" == "true" ]]; then
        expected_native_base_image=$(ci_base_tag_with_suffix \
            "${CI_BASE_STABLE_PROMOTION_REF:-rocm/vllm-dev:ci_base}" \
            "${BUILDKITE_COMMIT:-}")
        if [[ "${native_base_image}" != "${expected_native_base_image}" ]]; then
            echo "Native ROCm artifact requires the exact ci_base commit handoff: ${native_base_image}" >&2
            return 1
        fi
        expected_build_base_image=$(ci_base_tag_with_suffix \
            "${CI_BASE_STABLE_PROMOTION_REF:-rocm/vllm-dev:ci_base}" \
            "build-${BUILDKITE_BUILD_ID:-}")
        if [[ "${build_base_image}" != "${expected_build_base_image}" ]]; then
            echo "Native ROCm artifact requires the exact ci_base build handoff: ${build_base_image:-<empty>}" >&2
            return 1
        fi
        if [[ ! "${CI_BASE_IMAGE:-}" =~ @sha256:[0-9a-f]{64}$ ]]; then
            echo "ROCm artifact build base must be digest-pinned: ${CI_BASE_IMAGE:-<empty>}" >&2
            return 1
        fi
        build_base_digest="${CI_BASE_IMAGE##*@}"
        for base_alias in "${native_base_image}" "${build_base_image}"; do
            if ! native_base_digest=$(resolve_image_digest "${base_alias}"); then
                echo "Could not resolve native ci_base handoff: ${base_alias}" >&2
                return 1
            fi
            if [[ "${native_base_digest}" != "${build_base_digest}" ]]; then
                echo "Native ci_base handoff does not match the artifact build base" >&2
                echo "  native: ${base_alias}@${native_base_digest}" >&2
                echo "  build:  ${CI_BASE_IMAGE}" >&2
                return 1
            fi
        done
    fi

    echo "--- :package: Uploading ROCm vLLM install artifact"
    rm -rf "${artifact_dir}" "${metadata_dir}"
    mkdir -p "${artifact_dir}" "${metadata_dir}"

    printf '%s\n' "${BUILDKITE_COMMIT:-local}" > "${metadata_dir}/commit.txt"
    printf '%s\n' "${native_base_image}" > "${metadata_dir}/native-base-image.txt"
    printf '%s\n' "${native_base_image}" "${build_base_image}" \
        | awk 'NF && !seen[$0]++' > "${metadata_dir}/native-base-images.txt"
    printf '%s\n' "${CI_BASE_IMAGE:-}" > "${metadata_dir}/ci-base-image.txt"
    printf '%s\n' "${IMAGE_TAG:-}" > "${metadata_dir}/fallback-image.txt"
    printf '%s\n' "${whl_name}" > "${metadata_dir}/wheel-filename.txt"

    tar -C "${wheel_dir}" -czf "${artifact_dir}/${archive_name}" .
    (
        cd "${artifact_dir}"
        sha256sum "${archive_name}" > "${archive_name}.sha256"
    )
    echo "Created ${archive_name}: $(du -sh "${artifact_dir}/${archive_name}" | cut -f1)"
    cp "${metadata_dir}"/*.txt "${artifact_dir}/"
    cp "${whl}" "${artifact_dir}/${whl_name}"
    echo "Copied ${whl_name}: $(du -sh "${artifact_dir}/${whl_name}" | cut -f1)"

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "${artifact_dir}/*" || return 1
        echo "ROCm vLLM install artifacts uploaded to ${artifact_dir}/"
    elif [[ "${BUILDKITE:-false}" == "true" ]]; then
        echo "buildkite-agent not found; cannot upload required ROCm artifacts" >&2
        return 1
    else
        echo "Not in Buildkite, skipping artifact upload"
    fi

    rm -rf "${wheel_dir}"
}

main() {
    init_config "$@"
    configure_cache_write_scope
    print_header
    validate_inputs
    normalize_ci_worktree_modes
    load_ci_hcl
    init_bake_files
    compute_ci_base_hash_if_needed
    configure_ci_base_image_refs
    if ! maybe_skip_existing_image; then
        return 1
    fi
    setup_builder
    prepare_git_cache_metadata
    extract_dependency_pins
    write_rocm_build_arg_override
    compute_dependency_cache_keys
    write_ci_base_label_override
    compute_rocm_csrc_content_hash_if_needed
    compute_rocm_rust_content_hash_if_needed
    write_rocm_cache_override
    if ! resolve_ci_base_dependency_targets; then
        return 1
    fi
    print_bake_config
    if [[ "${BAKE_PRINT_ONLY:-0}" == "1" ]]; then
        echo "BAKE_PRINT_ONLY=1 set; skipping build"
        return 0
    fi
    if should_upload_wheel_artifacts; then
        # wheel-export is an output directory, not a BuildKit cache. Starting
        # clean prevents a failed/retried export from packaging a stale wheel.
        rm -rf ./wheel-export
    fi
    if is_ci_base_target; then
        set_buildkite_metadata "rocm-ci-base-build-required" "1" || return 1
    fi
    if ! seed_dependency_caches_if_needed; then
        return 1
    fi
    if ! run_bake; then
        return 1
    fi
    publish_ci_base_handoff_ref
    upload_wheel_artifacts_if_present
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
