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
DEFAULT_CI_BASE_CONTENT_FILES="requirements/common.txt requirements/rocm.txt requirements/test/rocm.txt docker/Dockerfile.rocm_base tools/install_torchcodec_rocm.sh tests/vllm_test_utils"
DEFAULT_CI_BASE_DOCKERFILE="docker/Dockerfile.rocm"
DEFAULT_CI_BASE_DOCKERFILE_STAGES="base build_rixl build_rocshmem build_deepep mori_base ci_base"
IMAGE_EXISTED_BEFORE_BUILD=0

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

    suffix="$(cache_scope_suffix)"
    prefix="$(clean_docker_tag "${repo_slug}")-$(clean_docker_tag "${branch}")"
    max_prefix_len=$((128 - ${#suffix} - 1))
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

compute_content_hash() {
    local path
    local file

    for path in "$@"; do
        if [[ -d "${path}" ]]; then
            while IFS= read -r -d '' file; do
                printf 'file:%s\n' "${file}"
                sha256sum "${file}"
            done < <(find "${path}" -type f -print0 | sort -z)
        elif [[ -f "${path}" ]]; then
            printf 'file:%s\n' "${path}"
            sha256sum "${path}"
        else
            printf 'missing:%s\n' "${path}"
        fi
    done | sha256sum | cut -d' ' -f1
}

compose_dependency_cache_key() {
    local prefix="$1"
    local material="$2"
    local cleaned_prefix=""

    cleaned_prefix=$(clean_docker_tag "${prefix}" | cut -c1-96)
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

compute_ci_base_content_hash() {
    local -a content_paths=()
    local -a content_args=()
    local dockerfile="${CI_BASE_DOCKERFILE:-}"
    local stages="${CI_BASE_DOCKERFILE_STAGES:-}"

    read -r -a content_paths <<< "${CI_BASE_CONTENT_FILES}"
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile}" "${stages}" "${CI_BASE_CONTENT_ARGS:-}"
    )

    {
        printf 'content-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        if [[ -n "${dockerfile}" ]]; then
            printf 'dockerfile:%s\n' "${dockerfile}"
            printf 'resolved-build-args:\n'
            hash_dockerfile_arg_values "${dockerfile}" "${content_args[@]}"
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

extract_dockerfile_arg_default() {
    local dockerfile="$1"
    local arg_name="$2"

    sed -n -E "s/^[[:space:]]*ARG[[:space:]]+${arg_name}=\"?([^\"[:space:]]+)\"?.*/\\1/p" \
        "${dockerfile}" | head -1
}

resolve_image_digest() {
    local image_ref="$1"

    docker buildx imagetools inspect "${image_ref}" 2>/dev/null \
        | sed -n -E 's/^Digest:[[:space:]]+//p' \
        | head -1
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
        printf 'arg:%s=%s\n' "${arg_name}" "${arg_value:-<empty>}"
        if [[ "${arg_name}" == "BASE_IMAGE" && -n "${arg_value}" ]]; then
            digest=$(resolve_image_digest "${arg_value}")
            printf 'arg:%s.digest=%s\n' "${arg_name}" "${digest:-unknown}"
        fi
    done
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
                image_ref + "@" + digest,
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

remote_image_exists() {
    local image_ref="$1"
    docker manifest inspect "${image_ref}" >/dev/null 2>&1
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
    CI_HCL_SOURCE="${CI_HCL_SOURCE:-${CI_HCL_FILE:-${DEFAULT_CI_HCL_SOURCE}}}"
    VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake-rocm.hcl}"
    BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
    BUILDKIT_SOCKET="${BUILDKIT_SOCKET:-/run/buildkit/buildkitd.sock}"
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
    CI_BASE_CONTENT_FILES="${CI_BASE_CONTENT_FILES:-${DEFAULT_CI_BASE_CONTENT_FILES}}"
    CI_BASE_DOCKERFILE="${CI_BASE_DOCKERFILE:-${DEFAULT_CI_BASE_DOCKERFILE}}"
    CI_BASE_DOCKERFILE_STAGES="${CI_BASE_DOCKERFILE_STAGES:-${DEFAULT_CI_BASE_DOCKERFILE_STAGES}}"
    CI_BASE_IMAGE_TAG="${CI_BASE_IMAGE_TAG:-rocm/vllm-dev:ci_base}"
    export PYTORCH_ROCM_ARCH

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

compute_ci_base_hash_if_needed() {
    if [[ -z "${CI_BASE_CONTENT_FILES:-}" ]]; then
        return 0
    fi

    CI_BASE_CONTENT_HASH=$(compute_ci_base_content_hash)
    export CI_BASE_CONTENT_HASH
    echo "ci_base content hash: ${CI_BASE_CONTENT_HASH:0:16}..."
}

should_push_stable_ci_base_tag() {
    if [[ "${CI_BASE_PUSH_STABLE_TAG:-}" == "1" ]]; then
        return 0
    fi
    if [[ "${CI_BASE_PUSH_STABLE_TAG:-}" == "0" ]]; then
        return 1
    fi

    [[ "${NIGHTLY:-0}" == "1" && "${BUILDKITE_BRANCH:-}" == "${CI_BASE_STABLE_BRANCH:-main}" ]]
}

ci_base_tag_with_suffix() {
    local base_tag="$1"
    local suffix="$2"

    printf '%s-%s\n' "${base_tag}" "$(clean_docker_tag "${suffix}")"
}

configure_ci_base_image_refs() {
    local stable_tag="${CI_BASE_IMAGE_TAG:-rocm/vllm-dev:ci_base}"
    local content_tag=""
    local commit_tag=""
    local primary_tag=""

    if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
        CI_BASE_IMAGE="${CI_BASE_IMAGE:-${stable_tag}}"
        export CI_BASE_IMAGE
        return 0
    fi

    content_tag=$(ci_base_tag_with_suffix "${stable_tag}" "${CI_BASE_CONTENT_HASH}")
    if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
        commit_tag=$(ci_base_tag_with_suffix "${stable_tag}" "${BUILDKITE_COMMIT}")
        CI_BASE_IMAGE_TAG_COMMIT="${commit_tag}"
        export CI_BASE_IMAGE_TAG_COMMIT
    fi

    if should_push_stable_ci_base_tag; then
        primary_tag="${content_tag}"
        CI_BASE_IMAGE_TAG_STABLE="${stable_tag}"
    else
        primary_tag="${commit_tag:-${content_tag}}"
        CI_BASE_IMAGE_TAG_STABLE=""
    fi
    CI_BASE_IMAGE_TAG="${primary_tag}"
    if [[ "${primary_tag}" == "${content_tag}" ]]; then
        CI_BASE_IMAGE_TAG_CONTENT=""
    else
        CI_BASE_IMAGE_TAG_CONTENT="${content_tag}"
    fi
    export CI_BASE_IMAGE_TAG CI_BASE_IMAGE_TAG_CONTENT CI_BASE_IMAGE_TAG_STABLE

    if is_ci_base_target; then
        IMAGE_TAG="${primary_tag}"
        export IMAGE_TAG

        echo "ci_base primary image tag: ${CI_BASE_IMAGE_TAG}"
        if [[ -n "${CI_BASE_IMAGE_TAG_COMMIT:-}" ]]; then
            echo "ci_base commit image tag: ${CI_BASE_IMAGE_TAG_COMMIT}"
        fi
        echo "ci_base content image tag: ${content_tag}"
        if [[ -n "${CI_BASE_IMAGE_TAG_STABLE}" ]]; then
            echo "ci_base stable alias will also be pushed: ${CI_BASE_IMAGE_TAG_STABLE}"
        else
            echo "ci_base stable alias will not be pushed for this build"
            echo "Set NIGHTLY=1 on ${CI_BASE_STABLE_BRANCH:-main} to refresh ${stable_tag}"
        fi
        return 0
    fi

    if [[ -z "${CI_BASE_IMAGE:-}" || "${CI_BASE_IMAGE}" == "${stable_tag}" ]]; then
        CI_BASE_IMAGE="${primary_tag}"
        export CI_BASE_IMAGE
        echo "Using ci_base image: ${CI_BASE_IMAGE}"
    else
        echo "Using provided CI_BASE_IMAGE override: ${CI_BASE_IMAGE}"
    fi
}

ci_base_candidate_refs() {
    printf '%s\n' \
        "${IMAGE_TAG:-}" \
        "${CI_BASE_IMAGE_TAG:-}" \
        "${CI_BASE_IMAGE_TAG_COMMIT:-}" \
        "${CI_BASE_IMAGE_TAG_CONTENT:-}" \
        "${CI_BASE_IMAGE_TAG_STABLE:-}" \
        | awk 'NF && !seen[$0]++'
}

find_matching_ci_base_ref() {
    local candidate=""
    local candidate_hash=""

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        remote_image_exists "${candidate}" || continue
        candidate_hash=$(get_remote_image_label "${candidate}" "vllm.ci_base.content_hash")
        if [[ "${candidate_hash}" == "${CI_BASE_CONTENT_HASH}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done < <(ci_base_candidate_refs)

    return 1
}

refresh_ci_base_tags_from_ref() {
    local source_ref="$1"
    local tag=""
    local tag_hash=""

    while IFS= read -r tag; do
        [[ -n "${tag}" ]] || continue
        [[ "${tag}" != "${source_ref}" ]] || continue
        tag_hash=$(get_remote_image_label "${tag}" "vllm.ci_base.content_hash")
        if [[ "${tag_hash}" == "${CI_BASE_CONTENT_HASH}" ]]; then
            echo "ci_base tag is already current: ${tag}"
            continue
        fi
        echo "Updating ci_base tag ${tag} -> ${source_ref}"
        docker buildx imagetools create -t "${tag}" "${source_ref}"
    done < <(ci_base_candidate_refs)
}

maybe_skip_existing_image() {
    local remote_hash=""
    local remote_revision=""
    local matching_ref=""

    if [[ -z "${IMAGE_TAG:-}" ]]; then
        return 0
    fi

    if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
        echo "FORCE_BUILD=1 set; skipping existing-image check"
        return 0
    fi

    echo "--- :mag: Checking image tag"
    echo "Image tag: ${IMAGE_TAG}"

    if ! remote_image_exists "${IMAGE_TAG}"; then
        if is_ci_base_target && [[ -n "${CI_BASE_CONTENT_HASH:-}" ]]; then
            matching_ref=$(find_matching_ci_base_ref || true)
            if [[ -n "${matching_ref}" ]]; then
                echo "Found existing ci_base image with matching content hash: ${matching_ref}"
                if ! refresh_ci_base_tags_from_ref "${matching_ref}"; then
                    echo "ci_base tag refresh failed; rebuilding to push expected tags"
                    return 0
                fi
                echo "Content hashes match -- ci_base is current"
                echo "Skipping build"
                exit 0
            fi
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

        remote_hash=$(get_remote_image_label "${IMAGE_TAG}" "vllm.ci_base.content_hash")
        if [[ -n "${remote_hash}" ]]; then
            echo "Remote ci_base content hash: ${remote_hash:0:16}..."
            if [[ "${remote_hash}" == "${CI_BASE_CONTENT_HASH}" ]]; then
                if ! refresh_ci_base_tags_from_ref "${IMAGE_TAG}"; then
                    echo "ci_base tag refresh failed; rebuilding to push expected tags"
                    return 0
                fi
                echo "Content hashes match -- ci_base is current"
                echo "Skipping build"
                exit 0
            fi

            echo "Content hashes differ -- ci_base is stale, rebuilding"
            return 0
        fi

        echo "Remote ci_base has no content-hash label; rebuilding to add one"
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

prepare_git_cache_metadata() {
    local cache_branch_name=""
    local cache_base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    local target_repo_slug=""
    local target_repo_url=""
    local merge_base_ref=""

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

write_ci_base_label_override() {
    local target_name=""
    local -a ci_base_targets=()

    BAKE_FILES=(-f "${VLLM_BAKE_FILE}" -f "${CI_HCL_PATH}")

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

    : > "${CI_BASE_LABEL_OVERRIDE_PATH}"
    for target_name in "${ci_base_targets[@]}"; do
        cat >> "${CI_BASE_LABEL_OVERRIDE_PATH}" <<EOF
target "${target_name}" {
  annotations = [
    "manifest:org.opencontainers.image.revision=",
  ]
  labels = {
    "org.opencontainers.image.revision" = ""
    "vllm.ci_base.content_hash" = "${CI_BASE_CONTENT_HASH}"
  }
}

EOF
    done

    BAKE_FILES+=(-f "${CI_BASE_LABEL_OVERRIDE_PATH}")
    echo "Appended ci_base content-hash label override for targets: ${ci_base_targets[*]}"
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

compute_rocm_csrc_content_hash() {
    local bake_dir=""
    local dockerfile_rocm=""
    local -a content_paths=(
        "requirements/common.txt"
        "requirements/rocm.txt"
        "setup.py"
        "CMakeLists.txt"
        "cmake"
        "csrc"
        "vllm/envs.py"
        "vllm/__init__.py"
    )
    local -a content_args=()

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    mapfile -t content_args < <(
        get_content_arg_names "${dockerfile_rocm}" "base csrc-build" "${ROCM_CSRC_CONTENT_ARGS:-}"
    )

    {
        printf 'csrc-input-files-hash:%s\n' "$(compute_content_hash "${content_paths[@]}")"
        printf 'dockerfile:%s\n' "${dockerfile_rocm}"
        printf 'resolved-build-args:\n'
        hash_dockerfile_arg_values "${dockerfile_rocm}" "${content_args[@]}"
        printf 'dockerfile-stages:base csrc-build\n'
        if [[ -f "${dockerfile_rocm}" ]]; then
            hash_dockerfile_stages "${dockerfile_rocm}" "base csrc-build"
        else
            printf 'missing:%s\n' "${dockerfile_rocm}"
        fi
    } | sha256sum | cut -d' ' -f1
}

compute_rocm_csrc_content_hash_if_needed() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"

    if [[ "${ROCM_CSRC_CONTENT_CACHE:-1}" == "0" ]] || ! uses_rocm_csrc_cache; then
        return 0
    fi

    ROCM_CSRC_CONTENT_HASH=$(compute_rocm_csrc_content_hash)
    ROCM_CSRC_CONTENT_CACHE_REF="${cache_repo}:csrc-rocm-input-${ROCM_CSRC_CONTENT_HASH}"
    export ROCM_CSRC_CONTENT_HASH
    export ROCM_CSRC_CONTENT_CACHE_REF
    echo "ROCm csrc content cache ref: ${ROCM_CSRC_CONTENT_CACHE_REF}"
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
            get_content_arg_names "${dockerfile_rocm}" "base csrc-build" "${ROCM_CSRC_CONTENT_ARGS:-}"
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

write_rocm_cache_override() {
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local csrc_cache_to_mode="${ROCM_CSRC_CACHE_TO_MODE:-max}"
    local rocm_cache_to_mode="${ROCM_FINAL_CACHE_TO_MODE:-min}"
    local -a content_cache_from=()
    local -a csrc_cache_to=()
    local -a rocm_cache_to=()
    local -a export_wheel_cache_to=()

    if ! uses_rocm_csrc_cache; then
        return 0
    fi

    validate_cache_export_mode "${csrc_cache_to_mode}" "ROCM_CSRC_CACHE_TO_MODE"
    validate_cache_export_mode "${rocm_cache_to_mode}" "ROCM_FINAL_CACHE_TO_MODE"
    echo "ROCm csrc cache export mode: ${csrc_cache_to_mode}"
    echo "ROCm final image cache export mode: ${rocm_cache_to_mode}"

    if [[ -n "${ROCM_CSRC_CONTENT_CACHE_REF:-}" ]]; then
        content_cache_from+=("type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF}")
        csrc_cache_to+=(
            "type=registry,ref=${ROCM_CSRC_CONTENT_CACHE_REF},mode=${csrc_cache_to_mode},ignore-error=true"
        )
    fi

    # Docker Hub cache exports are best-effort. A cache-only target failure can
    # otherwise cancel the sibling image target before its manifest is pushed.
    if [[ -n "${BUILDKITE_COMMIT:-}" ]]; then
        csrc_cache_to+=(
            "type=registry,ref=${cache_repo}:csrc-rocm-${BUILDKITE_COMMIT},mode=${csrc_cache_to_mode},ignore-error=true"
        )
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-${BUILDKITE_COMMIT},mode=${rocm_cache_to_mode},ignore-error=true"
        )
    fi

    if [[ -n "${ROCM_CACHE_BRANCH_TAG:-}" ]]; then
        csrc_cache_to+=(
            "type=registry,ref=${cache_repo}:csrc-rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${csrc_cache_to_mode},ignore-error=true"
        )
        rocm_cache_to+=(
            "type=registry,ref=${cache_repo}:rocm-branch-${ROCM_CACHE_BRANCH_TAG},mode=${rocm_cache_to_mode},ignore-error=true"
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
        write_hcl_string_list "    " "${content_cache_from[@]}"
        cat <<EOF
  )
EOF
        write_hcl_string_list_attr "  " "cache-to" "${csrc_cache_to[@]}"
        cat <<EOF
}

target "test-rocm-ci" {
  cache-from = concat(
    get_cache_from_rocm(),
EOF
        write_hcl_string_list "    " "${content_cache_from[@]}"
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
        write_hcl_string_list "    " "${content_cache_from[@]}"
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

    for var in RIXL_BRANCH UCX_BRANCH ROCSHMEM_BRANCH DEEPEP_BRANCH; do
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
    local rixl_branch=""
    local ucx_branch=""
    local rocshmem_branch=""
    local deepep_branch=""
    local rixl_material=""
    local rocshmem_material=""
    local deepep_material=""

    bake_dir=$(dirname "${VLLM_BAKE_FILE}")
    dockerfile_rocm="${bake_dir}/Dockerfile.rocm"
    rixl_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "RIXL_BRANCH")
    ucx_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "UCX_BRANCH")
    rocshmem_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "ROCSHMEM_BRANCH")
    deepep_branch=$(resolve_dockerfile_arg_value "${dockerfile_rocm}" "DEEPEP_BRANCH")

    if [[ -n "${rixl_branch}" && -n "${ucx_branch}" ]]; then
        rixl_material=$(compose_stage_cache_material "${dockerfile_rocm}" "base build_rixl")
        RIXL_CACHE_KEY=$(
            compose_dependency_cache_key \
                "${rixl_branch}-ucx-${ucx_branch}" \
                "${rixl_material}"
        )
        export RIXL_CACHE_KEY
        echo "RIXL dependency cache key: ${RIXL_CACHE_KEY}"
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
    docker buildx imagetools inspect "${cache_ref}" >/dev/null 2>&1
}

resolve_ci_base_dependency_targets() {
    local mode="${ROCM_DEP_CACHE_EXPORT_MODE:-missing}"
    local cache_repo="${DOCKERHUB_CACHE_REPO:-rocm/vllm-ci-cache}"
    local rixl_ref=""
    local rocshmem_ref=""
    local deepep_ref=""
    local -a seed_targets=()

    [[ "${TARGET}" == "ci-base-rocm-ci-with-deps" ]] || return 0

    case "${mode}" in
        always)
            echo "ROCM_DEP_CACHE_EXPORT_MODE=always; exporting all dependency caches"
            return 0
            ;;
        never)
            BAKE_TARGETS=("ci-base-rocm-ci")
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

    if [[ -n "${RIXL_CACHE_KEY:-}" ]]; then
        rixl_ref="${cache_repo}:rixl-rocm-${RIXL_CACHE_KEY}"
        if dependency_cache_ref_exists "${rixl_ref}"; then
            echo "RIXL dependency cache exists: ${rixl_ref}"
        else
            echo "RIXL dependency cache missing; will seed: ${rixl_ref}"
            seed_targets+=("rixl-rocm-ci")
        fi
    fi

    if [[ -n "${ROCSHMEM_CACHE_KEY:-}" ]]; then
        rocshmem_ref="${cache_repo}:rocshmem-rocm-${ROCSHMEM_CACHE_KEY}"
        if dependency_cache_ref_exists "${rocshmem_ref}"; then
            echo "ROCShmem dependency cache exists: ${rocshmem_ref}"
        else
            echo "ROCShmem dependency cache missing; will seed: ${rocshmem_ref}"
            seed_targets+=("rocshmem-rocm-ci")
        fi
    fi

    if [[ -n "${DEEPEP_CACHE_KEY:-}" ]]; then
        deepep_ref="${cache_repo}:deepep-rocm-${DEEPEP_CACHE_KEY}"
        if dependency_cache_ref_exists "${deepep_ref}"; then
            echo "DeepEP dependency cache exists: ${deepep_ref}"
        else
            echo "DeepEP dependency cache missing; will seed: ${deepep_ref}"
            seed_targets+=("deepep-rocm-ci")
        fi
    fi

    # DeepEP inherits from ROCShmem. If ROCShmem is being seeded, seed DeepEP too
    # so the pair stays consistent for future ci_base rebuilds.
    if printf '%s\n' "${seed_targets[@]}" | grep -qx "rocshmem-rocm-ci" \
        && ! printf '%s\n' "${seed_targets[@]}" | grep -qx "deepep-rocm-ci" \
        && [[ -n "${DEEPEP_BRANCH:-}" ]]; then
        echo "ROCShmem cache is missing; also seeding DeepEP cache"
        seed_targets+=("deepep-rocm-ci")
    fi

    BAKE_TARGETS=("${seed_targets[@]}" "ci-base-rocm-ci")
    if [[ ${#seed_targets[@]} -eq 0 ]]; then
        echo "All dependency caches exist; building ci_base without dependency cache exports"
    else
        echo "Resolved ci_base bake targets: ${BAKE_TARGETS[*]}"
    fi
}

print_bake_config() {
    echo "--- :page_facing_up: Resolved bake configuration"
    docker buildx bake "${BAKE_FILES[@]}" --print "${BAKE_TARGETS[@]}" | tee "${BAKE_CONFIG_FILE}"

    if command -v buildkite-agent >/dev/null 2>&1 && [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        buildkite-agent artifact upload "${BAKE_CONFIG_FILE}" || true
        echo "Uploaded ${BAKE_CONFIG_FILE} as Buildkite artifact"
    else
        echo "Saved bake config to ${BAKE_CONFIG_FILE} (not in Buildkite, skipping upload)"
    fi
}

confirm_remote_image_push() {
    local image_ref="$1"
    local remote_hash=""
    local remote_revision=""

    if ! remote_image_exists "${image_ref}"; then
        return 1
    fi

    if is_ci_base_target; then
        if [[ -z "${CI_BASE_CONTENT_HASH:-}" ]]; then
            return 0
        fi

        remote_hash=$(get_remote_image_label_with_retry "${image_ref}" "vllm.ci_base.content_hash")
        if [[ -n "${remote_hash}" && "${remote_hash}" == "${CI_BASE_CONTENT_HASH}" ]]; then
            return 0
        fi

        echo "Remote image exists but does not have the expected ci_base content hash."
        echo "  expected: ${CI_BASE_CONTENT_HASH:0:16}..."
        echo "  found:    ${remote_hash:0:16}..."
        return 1
    fi

    if is_commit_image_target; then
        remote_revision=$(get_remote_image_label_with_retry "${image_ref}" "org.opencontainers.image.revision")
        if [[ -n "${remote_revision}" && "${remote_revision}" == "${BUILDKITE_COMMIT}" ]]; then
            return 0
        fi

        if [[ -z "${remote_revision}" \
              && ${IMAGE_EXISTED_BEFORE_BUILD} -eq 0 \
              && image_tag_is_commit_scoped ]]; then
            echo "Remote image exists under a commit-scoped tag; accepting push despite missing revision label."
            return 0
        fi

        echo "Remote image exists but revision label does not match ${BUILDKITE_COMMIT}."
        echo "  found revision: ${remote_revision:-<missing>}"
        return 1
    fi

    return 0
}

annotate_cache_export_warning() {
    local build_rc="$1"

    if ! command -v buildkite-agent >/dev/null 2>&1; then
        return 0
    fi

    buildkite-agent annotate \
        --style warning \
        --context "cache-export-warning" \
        "### :warning: Docker cache export failed (non-fatal)

Image was pushed successfully: \`${IMAGE_TAG}\`

The BuildKit build returned exit code ${build_rc}, but the expected image
is present in the registry. Treating this as a registry cache export failure
so tests can continue with the pushed image." 2>/dev/null || true
}

run_bake() {
    local build_rc=0

    echo "--- :docker: Building ${TARGET}"
    docker buildx bake "${BAKE_FILES[@]}" --progress plain "${BAKE_TARGETS[@]}" || build_rc=$?

    if [[ ${build_rc} -eq 0 ]]; then
        echo "--- :white_check_mark: Build complete"
        return 0
    fi

    echo ""
    echo "WARNING: docker buildx bake exited with code ${build_rc}"

    if [[ -n "${IMAGE_TAG:-}" ]]; then
        echo "Checking if image was pushed successfully..."
        if confirm_remote_image_push "${IMAGE_TAG}"; then
            echo ""
            echo "WARNING: Build reported failure (rc=${build_rc}) but the"
            echo "         image was pushed successfully: ${IMAGE_TAG}"
            echo ""
            echo "         Treating this as a non-fatal registry cache export failure."
            echo "         The image is usable, but registry cache may be cold on the next build."
            echo ""
            annotate_cache_export_warning "${build_rc}"
            echo "--- :white_check_mark: Build complete"
            return 0
        fi

        echo ""
        echo "ERROR: Build failed and image was NOT confirmed: ${IMAGE_TAG}"
        echo "       This is a real build failure, not a cache export warning."
        echo ""
    fi

    return "${build_rc}"
}

upload_wheel_artifacts_if_present() {
    local wheel_dir="./wheel-export"
    local artifact_dir="artifacts/vllm-rocm-install"
    local archive_name="vllm-rocm-install.tar.gz"
    local whl=""
    local whl_name=""

    if ! should_upload_wheel_artifacts; then
        return 0
    fi

    if [[ ! -d "${wheel_dir}" ]] || ! ls "${wheel_dir}"/*.whl >/dev/null 2>&1; then
        echo "No ROCm wheel artifacts found in ${wheel_dir}"
        return 0
    fi

    echo "--- :package: Uploading ROCm vLLM install artifact"
    mkdir -p "${artifact_dir}"

    tar -C "${wheel_dir}" -czf "${artifact_dir}/${archive_name}" .
    echo "Created ${archive_name}: $(du -sh "${artifact_dir}/${archive_name}" | cut -f1)"
    printf '%s\n' "${CI_BASE_IMAGE:-}" > "${artifact_dir}/ci-base-image.txt"
    printf '%s\n' "${IMAGE_TAG:-}" > "${artifact_dir}/fallback-image.txt"

    for whl in "${wheel_dir}"/*.whl; do
        [[ -f "${whl}" ]] || continue
        whl_name=$(basename "${whl}")
        cp "${whl}" "${artifact_dir}/${whl_name}"
        echo "Copied ${whl_name}: $(du -sh "${artifact_dir}/${whl_name}" | cut -f1)"
    done

    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent artifact upload "${artifact_dir}/*"
        echo "ROCm vLLM install artifacts uploaded to ${artifact_dir}/"
    else
        echo "Not in Buildkite, skipping artifact upload"
    fi

    rm -rf "${wheel_dir}"
}

main() {
    init_config "$@"
    print_header
    validate_inputs
    load_ci_hcl
    compute_ci_base_hash_if_needed
    configure_ci_base_image_refs
    maybe_skip_existing_image
    setup_builder
    prepare_git_cache_metadata
    write_ci_base_label_override
    extract_dependency_pins
    write_rocm_build_arg_override
    compute_dependency_cache_keys
    compute_rocm_csrc_content_hash_if_needed
    write_rocm_cache_override
    resolve_ci_base_dependency_targets
    print_bake_config
    if [[ "${BAKE_PRINT_ONLY:-0}" == "1" ]]; then
        echo "BAKE_PRINT_ONLY=1 set; skipping build"
        return 0
    fi
    run_bake
    upload_wheel_artifacts_if_present
}

main "$@"
