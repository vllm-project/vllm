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
IMAGE_EXISTED_BEFORE_BUILD=0

TARGET=""
CI_HCL_PATH=""
CI_BASE_LABEL_OVERRIDE_PATH=""
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

compute_ci_base_content_hash() {
    local -a content_paths=()
    read -r -a content_paths <<< "${CI_BASE_CONTENT_FILES}"
    compute_content_hash "${content_paths[@]}"
}

is_ci_base_target() {
    [[ "${TARGET}" == *"ci-base-rocm"* || -n "${CI_BASE_CONTENT_FILES:-}" ]]
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
    CI_HCL_URL="${CI_HCL_URL:-https://raw.githubusercontent.com/vllm-project/ci-infra/main/docker/ci-rocm.hcl}"
    VLLM_BAKE_FILE="${VLLM_BAKE_FILE:-docker/docker-bake-rocm.hcl}"
    BUILDER_NAME="${BUILDER_NAME:-vllm-builder}"
    BUILDKIT_SOCKET="${BUILDKIT_SOCKET:-/run/buildkit/buildkitd.sock}"
    PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a;gfx942;gfx950}"
    export PYTORCH_ROCM_ARCH

    SCRIPT_TMP_DIR=$(mktemp -d -t ci-bake-rocm.XXXXXX)
    CI_HCL_PATH="${SCRIPT_TMP_DIR}/ci.hcl"
    CI_BASE_LABEL_OVERRIDE_PATH="${SCRIPT_TMP_DIR}/ci-base-label-override.hcl"
    BAKE_CONFIG_FILE="bake-config-build-${BUILDKITE_BUILD_NUMBER:-local}.json"
}

print_header() {
    echo "--- :docker: Setting up Docker buildx bake"
    echo "Target: ${TARGET}"
    echo "CI HCL URL: ${CI_HCL_URL}"
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

    if is_ci_base_target && [[ -z "${CI_BASE_CONTENT_FILES:-}" ]]; then
        echo "Warning: ci_base target has no CI_BASE_CONTENT_FILES configured"
        echo "         Existing ci_base images will be treated as reusable by tag."
    fi
}

download_ci_hcl() {
    echo "--- :arrow_down: Downloading ci.hcl"
    curl -sSfL -o "${CI_HCL_PATH}" "${CI_HCL_URL}"
    echo "Downloaded to ${CI_HCL_PATH}"
}

compute_ci_base_hash_if_needed() {
    if [[ -z "${CI_BASE_CONTENT_FILES:-}" ]]; then
        return 0
    fi

    CI_BASE_CONTENT_HASH=$(compute_ci_base_content_hash)
    export CI_BASE_CONTENT_HASH
    echo "ci_base content hash: ${CI_BASE_CONTENT_HASH:0:16}..."
}

maybe_skip_existing_image() {
    local remote_hash=""
    local remote_revision=""

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

    if git rev-parse --is-shallow-repository 2>/dev/null | grep -q "true"; then
        echo "Shallow clone detected - deepening for cache key computation"
        git fetch --deepen=1 origin 2>/dev/null || true
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
        local source_repo_slug=""
        local target_repo_slug=""
        source_repo_slug=$(get_buildkite_repo_slug)
        target_repo_slug=$(get_buildkite_target_repo_slug)
        if [[ "${source_repo_slug}" != "${target_repo_slug}" ]]; then
            ROCM_CACHE_UPSTREAM_BRANCH_TAG=$(
                compose_cache_branch_tag "${target_repo_slug}" "${BUILDKITE_PULL_REQUEST_BASE_BRANCH}"
            )
            export ROCM_CACHE_UPSTREAM_BRANCH_TAG
            echo "Computed ROCm upstream branch cache tag: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
        fi
    elif [[ -n "${ROCM_CACHE_UPSTREAM_BRANCH_TAG:-}" ]]; then
        echo "Using provided ROCM_CACHE_UPSTREAM_BRANCH_TAG: ${ROCM_CACHE_UPSTREAM_BRANCH_TAG}"
    fi

    if [[ -z "${VLLM_MERGE_BASE_COMMIT:-}" ]]; then
        git fetch --depth=1 origin main 2>/dev/null || true
        VLLM_MERGE_BASE_COMMIT=$(git merge-base HEAD origin/main 2>/dev/null || echo "")
        if [[ -n "${VLLM_MERGE_BASE_COMMIT}" ]]; then
            export VLLM_MERGE_BASE_COMMIT
            echo "Computed merge base commit for cache fallback: ${VLLM_MERGE_BASE_COMMIT}"
        else
            echo "Could not determine merge base"
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

    if [[ -n "${RIXL_BRANCH:-}" && -n "${UCX_BRANCH:-}" ]]; then
        rixl_ref="${cache_repo}:rixl-rocm-${RIXL_BRANCH}-ucx-${UCX_BRANCH}"
        if dependency_cache_ref_exists "${rixl_ref}"; then
            echo "RIXL dependency cache exists: ${rixl_ref}"
        else
            echo "RIXL dependency cache missing; will seed: ${rixl_ref}"
            seed_targets+=("rixl-rocm-ci")
        fi
    fi

    if [[ -n "${ROCSHMEM_BRANCH:-}" ]]; then
        rocshmem_ref="${cache_repo}:rocshmem-rocm-${ROCSHMEM_BRANCH}"
        if dependency_cache_ref_exists "${rocshmem_ref}"; then
            echo "ROCShmem dependency cache exists: ${rocshmem_ref}"
        else
            echo "ROCShmem dependency cache missing; will seed: ${rocshmem_ref}"
            seed_targets+=("rocshmem-rocm-ci")
        fi
    fi

    if [[ -n "${DEEPEP_BRANCH:-}" && -n "${ROCSHMEM_BRANCH:-}" ]]; then
        deepep_ref="${cache_repo}:deepep-rocm-${DEEPEP_BRANCH}-rocshmem-${ROCSHMEM_BRANCH}"
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
    download_ci_hcl
    compute_ci_base_hash_if_needed
    maybe_skip_existing_image
    setup_builder
    prepare_git_cache_metadata
    write_ci_base_label_override
    extract_dependency_pins
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
