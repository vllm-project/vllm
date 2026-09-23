#!/usr/bin/env bash
# Shared Dockerfile selection for the normal AMD CI build chain.
# CI_ROCM_DOCKERFILE_BASE and CI_ROCM_DOCKERFILE accept repository-relative paths.
# For Rock, set CI_ROCM_DOCKERFILE_BASE=docker/Dockerfile.rock_base and
# CI_ROCM_DOCKERFILE=docker/Dockerfile.rock.
# Custom recipes must preserve the CI targets and artifact layout. Additional
# COPY inputs also need the corresponding *_CONTENT_FILES overrides.

rocm_base_dockerfile() {
    local default="docker/Dockerfile.rocm_base"
    printf '%s\n' "${CI_ROCM_DOCKERFILE_BASE:-${ROCM_BASE_DOCKERFILE:-${default}}}"
}

rocm_ci_dockerfile() {
    local default="docker/Dockerfile.rocm"
    printf '%s\n' "${CI_ROCM_DOCKERFILE:-${CI_BASE_DOCKERFILE:-${default}}}"
}

using_custom_rocm_dockerfiles() {
    [[ "$(rocm_base_dockerfile)" != "docker/Dockerfile.rocm_base" \
        || "$(rocm_ci_dockerfile)" != "docker/Dockerfile.rocm" ]]
}

rocm_dockerfile_stages() {
    awk '
        toupper($1) == "FROM" {
            for (idx = 2; idx < NF; idx++) {
                if (tolower($idx) == "as") print tolower($(idx + 1))
            }
        }
    ' "$1"
}

validate_rocm_dockerfile() {
    local dockerfile="$1"
    local stage=""
    local stages=""
    shift

    case "${dockerfile}" in
        /*|../*|*/../*|*/..|*[[:space:]]*)
            echo "ROCm Dockerfile must be a repository-relative path without '..' or whitespace: ${dockerfile}" >&2
            return 2
            ;;
    esac
    if [[ ! -f "${dockerfile}" ]]; then
        echo "ROCm Dockerfile not found: ${dockerfile}" >&2
        return 2
    fi
    if (($#)) && ! awk '
        toupper($1) == "FROM" {
            named = 0
            for (idx = 2; idx < NF; idx++) {
                if (tolower($idx) == "as") named = 1
            }
            if (!named) exit 1
        }
    ' "${dockerfile}"; then
        echo "ROCm Dockerfile must name every build stage: ${dockerfile}" >&2
        return 2
    fi
    stages=$(rocm_dockerfile_stages "${dockerfile}")
    for stage in "$@"; do
        if ! grep -Fxq -- "${stage}" <<< "${stages}"; then
            echo "ROCm Dockerfile ${dockerfile} is missing required stage: ${stage}" >&2
            return 2
        fi
    done
}

configure_rocm_build() {
    ROCM_BASE_DOCKERFILE=$(rocm_base_dockerfile)
    CI_BASE_DOCKERFILE=$(rocm_ci_dockerfile)
    export ROCM_BASE_DOCKERFILE CI_BASE_DOCKERFILE
    using_custom_rocm_dockerfiles || return 0

    # Custom stacks may supply dependencies without separate build stages.
    export ROCM_DEP_CACHE_EXPORT_MODE=never

    # Preserve the standard runtime tags, which are unique to each build,
    # while preventing experimental builds from promoting stable images.
    export ROCM_BASE_PUSH_STABLE_TAG=0
    export CI_BASE_PUSH_STABLE_TAG=0
    echo "Custom ROCm Dockerfiles: ${ROCM_BASE_DOCKERFILE} and ${CI_BASE_DOCKERFILE}"
}
