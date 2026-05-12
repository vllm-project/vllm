#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

default_python_bin() {
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        printf '%s\n' "${VIRTUAL_ENV}/bin/python"
        return 0
    fi

    if [[ -x ".venv/bin/python" ]]; then
        printf '%s\n' ".venv/bin/python"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi

    printf '%s\n' "python"
}

PYTHON_BIN="${PYTHON_BIN:-$(default_python_bin)}"
INSTALL_LOG="${INSTALL_LOG:-install.log}"
AUTO_INSTALL_CUDA_DEV_HEADERS="${AUTO_INSTALL_CUDA_DEV_HEADERS:-1}"
AUTO_INSTALL_SYSTEM_DEPS="${AUTO_INSTALL_SYSTEM_DEPS:-1}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-${TORCH_BACKEND:-auto}}"

append_cmake_arg() {
    local arg="$1"
    if [[ " ${CMAKE_ARGS:-} " != *" ${arg} "* ]]; then
        export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }${arg}"
    fi
}

prepend_env_path() {
    local var_name="$1"
    local path="$2"
    local current="${!var_name:-}"

    case ":${current}:" in
        *":${path}:"*) ;;
        *)
            export "${var_name}=${path}${current:+:${current}}"
            ;;
    esac
}

ensure_uv() {
    local home_dir="${HOME:-/root}"
    local -a prefix=()

    if command -v uv >/dev/null 2>&1; then
        return 0
    fi

    if [[ -x "${home_dir}/.local/bin/uv" ]]; then
        prepend_env_path PATH "${home_dir}/.local/bin"
        return 0
    fi

    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        prefix=(sudo)
    fi

    if ! command -v curl >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1 && { [[ "$(id -u)" -eq 0 ]] || [[ ${#prefix[@]} -gt 0 ]]; }; then
        echo "curl is missing; attempting to install curl so uv can be bootstrapped." >&2
        if ! "${prefix[@]}" apt-get update; then
            echo "warning: failed to refresh apt package metadata before installing curl." >&2
        elif ! "${prefix[@]}" apt-get install -y ca-certificates curl; then
            echo "warning: failed to install curl for uv bootstrap." >&2
        fi
    fi

    if command -v curl >/dev/null 2>&1; then
        echo "uv was not found; installing uv into the current user environment." >&2
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            prepend_env_path PATH "${home_dir}/.local/bin"
        fi
    fi

    command -v uv >/dev/null 2>&1
}

pip_install() {
    local -a uv_args=()

    if ensure_uv; then
        uv_args=(pip install --python "${PYTHON_BIN}")
        if uv pip install --help 2>/dev/null | grep -q -- "--torch-backend"; then
            uv_args+=(--torch-backend="${UV_TORCH_BACKEND}")
        else
            echo "warning: this uv version does not support --torch-backend; install may select the wrong torch wheel." >&2
        fi
        uv "${uv_args[@]}" "$@"
        return $?
    fi

    echo "error: uv is required for this installer so PyTorch backend selection is correct." >&2
    echo "Install uv manually, then rerun: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
}

run_logged() {
    local mode="$1"
    shift

    if [[ "${mode}" == "append" ]]; then
        "$@" 2>&1 | tee -a "${INSTALL_LOG}"
    else
        "$@" 2>&1 | tee "${INSTALL_LOG}"
    fi
    return "${PIPESTATUS[0]}"
}

join_by() {
    local delimiter="$1"
    shift || true
    local first=1
    local value

    for value in "$@"; do
        if (( first )); then
            printf '%s' "${value}"
            first=0
        else
            printf '%s%s' "${delimiter}" "${value}"
        fi
    done
}

find_default_cuda_roots() {
    local candidate

    for candidate in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*; do
        if [[ -d "${candidate}" ]]; then
            (cd -- "${candidate}" && pwd)
        fi
    done
}

find_cuda_target_subdirs() {
    local cuda_root="$1"
    local suffix="$2"
    local target_dir

    for target_dir in "${cuda_root}"/targets/*; do
        if [[ -d "${target_dir}/${suffix}" ]]; then
            (cd -- "${target_dir}/${suffix}" && pwd)
        fi
    done
}

resolve_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        printf '%s\n' "${CUDA_HOME}"
        return 0
    fi

    if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}" ]]; then
        printf '%s\n' "${CUDA_PATH}"
        return 0
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        printf '%s\n' "$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
        return 0
    fi

    return 1
}

find_cuda_header_dirs_by_scan() {
    local root
    local -A seen=()
    local -a headers=(
        "cublas_v2.h"
        "cusparse.h"
        "cusolverDn.h"
        "cusolver_common.h"
        "nvrtc.h"
    )
    local header
    local match
    local parent

    for root in "$@"; do
        if [[ -z "${root}" || ! -d "${root}" ]]; then
            continue
        fi

        for header in "${headers[@]}"; do
            while IFS= read -r match; do
                if [[ -z "${match}" ]]; then
                    continue
                fi

                parent="$(cd -- "$(dirname -- "${match}")" && pwd)"
                if [[ -z "${seen[${parent}]:-}" ]]; then
                    seen["${parent}"]=1
                    printf '%s\n' "${parent}"
                fi
            done < <(find "${root}" -maxdepth 6 -type f -name "${header}" 2>/dev/null)
        done
    done
}

find_nvrtc() {
    if [[ -n "${CUDA_NVRTC_LIBRARY:-}" && -f "${CUDA_NVRTC_LIBRARY}" ]]; then
        printf '%s\n' "${CUDA_NVRTC_LIBRARY}"
        return 0
    fi

    local cuda_home="${1:-}"
    local candidates=()
    local cuda_root
    local multiarch_lib_dir
    local target_lib_dir

    if [[ -n "${cuda_home}" ]]; then
        candidates+=(
            "${cuda_home}/lib64/libnvrtc.so"
            "${cuda_home}/lib64/libnvrtc.so."*
        )

        while IFS= read -r target_lib_dir; do
            candidates+=(
                "${target_lib_dir}/libnvrtc.so"
                "${target_lib_dir}/libnvrtc.so."*
            )
        done < <(find_cuda_target_subdirs "${cuda_home}" "lib")
    fi

    candidates+=(
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so"
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so."*
        "/usr/local/cuda/lib64/libnvrtc.so"
        "/usr/local/cuda/lib64/libnvrtc.so."*
    )

    for multiarch_lib_dir in /usr/lib/*-linux-gnu; do
        if [[ -d "${multiarch_lib_dir}" ]]; then
            candidates+=(
                "${multiarch_lib_dir}/libnvrtc.so"
                "${multiarch_lib_dir}/libnvrtc.so."*
            )
        fi
    done

    for cuda_root in $(find_default_cuda_roots); do
        while IFS= read -r target_lib_dir; do
            candidates+=(
                "${target_lib_dir}/libnvrtc.so"
                "${target_lib_dir}/libnvrtc.so."*
            )
        done < <(find_cuda_target_subdirs "${cuda_root}" "lib")
    done

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if command -v ldconfig >/dev/null 2>&1; then
        local ldconfig_match
        ldconfig_match="$(ldconfig -p 2>/dev/null | awk '/libnvrtc\.so/ { print $NF; exit }')"
        if [[ -n "${ldconfig_match}" && -f "${ldconfig_match}" ]]; then
            printf '%s\n' "${ldconfig_match}"
            return 0
        fi
    fi

    return 1
}

find_python_cuda_include_dirs() {
    "${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
from pathlib import Path
import site
import sysconfig

roots = []
seen_roots = set()

def add_root(path_str):
    if not path_str:
        return
    path = Path(path_str)
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            roots.append(path)

for site_path in getattr(site, "getsitepackages", lambda: [])():
    add_root(site_path)

add_root(getattr(site, "getusersitepackages", lambda: None)())
add_root(sysconfig.get_path("purelib"))
add_root(sysconfig.get_path("platlib"))

seen_dirs = set()
for root in roots:
    for rel_path in (
        "nvidia/cuda_runtime/include",
        "nvidia/cublas/include",
        "nvidia/cusparse/include",
        "nvidia/cusolver/include",
        "nvidia/cuda_nvrtc/include",
    ):
        include_dir = root / rel_path
        if include_dir.is_dir():
            resolved = str(include_dir.resolve())
            if resolved not in seen_dirs:
                seen_dirs.add(resolved)
                print(resolved)
PY
}

find_cuda_include_dirs() {
    local cuda_home="${1:-}"
    local candidates=()
    local scan_roots=()
    local candidate
    local path_var
    local value
    local -A seen=()
    local -a found=()
    local -a include_dirs=()
    local cuda_root
    local has_cublas=0
    local has_cusparse=0
    local has_cusolver=0
    local has_nvrtc_headers=0

    if [[ -n "${CUDA_INCLUDE_DIR:-}" && -d "${CUDA_INCLUDE_DIR}" ]]; then
        candidates+=("${CUDA_INCLUDE_DIR}")
        scan_roots+=("${CUDA_INCLUDE_DIR}")
    fi

    if [[ -n "${cuda_home}" ]]; then
        candidates+=("${cuda_home}/include")
        scan_roots+=("${cuda_home}")
        while IFS= read -r candidate; do
            candidates+=("${candidate}")
        done < <(find_cuda_target_subdirs "${cuda_home}" "include")
    fi

    candidates+=(
        "/usr/local/cuda/include"
        "/usr/include"
        "/usr/include/x86_64-linux-gnu"
    )

    for cuda_root in $(find_default_cuda_roots); do
        candidates+=("${cuda_root}/include")
        scan_roots+=("${cuda_root}")
        while IFS= read -r candidate; do
            candidates+=("${candidate}")
        done < <(find_cuda_target_subdirs "${cuda_root}" "include")
    done

    for candidate in /usr/include/*-linux-gnu; do
        if [[ -d "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
    done

    for path_var in CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH INCLUDE; do
        value="${!path_var:-}"
        if [[ -n "${value}" ]]; then
            IFS=':' read -r -a include_dirs <<< "${value}"
            candidates+=("${include_dirs[@]}")
        fi
    done

    while IFS= read -r candidate; do
        if [[ -n "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
    done < <(find_python_cuda_include_dirs)

    while IFS= read -r candidate; do
        if [[ -n "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
    done < <(find_cuda_header_dirs_by_scan "${scan_roots[@]}")

    for candidate in "${candidates[@]}"; do
        if [[ -z "${candidate}" || ! -d "${candidate}" ]]; then
            continue
        fi

        candidate="$(cd -- "${candidate}" && pwd)"
        if [[ -n "${seen[${candidate}]:-}" ]]; then
            continue
        fi

        if [[ -f "${candidate}/cublas_v2.h" || -f "${candidate}/cusparse.h" || -f "${candidate}/cusolverDn.h" || -f "${candidate}/cusolver_common.h" || -f "${candidate}/nvrtc.h" ]]; then
            seen["${candidate}"]=1
            found+=("${candidate}")
            if [[ -f "${candidate}/cublas_v2.h" ]]; then
                has_cublas=1
            fi
            if [[ -f "${candidate}/cusparse.h" ]]; then
                has_cusparse=1
            fi
            if [[ -f "${candidate}/cusolverDn.h" || -f "${candidate}/cusolver_common.h" ]]; then
                has_cusolver=1
            fi
            if [[ -f "${candidate}/nvrtc.h" ]]; then
                has_nvrtc_headers=1
            fi
        fi
    done

    if (( has_cublas && has_cusparse && has_cusolver && has_nvrtc_headers )); then
        printf '%s\n' "${found[@]}"
        return 0
    fi

    return 1
}

describe_missing_cuda_headers() {
    local include_dir
    local missing=()
    local has_cublas=0
    local has_cusparse=0
    local has_cusolver=0
    local has_nvrtc_headers=0

    for include_dir in "$@"; do
        if [[ -f "${include_dir}/cublas_v2.h" ]]; then
            has_cublas=1
        fi
        if [[ -f "${include_dir}/cusparse.h" ]]; then
            has_cusparse=1
        fi
        if [[ -f "${include_dir}/cusolverDn.h" || -f "${include_dir}/cusolver_common.h" ]]; then
            has_cusolver=1
        fi
        if [[ -f "${include_dir}/nvrtc.h" ]]; then
            has_nvrtc_headers=1
        fi
    done

    if (( ! has_cublas )); then
        missing+=("cublas_v2.h")
    fi
    if (( ! has_cusparse )); then
        missing+=("cusparse.h")
    fi
    if (( ! has_cusolver )); then
        missing+=("cusolverDn.h")
    fi
    if (( ! has_nvrtc_headers )); then
        missing+=("nvrtc.h")
    fi

    join_by ", " "${missing[@]}"
}

extract_cuda_major_minor() {
    local value="$1"
    local version=""

    version="$(printf '%s\n' "${value}" | sed -nE 's/^[^0-9]*([0-9]+)[.-]([0-9]+).*/\1.\2/p' | head -n1)"
    if [[ -n "${version}" ]]; then
        printf '%s\n' "${version}"
        return 0
    fi

    return 1
}

is_plausible_cuda_version() {
    local version="$1"
    local major="${version%%.*}"

    [[ "${major}" =~ ^[0-9]+$ && "${major}" -ge 10 ]]
}

is_valid_python_package_version() {
    local version="$1"

    "${PYTHON_BIN}" - "${version}" <<'PY' >/dev/null 2>&1
from packaging.version import Version, InvalidVersion
import sys

try:
    Version(sys.argv[1])
except InvalidVersion:
    raise SystemExit(1)
PY
}

read_python_assigned_version() {
    local file="$1"
    local name="$2"

    if [[ ! -f "${file}" ]]; then
        return 1
    fi

    "${PYTHON_BIN}" - "${file}" "${name}" <<'PY' 2>/dev/null
import ast
import sys

path, name = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=path)

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if not any(isinstance(target, ast.Name) and target.id == name
               for target in node.targets):
        continue
    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        print(node.value.value)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

read_vllm_version_py_fallback() {
    local file="${ROOT_DIR}/vllm/version.py"

    if [[ ! -f "${file}" ]]; then
        return 1
    fi

    "${PYTHON_BIN}" - "${file}" <<'PY' 2>/dev/null
import ast
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=path)

assignments = {}
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    for target in node.targets:
        if isinstance(target, ast.Name):
            assignments[target.id] = node.value

version_node = assignments.get("__version__")
if not (isinstance(version_node, ast.Constant)
        and isinstance(version_node.value, str)
        and version_node.value == "dev"):
    raise SystemExit(1)

tuple_node = assignments.get("__version_tuple__")
if not isinstance(tuple_node, ast.Tuple) or len(tuple_node.elts) < 3:
    raise SystemExit(1)

major_node, minor_node, dev_node = tuple_node.elts[:3]
if not (isinstance(major_node, ast.Constant)
        and isinstance(major_node.value, int)
        and isinstance(minor_node, ast.Constant)
        and isinstance(minor_node.value, int)
        and isinstance(dev_node, ast.Name)
        and dev_node.id == "__version__"):
    raise SystemExit(1)

print(f"{major_node.value}.{minor_node.value}.dev0")
PY
}

infer_vllm_version_override() {
    local candidate=""

    for version_file in "${ROOT_DIR}/vllm/_version.py" "${ROOT_DIR}/vllm/version.py"; do
        if candidate="$(read_python_assigned_version "${version_file}" "__version__")"; then
            if [[ "${candidate}" != "dev" ]] && is_valid_python_package_version "${candidate}"; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        fi
    done

    if candidate="$(read_vllm_version_py_fallback)"; then
        if is_valid_python_package_version "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    candidate="$(sed -nE 's/^[[:space:]]*"version":[[:space:]]*"([^"]+)".*/\1/p' "${ROOT_DIR}/setup.py" | head -n1)"
    if [[ -n "${candidate}" ]] && is_valid_python_package_version "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    printf '%s\n' "0.0.dev0"
}

detect_cuda_version() {
    local cuda_home="${1:-}"
    local cuda_root
    local nvcc_output=""
    local version=""

    if [[ -n "${CUDA_VERSION:-}" ]]; then
        if version="$(extract_cuda_major_minor "${CUDA_VERSION}")"; then
            if is_plausible_cuda_version "${version}"; then
                printf '%s\n' "${version}"
                return 0
            fi
        fi
    fi

    if command -v nvcc >/dev/null 2>&1; then
        nvcc_output="$(nvcc --version 2>/dev/null || true)"
        if version="$(extract_cuda_major_minor "${nvcc_output}")"; then
            if is_plausible_cuda_version "${version}"; then
                printf '%s\n' "${version}"
                return 0
            fi
        fi
    fi

    if [[ -n "${cuda_home}" ]]; then
        if version="$(extract_cuda_major_minor "$(basename "${cuda_home}")")"; then
            if is_plausible_cuda_version "${version}"; then
                printf '%s\n' "${version}"
                return 0
            fi
        fi
    fi

    for cuda_root in $(find_default_cuda_roots); do
        if version="$(extract_cuda_major_minor "$(basename "${cuda_root}")")"; then
            if is_plausible_cuda_version "${version}"; then
                printf '%s\n' "${version}"
                return 0
            fi
        fi

        version="$(basename "${cuda_root}" | sed -nE 's/^cuda-([0-9]+)$/\1.0/p')"
        if [[ -n "${version}" ]] && is_plausible_cuda_version "${version}"; then
            printf '%s\n' "${version}"
            return 0
        fi
    done

    return 1
}

detect_package_manager() {
    local manager

    for manager in apt-get dnf yum zypper microdnf; do
        if command -v "${manager}" >/dev/null 2>&1; then
            printf '%s\n' "${manager}"
            return 0
        fi
    done

    return 1
}

install_system_build_dependencies() {
    local package_manager=""
    local -a prefix=()

    if [[ "${AUTO_INSTALL_SYSTEM_DEPS}" =~ ^(0|false)$ ]]; then
        return 0
    fi

    if command -v git >/dev/null 2>&1 && command -v c++ >/dev/null 2>&1; then
        return 0
    fi

    if ! package_manager="$(detect_package_manager)"; then
        echo "warning: no supported package manager found for automatic system build dependency installation." >&2
        return 0
    fi

    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            prefix=(sudo)
        else
            echo "warning: installing system build dependencies requires root or sudo." >&2
            return 0
        fi
    fi

    echo "Installing missing system build dependencies via ${package_manager}." >&2
    case "${package_manager}" in
        apt-get)
            if ! "${prefix[@]}" apt-get update; then
                echo "warning: failed to refresh apt package metadata for system build dependencies." >&2
                return 0
            fi
            "${prefix[@]}" apt-get install -y git build-essential ca-certificates curl || true
            ;;
        dnf)
            "${prefix[@]}" dnf install -y git gcc gcc-c++ make ca-certificates curl || true
            ;;
        microdnf)
            "${prefix[@]}" microdnf install -y git gcc gcc-c++ make ca-certificates curl || true
            ;;
        yum)
            "${prefix[@]}" yum install -y git gcc gcc-c++ make ca-certificates curl || true
            ;;
        zypper)
            "${prefix[@]}" zypper --non-interactive install git gcc gcc-c++ make ca-certificates curl || true
            ;;
        *)
            echo "warning: unsupported package manager for automatic system build dependency installation: ${package_manager}" >&2
            ;;
    esac
}

configure_git_for_repo() {
    if ! command -v git >/dev/null 2>&1; then
        return 1
    fi

    if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi

    if git -C "${ROOT_DIR}" status >/dev/null 2>&1; then
        return 0
    fi

    echo "Configuring git safe.directory for ${ROOT_DIR}." >&2
    git config --global --add safe.directory "${ROOT_DIR}" || true

    git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

suggest_cuda_toolkit_install_command() {
    local package_manager="$1"
    shift
    local packages=("$@")
    local prefix="sudo "

    if [[ "$(id -u)" -eq 0 ]]; then
        prefix=""
    fi

    case "${package_manager}" in
        apt-get)
            printf '%s\n' "${prefix}apt-get update && ${prefix}apt-get install -y ${packages[*]}"
            ;;
        dnf|microdnf|yum)
            printf '%s\n' "${prefix}${package_manager} install -y ${packages[*]}"
            ;;
        zypper)
            printf '%s\n' "${prefix}zypper --non-interactive install ${packages[*]}"
            ;;
        *)
            return 1
            ;;
    esac
}

apt_cuda_repo_id() {
    local os_id=""
    local version_id=""
    local arch=""

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        os_id="${ID:-}"
        version_id="${VERSION_ID:-}"
    fi

    arch="$(dpkg --print-architecture 2>/dev/null || true)"
    case "${arch}" in
        amd64)
            arch="x86_64"
            ;;
        arm64)
            arch="sbsa"
            ;;
        ppc64el)
            arch="ppc64le"
            ;;
    esac

    case "${os_id}" in
        ubuntu)
            printf 'ubuntu%s/%s\n' "${version_id//./}" "${arch}"
            ;;
        debian)
            printf 'debian%s/%s\n' "${version_id%%.*}" "${arch}"
            ;;
        *)
            return 1
            ;;
    esac
}

ensure_cuda_apt_repo() {
    local repo_id=""
    local keyring_deb="/tmp/cuda-keyring.deb"
    local keyring_url=""
    local -a prefix=()

    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            prefix=(sudo)
        else
            echo "warning: CUDA apt repository bootstrap requires root or sudo." >&2
            return 1
        fi
    fi

    if apt-cache policy 2>/dev/null | grep -qi 'developer\.download\.nvidia\.com/compute/cuda'; then
        return 0
    fi

    if ! repo_id="$(apt_cuda_repo_id)"; then
        echo "warning: could not infer the NVIDIA CUDA apt repository for this OS." >&2
        return 1
    fi

    keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/cuda-keyring_1.1-1_all.deb"
    echo "NVIDIA CUDA apt repository is not configured; attempting to install cuda-keyring from ${keyring_url}." >&2

    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        echo "curl and wget are missing; attempting to install curl for CUDA repo bootstrap." >&2
        if ! "${prefix[@]}" apt-get update; then
            echo "warning: failed to refresh apt package metadata before installing curl." >&2
            return 1
        fi
        if ! "${prefix[@]}" apt-get install -y ca-certificates curl; then
            echo "warning: failed to install curl for CUDA repo bootstrap." >&2
            return 1
        fi
    fi

    if command -v curl >/dev/null 2>&1; then
        if ! curl -fsSL "${keyring_url}" -o "${keyring_deb}"; then
            echo "warning: failed to download CUDA apt keyring with curl." >&2
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget -qO "${keyring_deb}" "${keyring_url}"; then
            echo "warning: failed to download CUDA apt keyring with wget." >&2
            return 1
        fi
    else
        echo "warning: curl or wget is required to bootstrap the NVIDIA CUDA apt repository." >&2
        return 1
    fi

    if ! "${prefix[@]}" dpkg -i "${keyring_deb}"; then
        echo "warning: failed to install CUDA apt keyring package." >&2
        return 1
    fi

    return 0
}

install_cuda_dev_headers() {
    local cuda_home="${1:-}"
    local cuda_version=""
    local cuda_version_pkg=""
    local package_manager=""
    local toolkit_package=""
    local suggested_command=""
    local -a prefix=()
    local -a dev_packages=()

    if ! cuda_version="$(detect_cuda_version "${cuda_home}")"; then
        echo "warning: could not determine the CUDA version needed to auto-install dev headers." >&2
        return 1
    fi
    cuda_version_pkg="${cuda_version//./-}"

    if ! package_manager="$(detect_package_manager)"; then
        echo "warning: no supported package manager found for automatic CUDA toolkit installation." >&2
        return 1
    fi

    toolkit_package="cuda-toolkit-${cuda_version_pkg}"
    dev_packages=(
        "cuda-cudart-dev-${cuda_version_pkg}"
        "cuda-driver-dev-${cuda_version_pkg}"
        "cuda-nvrtc-dev-${cuda_version_pkg}"
        "libcublas-dev-${cuda_version_pkg}"
        "libcusolver-dev-${cuda_version_pkg}"
        "libcusparse-dev-${cuda_version_pkg}"
    )
    suggested_command="$(suggest_cuda_toolkit_install_command "${package_manager}" "${dev_packages[@]}" || true)"

    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            prefix=(sudo)
        else
            echo "warning: automatic CUDA toolkit installation requires root or sudo." >&2
            if [[ -n "${suggested_command}" ]]; then
                echo "Run this manually: ${suggested_command}" >&2
            fi
            return 1
        fi
    fi

    echo "CUDA dev headers are missing; attempting to install CUDA ${cuda_version} development packages via ${package_manager}." >&2
    case "${package_manager}" in
        apt-get)
            ensure_cuda_apt_repo || true
            if ! "${prefix[@]}" apt-get update; then
                echo "warning: failed to refresh apt package metadata." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: ${suggested_command}" >&2
                fi
                return 1
            fi
            if "${prefix[@]}" apt-get install -y "${dev_packages[@]}"; then
                return 0
            fi
            echo "warning: failed to install minimal CUDA dev packages via apt-get; trying ${toolkit_package}." >&2
            if ! "${prefix[@]}" apt-get install -y "${toolkit_package}"; then
                echo "warning: failed to install CUDA ${cuda_version} dev headers via apt-get." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: ${suggested_command}" >&2
                fi
                return 1
            fi
            ;;
        dnf)
            if ! "${prefix[@]}" dnf install -y "${toolkit_package}"; then
                echo "warning: failed to install ${toolkit_package} via dnf." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: $(suggest_cuda_toolkit_install_command "${package_manager}" "${toolkit_package}" || true)" >&2
                fi
                return 1
            fi
            ;;
        microdnf)
            if ! "${prefix[@]}" microdnf install -y "${toolkit_package}"; then
                echo "warning: failed to install ${toolkit_package} via microdnf." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: $(suggest_cuda_toolkit_install_command "${package_manager}" "${toolkit_package}" || true)" >&2
                fi
                return 1
            fi
            ;;
        yum)
            if ! "${prefix[@]}" yum install -y "${toolkit_package}"; then
                echo "warning: failed to install ${toolkit_package} via yum." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: $(suggest_cuda_toolkit_install_command "${package_manager}" "${toolkit_package}" || true)" >&2
                fi
                return 1
            fi
            ;;
        zypper)
            if ! "${prefix[@]}" zypper --non-interactive install "${toolkit_package}"; then
                echo "warning: failed to install ${toolkit_package} via zypper." >&2
                if [[ -n "${suggested_command}" ]]; then
                    echo "Run this manually: $(suggest_cuda_toolkit_install_command "${package_manager}" "${toolkit_package}" || true)" >&2
                fi
                return 1
            fi
            ;;
        *)
            echo "warning: unsupported package manager for automatic CUDA toolkit installation: ${package_manager}" >&2
            return 1
            ;;
    esac

    if [[ -L /usr/local/cuda && -n "${cuda_home}" && "${cuda_home}" != "/usr/local/cuda" ]]; then
        echo "Keeping existing /usr/local/cuda symlink; using detected CUDA root ${cuda_home}." >&2
    fi

    return 0
}

configure_cuda_include_dirs() {
    local include_dir
    local raw_include_dirs="$1"

    CUDA_INCLUDE_DIR_DETECTED=""
    CUDA_INCLUDE_DIRS_DETECTED=()
    mapfile -t CUDA_INCLUDE_DIRS_DETECTED <<< "${raw_include_dirs}"

    for include_dir in "${CUDA_INCLUDE_DIRS_DETECTED[@]}"; do
        if [[ -f "${include_dir}/cuda_runtime.h" || -f "${include_dir}/cuda.h" ]]; then
            CUDA_INCLUDE_DIR_DETECTED="${include_dir}"
            break
        fi
    done

    if [[ -z "${CUDA_INCLUDE_DIR_DETECTED}" ]]; then
        CUDA_INCLUDE_DIR_DETECTED="${CUDA_INCLUDE_DIRS_DETECTED[0]}"
    fi

    export CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR_DETECTED}"
    for include_dir in "${CUDA_INCLUDE_DIRS_DETECTED[@]}"; do
        prepend_env_path CPATH "${include_dir}"
        prepend_env_path CPLUS_INCLUDE_PATH "${include_dir}"
    done
}

refresh_cuda_environment() {
    CUDA_HOME_DETECTED=""
    if CUDA_HOME_DETECTED="$(resolve_cuda_home)"; then
        export CUDA_HOME="${CUDA_HOME_DETECTED}"
        append_cmake_arg "-DCUDAToolkit_ROOT=${CUDA_HOME}"
    fi

    NVRTC_LIBRARY=""
    if NVRTC_LIBRARY="$(find_nvrtc "${CUDA_HOME_DETECTED:-}")"; then
        export CUDA_NVRTC_LIBRARY="${NVRTC_LIBRARY}"
        append_cmake_arg "-DCUDA_nvrtc_LIBRARY=${NVRTC_LIBRARY}"
    else
        echo "warning: libnvrtc.so was not found automatically; the install may still fail during CMake configure." >&2
    fi

    CUDA_INCLUDE_DIRS_RAW=""
    if CUDA_INCLUDE_DIRS_RAW="$(find_cuda_include_dirs "${CUDA_HOME_DETECTED:-}")"; then
        configure_cuda_include_dirs "${CUDA_INCLUDE_DIRS_RAW}"
        return 0
    fi

    return 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

install_system_build_dependencies
if ! command -v git >/dev/null 2>&1; then
    echo "error: git is required so setuptools-scm can determine the vLLM version." >&2
    echo "Install git in the container, or set VLLM_VERSION_OVERRIDE to a valid version string and rerun." >&2
    exit 1
fi
if ! configure_git_for_repo; then
    export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-$(infer_vllm_version_override)}"
    echo "warning: git cannot introspect ${ROOT_DIR}; using VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE}." >&2
fi

CUDA_HOME_DETECTED=""
NVRTC_LIBRARY=""
CUDA_INCLUDE_DIR_DETECTED=""
CUDA_INCLUDE_DIRS_DETECTED=()
CUDA_INCLUDE_DIRS_RAW=""
if ! refresh_cuda_environment; then
    scanned_locations=()
    if [[ -n "${CUDA_INCLUDE_DIR:-}" ]]; then
        scanned_locations+=("${CUDA_INCLUDE_DIR}")
    fi
    if [[ -n "${CUDA_HOME_DETECTED:-}" ]]; then
        scanned_locations+=("${CUDA_HOME_DETECTED}")
    fi
    while IFS= read -r include_root; do
        scanned_locations+=("${include_root}")
    done < <(find_default_cuda_roots)
    missing_headers="$(describe_missing_cuda_headers "${scanned_locations[@]}")"
    if [[ "${VLLM_USE_PRECOMPILED:-}" =~ ^(1|true)$ || "${ALLOW_NO_CUDA_HEADERS:-0}" == "1" ]]; then
        echo "warning: CUDA headers (cublas_v2.h/cusparse.h/cusolverDn.h/nvrtc.h) not found; build may fail without dev headers." >&2
    elif [[ "${AUTO_INSTALL_CUDA_DEV_HEADERS}" =~ ^(1|true)$ ]] && install_cuda_dev_headers "${CUDA_HOME_DETECTED:-}"; then
        echo "Rechecking CUDA environment after CUDA toolkit installation." >&2
        if ! refresh_cuda_environment; then
            echo "error: CUDA toolkit installation completed but headers are still missing from the detected environment." >&2
            if [[ -n "${missing_headers}" ]]; then
                echo "Missing header families before install attempt: ${missing_headers}" >&2
            fi
            exit 1
        fi
    else
        echo "error: CUDA headers (cublas_v2.h/cusparse.h/cusolverDn.h/nvrtc.h) not found." >&2
        if [[ -n "${missing_headers}" ]]; then
            echo "Missing header families: ${missing_headers}" >&2
        fi
        echo "Install CUDA Toolkit dev headers or set CUDA_INCLUDE_DIR to a valid include path." >&2
        if [[ ${#scanned_locations[@]} -gt 0 ]]; then
            echo "Scanned CUDA roots: ${scanned_locations[*]}" >&2
        fi
        if [[ "${AUTO_INSTALL_CUDA_DEV_HEADERS}" =~ ^(0|false)$ ]]; then
            echo "Automatic package installation is disabled via AUTO_INSTALL_CUDA_DEV_HEADERS=${AUTO_INSTALL_CUDA_DEV_HEADERS}." >&2
        fi
        echo "If you cannot install headers, try: VLLM_USE_PRECOMPILED=1 ./scripts/install.sh" >&2
        exit 1
    fi
fi

# Reuse the already-installed torch/CUDA stack instead of creating an isolated
# build env that may lose local toolkit visibility.
export PIP_NO_BUILD_ISOLATION="${PIP_NO_BUILD_ISOLATION:-1}"

echo "Using ${PYTHON_BIN}: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
if [[ -n "${CUDA_HOME_DETECTED}" ]]; then
    echo "Using CUDA_HOME=${CUDA_HOME_DETECTED}"
fi
if [[ -n "${NVRTC_LIBRARY}" ]]; then
    echo "Using CUDA_NVRTC_LIBRARY=${NVRTC_LIBRARY}"
fi
if [[ -n "${CUDA_INCLUDE_DIR_DETECTED}" ]]; then
    echo "Using CUDA_INCLUDE_DIR=${CUDA_INCLUDE_DIR_DETECTED}"
fi
if [[ ${#CUDA_INCLUDE_DIRS_DETECTED[@]} -gt 1 ]]; then
    echo "Using additional CUDA include dirs: ${CUDA_INCLUDE_DIRS_DETECTED[*]}"
fi
if [[ -n "${CMAKE_ARGS:-}" ]]; then
    echo "Using CMAKE_ARGS=${CMAKE_ARGS}"
fi

# When build isolation is disabled, setuptools.build_meta imports run inside the
# active environment. Install the mirrored build requirements first so setup.py
# can import torch during editable metadata generation.
echo "Installing CUDA build requirements with torch backend ${UV_TORCH_BACKEND}."
if ! run_logged truncate pip_install -r requirements/build/cuda.txt numpy; then
    exit 1
fi

echo "Installing vLLM editable package."
run_logged append pip_install --no-build-isolation -e . "$@"
exit "$?"
