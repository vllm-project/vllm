#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="auto"
TARGET_VENV=""
PYTHON_BIN="${PYTHON_BIN:-}"
EDITABLE=0
HOST_OS="$(uname -s)"

usage() {
  cat <<'EOF'
Usage: ./scripts/install.sh [--system | --user | --venv PATH] [--editable] [--python PYTHON]

Install this vLLM checkout with an Ollama-like local CLI launcher.

Modes:
  --system       Install into /opt/vllm and /usr/local/bin/vllm
  --user         Install into ~/.local/share/vllm and ~/.local/bin/vllm
  --venv PATH    Install into an explicit Python virtual environment

Options:
  --editable     Install the repo in editable mode
  --python BIN   Python interpreter to use when creating the environment
  -h, --help     Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      MODE="system"
      shift
      ;;
    --user)
      MODE="user"
      shift
      ;;
    --venv)
      MODE="venv"
      TARGET_VENV="${2:-}"
      if [[ -z "${TARGET_VENV}" ]]; then
        echo "--venv requires a target path" >&2
        exit 1
      fi
      shift 2
      ;;
    --editable)
      EDITABLE=1
      shift
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      if [[ -z "${PYTHON_BIN}" ]]; then
        echo "--python requires an interpreter path" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${MODE}" == "auto" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    MODE="system"
  else
    MODE="user"
  fi
fi

if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
  echo "System mode writes to /opt/vllm and /usr/local/bin. Re-run with sudo or use --user/--venv." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to bootstrap uv." >&2
  exit 1
fi

pick_python() {
  local candidate=""

  if [[ -n "${PYTHON_BIN}" ]]; then
    candidate="${PYTHON_BIN}"
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      echo "Python interpreter not found: ${candidate}" >&2
      exit 1
    fi
    echo "${candidate}"
    return
  fi

  for candidate in python3.13 python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi

  echo "No supported Python interpreter found. Install Python 3.10-3.13 or pass --python." >&2
  exit 1
}

validate_python_version() {
  local candidate="$1"
  if ! "${candidate}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)'; then
    local actual_version
    actual_version="$("${candidate}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    echo "Unsupported Python version: ${actual_version}. vLLM requires Python >=3.10,<3.14." >&2
    exit 1
  fi
}

PYTHON_BIN="$(pick_python)"
validate_python_version "${PYTHON_BIN}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi

  echo "Installing uv..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    echo "${HOME}/.local/bin/uv"
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi

  echo "uv installation succeeded but uv is not on PATH." >&2
  exit 1
}

UV_BIN="$(ensure_uv)"

case "${MODE}" in
  system)
    INSTALL_ROOT="${VLLM_INSTALL_ROOT:-/opt/vllm}"
    BIN_DIR="${VLLM_BIN_DIR:-/usr/local/bin}"
    VENV_DIR="${INSTALL_ROOT}/venv"
    ;;
  user)
    INSTALL_ROOT="${VLLM_INSTALL_ROOT:-${HOME}/.local/share/vllm}"
    BIN_DIR="${VLLM_BIN_DIR:-${HOME}/.local/bin}"
    VENV_DIR="${INSTALL_ROOT}/venv"
    ;;
  venv)
    INSTALL_ROOT=""
    BIN_DIR=""
    VENV_DIR="${TARGET_VENV}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${VENV_DIR}")"
"${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" certifi
CERT_BUNDLE="$("${VENV_DIR}/bin/python" -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="${CERT_BUNDLE}"
export REQUESTS_CA_BUNDLE="${CERT_BUNDLE}"

INSTALL_TARGET="${ROOT_DIR}"
if [[ "${HOST_OS}" == "Darwin" ]]; then
  echo "macOS detected: using the Apple Silicon CPU source-build path." >&2
  "${UV_BIN}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-strategy unsafe-best-match \
    -r "${ROOT_DIR}/requirements/cpu.txt"
  export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cpu}"
  export VLLM_USE_PRECOMPILED=0
else
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
fi

if [[ "${EDITABLE}" -eq 1 ]]; then
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -e "${INSTALL_TARGET}"
else
  "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" "${INSTALL_TARGET}"
fi

if [[ "${MODE}" != "venv" ]]; then
  mkdir -p "${BIN_DIR}"
  cat > "${BIN_DIR}/vllm" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${VENV_DIR}/bin/python" -m vllm.entrypoints.cli.main "\$@"
EOF
  chmod +x "${BIN_DIR}/vllm"
fi

echo
echo "vLLM install complete."
case "${MODE}" in
  system)
    echo "Launcher: ${BIN_DIR}/vllm"
    ;;
  user)
    echo "Launcher: ${BIN_DIR}/vllm"
    if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
      echo "Add ${BIN_DIR} to PATH if needed."
    fi
    ;;
  venv)
    echo "Environment: ${VENV_DIR}"
    echo "Activate with: source ${VENV_DIR}/bin/activate"
    ;;
esac
echo "Try: vllm --help"
