#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="auto"
TARGET_VENV=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
EDITABLE=0

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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to bootstrap uv." >&2
  exit 1
fi

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

INSTALL_TARGET="${ROOT_DIR}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
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
