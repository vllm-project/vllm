#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" != "style-clippy" && "$MODE" != "test" ]]; then
  echo "Usage: $0 {style-clippy|test}" >&2
  exit 2
fi

if ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
fi
cd "$ROOT_DIR"

export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"

PROTOC_VERSION="${PROTOC_VERSION:-31.1}"
CARGO_BINSTALL_VERSION="${CARGO_BINSTALL_VERSION:-1.20.1}"
UV_VERSION="${UV_VERSION:-0.11.28}"
PYO3_PYTHON_VERSION="${PYO3_PYTHON_VERSION:-3.12}"

CARGO_SORT_VERSION_REQ="${CARGO_SORT_VERSION_REQ:-2}"
CARGO_DENY_VERSION_REQ="${CARGO_DENY_VERSION_REQ:-0.20}"
CARGO_NEXTEST_VERSION_REQ="${CARGO_NEXTEST_VERSION_REQ:-0.9}"

log_section() {
  echo "--- $*"
}

install_protoc() {
  local arch
  case "$(uname -m)" in
    x86_64)
      arch="x86_64"
      ;;
    aarch64|arm64)
      arch="aarch_64"
      ;;
    *)
      echo "Unsupported protoc architecture: $(uname -m)" >&2
      return 1
      ;;
  esac

  local url="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-${arch}.zip"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  log_section "Installing protoc ${PROTOC_VERSION}"
  curl -L --proto '=https' --tlsv1.2 -sSf "$url" -o "$tmp_dir/protoc.zip"
  mkdir -p "$CARGO_HOME/bin"
  unzip -q "$tmp_dir/protoc.zip" bin/protoc 'include/*' -d "$CARGO_HOME"
  chmod +x "$CARGO_HOME/bin/protoc"
  rm -rf "$tmp_dir"
  protoc --version
}

rust_toolchain() {
  awk -F '"' '/channel[[:space:]]*=/ { print $2; exit }' rust-toolchain.toml
}

install_rust_toolchain() {
  log_section "Installing Rust toolchain"
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none
  fi

  local toolchain
  toolchain="$(rust_toolchain)"
  rustup toolchain install "$toolchain" --profile minimal --component rustfmt,clippy
  rustup component add --toolchain "$toolchain" rustfmt clippy
}

install_cargo_binstall() {
  log_section "Installing cargo-binstall ${CARGO_BINSTALL_VERSION}"
  curl -L --proto '=https' --tlsv1.2 -sSf \
    "https://raw.githubusercontent.com/cargo-bins/cargo-binstall/v${CARGO_BINSTALL_VERSION}/install-from-binstall-release.sh" \
    | env BINSTALL_VERSION="$CARGO_BINSTALL_VERSION" bash
  cargo-binstall -V
}

install_cargo_sort() {
  log_section "Installing cargo-sort ${CARGO_SORT_VERSION_REQ}"
  cargo binstall --no-confirm --force "cargo-sort@${CARGO_SORT_VERSION_REQ}"
}

install_cargo_deny() {
  log_section "Installing cargo-deny ${CARGO_DENY_VERSION_REQ}"
  cargo binstall --no-confirm --force "cargo-deny@${CARGO_DENY_VERSION_REQ}"
}

install_cargo_nextest() {
  log_section "Installing cargo-nextest ${CARGO_NEXTEST_VERSION_REQ}"
  cargo binstall \
    --no-confirm \
    --force \
    --secure \
    "cargo-nextest@${CARGO_NEXTEST_VERSION_REQ}"
}

install_uv() {
  log_section "Installing uv ${UV_VERSION}"
  curl -L --proto '=https' --tlsv1.2 -sSf \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" \
    | env UV_INSTALL_DIR="$CARGO_HOME/bin" sh
  uv --version
}

setup_pyo3_python() {
  log_section "Installing Python ${PYO3_PYTHON_VERSION} for PyO3 tests"
  uv python install "$PYO3_PYTHON_VERSION"
  PYO3_PYTHON="$(uv python find \
    --managed-python \
    --no-project \
    --resolve-links \
    "$PYO3_PYTHON_VERSION")"
  export PYO3_PYTHON

  local python_libdir
  python_libdir="$("$PYO3_PYTHON" - <<'PY'
import pathlib
import sysconfig

libdir = pathlib.Path(sysconfig.get_config_var("LIBDIR"))
ldlibrary = sysconfig.get_config_var("LDLIBRARY")
assert sysconfig.get_config_var("Py_ENABLE_SHARED") == 1
assert ldlibrary
assert (libdir / ldlibrary).exists(), libdir / ldlibrary
print(libdir)
PY
)"

  export LD_LIBRARY_PATH="${python_libdir}:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${python_libdir}:${LIBRARY_PATH:-}"
}

run_style_clippy() {
  install_cargo_binstall
  install_cargo_sort
  install_cargo_deny

  log_section "Checking Rust formatting"
  cargo fmt --manifest-path rust/Cargo.toml --all -- --check

  log_section "Checking Cargo.toml ordering"
  cargo sort --workspace --check rust

  log_section "Checking Rust dependency bans"
  cargo deny \
    --manifest-path rust/Cargo.toml \
    --config rust/deny.toml \
    check \
    bans

  log_section "Running clippy"
  cargo clippy \
    --manifest-path rust/Cargo.toml \
    --workspace \
    --all-targets \
    --all-features \
    --locked \
    -- \
    -D warnings
}

run_tests() {
  install_uv
  setup_pyo3_python
  install_cargo_binstall
  install_cargo_nextest

  log_section "Running cargo nextest"
  cargo nextest run \
    --manifest-path rust/Cargo.toml \
    --workspace \
    --all-features \
    --locked \
    --no-fail-fast
}

install_protoc
install_rust_toolchain

case "$MODE" in
  style-clippy)
    run_style_clippy
    ;;
  test)
    run_tests
    ;;
esac
