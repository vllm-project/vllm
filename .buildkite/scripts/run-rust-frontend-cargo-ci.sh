#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" != "style-clippy" && "$MODE" != "test" ]]; then
  echo "Usage: $0 {style-clippy|test}" >&2
  exit 2
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export PATH="$CARGO_HOME/bin:$PATH"

log_section() {
  echo "--- $*"
}

install_protoc() {
  if command -v protoc >/dev/null 2>&1; then
    return
  fi

  local version="${PROTOC_VERSION:-31.1}"
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

  local url="https://github.com/protocolbuffers/protobuf/releases/download/v${version}/protoc-${version}-linux-${arch}.zip"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  log_section "Installing protoc ${version}"
  curl -L --proto '=https' --tlsv1.2 -sSf "$url" -o "$tmp_dir/protoc.zip"
  mkdir -p "$CARGO_HOME/bin"
  unzip -q "$tmp_dir/protoc.zip" bin/protoc 'include/*' -d "$CARGO_HOME"
  chmod +x "$CARGO_HOME/bin/protoc"
  rm -rf "$tmp_dir"
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
  if command -v cargo-binstall >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-binstall"
  curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh \
    | bash
}

install_cargo_sort() {
  if command -v cargo-sort >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-sort"
  install_cargo_binstall
  cargo binstall --no-confirm cargo-sort
}

install_cargo_nextest() {
  if command -v cargo-nextest >/dev/null 2>&1; then
    return
  fi

  log_section "Installing cargo-nextest"
  install_cargo_binstall
  cargo binstall --no-confirm --secure cargo-nextest
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  log_section "Installing uv"
  curl -LsSf --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$CARGO_HOME/bin" sh
}

setup_pyo3_python() {
  local python_version="${PYO3_PYTHON_VERSION:-3.12}"

  log_section "Installing Python ${python_version} for PyO3 tests"
  uv python install "$python_version"
  PYO3_PYTHON="$(uv python find \
    --managed-python \
    --no-project \
    --resolve-links \
    "$python_version")"
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
  install_cargo_sort

  log_section "Checking Rust formatting"
  cargo fmt --manifest-path rust/Cargo.toml --all -- --check

  log_section "Checking Cargo.toml ordering"
  cargo sort --workspace --check rust

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
