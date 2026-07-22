#!/bin/bash
# Build vLLM Rust artifacts and install them into the vllm package.
# Usage: ./build_rust.sh [--debug]
#
# By default builds in release mode. Pass --debug for faster compile times
# during development.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CARGO_LLVM_COV_VERSION="0.8.7"
COVERAGE_TOOLS_DIR="$REPO_ROOT/rust-coverage-tools"

# Read the required toolchain from rust-toolchain.toml.
TOOLCHAIN=$(grep '^channel' "$REPO_ROOT/rust-toolchain.toml" | sed 's/.*= *"\(.*\)"/\1/')

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARG="--debug"
else
    PROFILE_ARG="--release"
fi

rm -rf "$COVERAGE_TOOLS_DIR"
mkdir -p "$COVERAGE_TOOLS_DIR/bin" "$COVERAGE_TOOLS_DIR/lib"

if [[ "${VLLM_RUST_COVERAGE:-0}" == "1" ]]; then
    # rustc wrapper flags are invisible to Cargo's normal fingerprinting.
    # Keep instrumented intermediates isolated when local builds switch modes.
    export CARGO_TARGET_DIR="$REPO_ROOT/rust/target/coverage"
    rustup component add --toolchain "$TOOLCHAIN" llvm-tools-preview
    cargo +"$TOOLCHAIN" install \
        --locked \
        --version "$CARGO_LLVM_COV_VERSION" \
        cargo-llvm-cov

    eval "$(
        cargo +"$TOOLCHAIN" llvm-cov show-env \
            --manifest-path "$REPO_ROOT/rust/Cargo.toml" \
            --sh
    )"

    # Build scripts and proc macros can run during compilation. Their profiles
    # are unrelated to runtime coverage and would otherwise pollute the tree.
    export LLVM_PROFILE_FILE=/dev/null
fi

python3 "$REPO_ROOT/tools/build_rust.py" "$PROFILE_ARG"

if [[ "${VLLM_RUST_COVERAGE:-0}" == "1" ]]; then
    LLVM_BIN_DIR="$(dirname "$(rustup run "$TOOLCHAIN" rustc \
        --print target-libdir)")/bin"

    cp "$LLVM_BIN_DIR"/{llvm-cov,llvm-profdata} "$COVERAGE_TOOLS_DIR/bin/"
    chmod 0755 "$COVERAGE_TOOLS_DIR/bin/"*
    cp -L "$LLVM_BIN_DIR"/../lib/libLLVM.so* "$COVERAGE_TOOLS_DIR/lib/"
    chmod 0644 "$COVERAGE_TOOLS_DIR/lib/"*
fi
