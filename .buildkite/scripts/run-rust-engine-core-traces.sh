#!/usr/bin/env bash
# Record fresh traces from the Python engine core and replay them against the
# Rust port (`vllm-engine-core`). See rust/src/engine-core/tests/README.md.
#
# Usage: run-rust-engine-core-traces.sh [SEEDS]
#   SEEDS: a count or `lo-hi` passed to every generator (default: 3).
#
# Needs vLLM installed for the Python interpreter (PYTHON, default python3) and
# a C++ compiler; installs the pinned Rust toolchain if cargo is missing.
set -euo pipefail

SEEDS="${1:-3}"
PYTHON="${PYTHON:-python3}"

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

if ! command -v rustup >/dev/null 2>&1; then
  echo "--- Installing Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain none
fi
rustup toolchain install "$(awk -F '"' '/channel[[:space:]]*=/ { print $2; exit }' rust-toolchain.toml)" \
  --profile minimal

TRACE_DIR="$(mktemp -d)"
trap 'rm -rf "$TRACE_DIR"' EXIT

ENGINE_CORE=rust/src/engine-core/src
echo "--- Recording traces (seeds: $SEEDS) into $TRACE_DIR"
"$PYTHON" "$ENGINE_CORE/kv_cache/tests/gen_traces.py" \
  --seeds "$SEEDS" --out "$TRACE_DIR/kv_cache"
"$PYTHON" "$ENGINE_CORE/sched/scheduler/tests/gen_traces.py" \
  --seeds "$SEEDS" --out "$TRACE_DIR/sched"
"$PYTHON" "$ENGINE_CORE/sched/scheduler/tests/gen_nixl_traces.py" \
  --seeds "$SEEDS" --out "$TRACE_DIR/nixl"
"$PYTHON" "$ENGINE_CORE/sched/scheduler/tests/gen_mooncake_traces.py" \
  --seeds "$SEEDS" --out "$TRACE_DIR/mooncake"
du -sh "$TRACE_DIR"/*

echo "--- Replaying traces against vllm-engine-core"
ENGINE_CORE_TRACE_DIR="$TRACE_DIR" cargo test \
  --manifest-path rust/Cargo.toml \
  --locked \
  --release \
  -p vllm-engine-core \
  -- \
  kv_cache::tests::differential \
  sched::scheduler::tests::differential \
  sched::scheduler::tests::nixl_differential \
  sched::scheduler::tests::mooncake_store
