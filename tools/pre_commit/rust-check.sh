#!/usr/bin/env bash
# Wrapper for the rust-* pre-commit hooks.
#
# Skips (with a warning) when `cargo` or the requested cargo subcommand is
# not installed, so contributors who don't touch the Rust code aren't forced
# to install the Rust toolchain (or niche cargo extensions like cargo-sort
# / cargo-autoinherit). Buildkite CI covers the rust hooks regardless.
#
# Usage: tools/pre_commit/rust-check.sh <cargo-subcommand> [extra cargo args...]

set -euo pipefail

# Pre-commit captures stdout/stderr and only replays on failure. Try to write
# to /dev/tty so the warning is visible during a normal `git commit` even
# though we exit 0; fall back to stderr where there's no controlling tty
# (e.g. CI).
#
# The leading newline pushes the warning off pre-commit's dot-leader line so
# the message doesn't mash into "Rust - ... ........WARNING:". The hook's
# "Passed" still lands on its own line just below the warning.
warn() {
    { printf '\n%s\n' "$*" >/dev/tty; } 2>/dev/null || printf '\n%s\n' "$*" >&2
}

subcommand="$1"
shift

if ! command -v cargo >/dev/null 2>&1; then
    warn "WARNING: 'cargo' not found in PATH; skipping rust pre-commit hook (cargo ${subcommand}).
         Install the Rust toolchain via https://rustup.rs/ if you need to run rust hooks locally."
    exit 0
fi

# Cargo subcommands resolve to a `cargo-<name>` binary on PATH. Check up-front
# so a missing helper produces a friendly skip instead of a cargo error.
if ! command -v "cargo-${subcommand}" >/dev/null 2>&1; then
    case "${subcommand}" in
        fmt) install_hint="rustup component add rustfmt" ;;
        *)   install_hint="cargo install cargo-${subcommand}" ;;
    esac
    warn "WARNING: 'cargo ${subcommand}' is not installed; skipping rust pre-commit hook.
         Install it with: ${install_hint}"
    exit 0
fi

cd "$(git rev-parse --show-toplevel)/rust"
exec cargo "${subcommand}" "$@"
