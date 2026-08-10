// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Flag parity with the Python `vllm bench serve` parser.
//!
//! `VLLM_USE_RUST_BENCH=1` makes `vllm bench serve` exec this crate's binary
//! with the raw argv, so every documented Python flag must either parse here
//! or be a deliberate, allowlisted gap. The snapshot is kept current by
//! `tests/benchmarks/test_rust_bench_cli_parity.py` in the repo root.

use std::collections::HashSet;

use clap::{CommandFactory, Parser};
use vllm_bench::BenchServeArgs;

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    args: BenchServeArgs,
}

const PYTHON_FLAGS: &str = include_str!("python_serve_flags.txt");

/// Python-only flags: `vllm bench serve` features vllm-bench deliberately does
/// not implement. Adding an entry is a decision to let the flag fail under
/// Rust delegation; remove the entry once the flag gains Rust support.
const PYTHON_ONLY: &[&str] = &[
    // asr dataset
    "--asr-max-audio-len-sec",
    "--asr-min-audio-len-sec",
    // bfcl dataset
    "--bfcl-categories",
    // blazedit dataset
    "--blazedit-max-distance",
    "--blazedit-min-distance",
    // spec_bench dataset
    "--spec-bench-category",
    "--spec-bench-output-len",
    // timed-trace dataset
    "--timed-trace-chunk-hash-size",
    "--timed-trace-label-hash-ids",
    "--timed-trace-label-input-length",
    "--timed-trace-label-output-length",
    "--timed-trace-label-timestamp",
    "--timed-trace-sec-multiplier",
    // client-side chat templating / request shaping
    "--chat-template-kwargs",
    "--custom-ensure-client-side-data",
    "--use-beam-search",
    // result post-processing and plotting
    "--plot-dataset-stats",
    "--plot-timeline",
    "--timeline-itl-thresholds",
    // misc python-side controls
    "--hf-name",
    "--no-self-timed",
    "--self-timed",
    "--no-stream",
    "--probe-request-rate",
];

fn rust_flags() -> HashSet<String> {
    let cmd = TestCli::command();
    let mut flags = HashSet::new();
    for arg in cmd.get_arguments() {
        if let Some(long) = arg.get_long() {
            flags.insert(format!("--{long}"));
        }
        if let Some(aliases) = arg.get_all_aliases() {
            for alias in aliases {
                flags.insert(format!("--{alias}"));
            }
        }
    }
    flags
}

fn python_flags() -> Vec<&'static str> {
    PYTHON_FLAGS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

#[test]
fn python_serve_flags_parse_or_are_allowlisted() {
    let known = rust_flags();
    let missing: Vec<&str> = python_flags()
        .into_iter()
        .filter(|flag| !known.contains(*flag) && !PYTHON_ONLY.contains(flag))
        .collect();
    assert!(
        missing.is_empty(),
        "Python `vllm bench serve` flags unknown to vllm-bench: {missing:?}. \
         Support them in src/cli.rs (a clap alias is enough for renames) or \
         add them to PYTHON_ONLY above as a deliberate gap."
    );
}

#[test]
fn python_only_allowlist_is_not_stale() {
    let known = rust_flags();
    let stale: Vec<&&str> = PYTHON_ONLY.iter().filter(|f| known.contains(**f)).collect();
    assert!(
        stale.is_empty(),
        "PYTHON_ONLY entries now supported by vllm-bench, remove them: {stale:?}"
    );
}

#[test]
fn python_only_entries_exist_in_snapshot() {
    let snapshot: HashSet<&str> = python_flags().into_iter().collect();
    let gone: Vec<&&str> = PYTHON_ONLY.iter().filter(|f| !snapshot.contains(**f)).collect();
    assert!(
        gone.is_empty(),
        "PYTHON_ONLY entries no longer exist in the Python parser, remove them: {gone:?}"
    );
}
