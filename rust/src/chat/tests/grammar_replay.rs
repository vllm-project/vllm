// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays the grammar cases exported by the roundtrip tests through the real
//! XGrammar, so that parser-built output grammars are checked against the
//! engine's compiler and matcher.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn xgrammar_accepts_roundtrip_generations() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/grammar_replay.py");
    let output = Command::new(&script)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {:?}: {error}", script));
    assert!(
        output.status.success(),
        "grammar replay failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
