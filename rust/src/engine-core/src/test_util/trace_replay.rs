// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Discovery and replay of the traces recorded from the Python implementation
//! (see `tests/README.md`).

use std::panic::RefUnwindSafe;
use std::path::{Path, PathBuf};

/// Root of freshly recorded traces, one subdirectory per suite. When unset,
/// each suite replays its small checked-in set.
const TRACE_DIR_ENV: &str = "ENGINE_CORE_TRACE_DIR";

/// Replays every `*.{ext}` trace of `suite` and fails with all the traces that
/// panicked. `checked_in` is the suite's directory of checked-in traces.
pub fn replay_all(
    suite: &str,
    checked_in: &Path,
    ext: &str,
    replay: impl Fn(&Path) + RefUnwindSafe,
) {
    let dir = std::env::var_os(TRACE_DIR_ENV).map_or_else(
        || checked_in.to_path_buf(),
        |root| PathBuf::from(root).join(suite),
    );
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no traces in {}", dir.display());
    let failures: Vec<String> = paths
        .iter()
        .filter_map(|path| {
            std::panic::catch_unwind(|| replay(path)).err().map(|e| {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                format!("{}: {msg}", path.display())
            })
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} traces failed:\n{}",
        failures.len(),
        paths.len(),
        failures.join("\n")
    );
}
