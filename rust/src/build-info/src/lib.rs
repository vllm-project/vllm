// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::path::Path;
use std::sync::LazyLock;

/// The version of the vLLM package that ships this binary.
///
/// Read at run time from `_version.py` next to the executable, where the vLLM
/// wheel installs `vllm-rs`, so the binary does not change with every commit.
/// Binaries run outside the package, such as `cargo run`, fall back to the
/// internal crate version.
pub fn version() -> &'static str {
    static VERSION: LazyLock<String> = LazyLock::new(|| {
        std::env::current_exe()
            .and_then(std::fs::canonicalize)
            .ok()
            .and_then(|exe| package_version(&exe.with_file_name("_version.py")))
            .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_owned())
    });
    &VERSION
}

fn package_version(version_file: &Path) -> Option<String> {
    parse_version_file(&std::fs::read_to_string(version_file).ok()?)
}

/// Parse the `__version__ = version = '...'` line that setuptools-scm writes.
fn parse_version_file(text: &str) -> Option<String> {
    let value = text.lines().find_map(|line| line.strip_prefix("__version__ = version = "))?;
    let version = value.trim().trim_matches(['\'', '"']);
    (!version.is_empty()).then(|| version.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_setuptools_scm_version_file() {
        let text = "version: str\n__version__: str\n\
                    __version__ = version = '0.34.0rc1.dev648+g94de86707'\n\
                    __version_tuple__ = version_tuple = (0, 34, 0)\n";
        assert_eq!(
            parse_version_file(text).as_deref(),
            Some("0.34.0rc1.dev648+g94de86707")
        );
    }

    #[test]
    fn rejects_file_without_version() {
        assert_eq!(parse_version_file("version: str\n"), None);
        assert_eq!(parse_version_file("__version__ = version = ''\n"), None);
    }
}
