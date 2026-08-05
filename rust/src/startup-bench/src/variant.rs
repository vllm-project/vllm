// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! A single named `--variant NAME=COMMAND` entry.

use std::fmt;

/// One command to benchmark, identified by a short name used in output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub name: String,
    pub command: String,
}

impl Variant {
    /// Parse a `NAME=COMMAND` string into a [`Variant`].
    ///
    /// Only the first `=` is significant, since `COMMAND` is a full shell
    /// command line that may itself contain `=` (e.g. `FOO=1 vllm serve ...`).
    pub fn parse(s: &str) -> Result<Self, String> {
        let (name, command) =
            s.split_once('=').ok_or_else(|| format!("expected NAME=COMMAND, got {s:?}"))?;
        if name.trim().is_empty() {
            return Err(format!("variant name must not be empty, got {s:?}"));
        }
        if command.trim().is_empty() {
            return Err(format!("variant command must not be empty, got {s:?}"));
        }
        Ok(Self {
            name: name.to_string(),
            command: command.to_string(),
        })
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::Variant;

    #[test]
    fn parses_name_and_command() {
        let variant = Variant::parse("python=vllm serve Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(variant.name, "python");
        assert_eq!(variant.command, "vllm serve Qwen/Qwen3-0.6B");
    }

    #[test]
    fn command_may_contain_equals_signs() {
        let variant =
            Variant::parse("rust=VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(variant.name, "rust");
        assert_eq!(
            variant.command,
            "VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B"
        );
    }

    #[test]
    fn rejects_missing_equals() {
        assert!(Variant::parse("no-equals-sign-here").is_err());
    }

    #[test]
    fn rejects_empty_name() {
        assert!(Variant::parse("=vllm serve foo").is_err());
    }

    #[test]
    fn rejects_empty_command() {
        assert!(Variant::parse("name=").is_err());
    }
}
