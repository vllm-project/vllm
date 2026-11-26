// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The `system_fingerprint` of chat and completion responses.

use serde::Serialize;
use vllm_llm::LlmFingerprint;

/// What `system_fingerprint` reports (`--fingerprint-mode`).
///
/// Original Python definition: `vllm/entrypoints/serve/utils/fingerprint.py`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub enum FingerprintMode {
    /// `vllm-<version>[-tpN][-ppN][-dpN][-ep]-<hash8>`: the server version,
    /// any non-trivial parallelism, and a prefix of the engine config hash.
    #[default]
    Full,
    /// `vllm-<version>-<hash8>`.
    Hash,
    /// A literal string (`--fingerprint-value`).
    Custom(String),
    /// No fingerprint.
    None,
}

impl FingerprintMode {
    /// Build the fingerprint of the engine that `fingerprint` describes, like
    /// Python's `build_system_fingerprint`.
    pub(crate) fn fingerprint(&self, fingerprint: &LlmFingerprint) -> Option<String> {
        let hash8 = || {
            (fingerprint.config_hash.as_deref())
                .map_or("nohash", |hash| hash.get(..8).unwrap_or(hash))
        };
        match self {
            Self::None => None,
            Self::Custom(value) => Some(value.clone()),
            Self::Hash => Some(format!("vllm-{}-{}", fingerprint.vllm_version, hash8())),
            Self::Full => {
                let mut parts = vec![format!("vllm-{}", fingerprint.vllm_version)];
                if fingerprint.tensor_parallel_size > 1 {
                    parts.push(format!("tp{}", fingerprint.tensor_parallel_size));
                }
                if fingerprint.pipeline_parallel_size > 1 {
                    parts.push(format!("pp{}", fingerprint.pipeline_parallel_size));
                }
                if fingerprint.data_parallel_size > 1 {
                    parts.push(format!("dp{}", fingerprint.data_parallel_size));
                }
                if fingerprint.enable_expert_parallel {
                    parts.push("ep".to_string());
                }
                parts.push(hash8().to_string());
                Some(parts.join("-"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    #[test]
    fn fingerprint_follows_python_modes() {
        let engine = |data_parallel_size, config_hash: Option<&str>| LlmFingerprint {
            vllm_version: "0.33.0".to_string(),
            config_hash: config_hash.map(str::to_string),
            tensor_parallel_size: 2,
            pipeline_parallel_size: 1,
            data_parallel_size,
            enable_expert_parallel: true,
        };
        let hashed = Some("c84c7d4f7a");
        let rows = [
            (FingerprintMode::Full, engine(4, hashed)),
            (FingerprintMode::Full, engine(1, hashed)),
            (FingerprintMode::Full, engine(1, None)),
            (FingerprintMode::Hash, engine(4, hashed)),
            (
                FingerprintMode::Custom("fp-1".to_string()),
                engine(4, hashed),
            ),
            (FingerprintMode::None, engine(4, hashed)),
        ];
        let fingerprints = rows.map(|(mode, engine)| {
            let dp = engine.data_parallel_size;
            format!("{mode:?} dp{dp}: {:?}", mode.fingerprint(&engine))
        });
        expect![[r#"
            Full dp4: Some("vllm-0.33.0-tp2-dp4-ep-c84c7d4f")
            Full dp1: Some("vllm-0.33.0-tp2-ep-c84c7d4f")
            Full dp1: Some("vllm-0.33.0-tp2-ep-nohash")
            Hash dp4: Some("vllm-0.33.0-c84c7d4f")
            Custom("fp-1") dp4: Some("fp-1")
            None dp4: None"#]]
        .assert_eq(&fingerprints.join("\n"));
    }
}
