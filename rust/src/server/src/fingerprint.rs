// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The `system_fingerprint` of chat and completion responses.

use serde::Serialize;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

/// What `system_fingerprint` reports (`--fingerprint-mode`).
///
/// Original Python definition: `vllm/entrypoints/serve/utils/fingerprint.py`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub enum SystemFingerprintMode {
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

impl SystemFingerprintMode {
    /// Build the fingerprint from the engines' ready responses and the
    /// deployment-wide data-parallel size, like Python's
    /// `build_system_fingerprint`.
    pub(crate) fn fingerprint(
        &self,
        engines: &[&EngineCoreReadyResponse],
        data_parallel_size: usize,
    ) -> Option<String> {
        // Dense DP ranks auto-fit `max_model_len` independently; hash the
        // engine with the smallest one, the deployment's effective limit (the
        // value Python's frontend keeps).
        let ready = engines
            .iter()
            .min_by_key(|ready| ready.max_model_len)
            .expect("engine core client requires at least one engine");
        let hash8 = || {
            (ready.config_hash.as_deref()).map_or("nohash", |hash| hash.get(..8).unwrap_or(hash))
        };
        match self {
            Self::None => None,
            Self::Custom(value) => Some(value.clone()),
            Self::Hash => Some(format!("vllm-{}-{}", ready.vllm_version, hash8())),
            Self::Full => {
                let mut parts = vec![format!("vllm-{}", ready.vllm_version)];
                if ready.tensor_parallel_size > 1 {
                    parts.push(format!("tp{}", ready.tensor_parallel_size));
                }
                if ready.pipeline_parallel_size > 1 {
                    parts.push(format!("pp{}", ready.pipeline_parallel_size));
                }
                if data_parallel_size > 1 {
                    parts.push(format!("dp{data_parallel_size}"));
                }
                if ready.enable_expert_parallel {
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
    use vllm_engine_core_client::mock_engine::default_ready_response;

    use super::*;

    #[test]
    fn fingerprint_follows_python_modes() {
        let ready = EngineCoreReadyResponse {
            vllm_version: "0.33.0".to_string(),
            tensor_parallel_size: 2,
            pipeline_parallel_size: 1,
            enable_expert_parallel: true,
            config_hash: Some("c84c7d4f7a".to_string()),
            ..default_ready_response()
        };
        let unhashed = EngineCoreReadyResponse {
            config_hash: None,
            ..ready.clone()
        };
        let rows = [
            (SystemFingerprintMode::Full, vec![&ready], 4),
            (SystemFingerprintMode::Full, vec![&ready], 1),
            (SystemFingerprintMode::Full, vec![&unhashed], 1),
            (SystemFingerprintMode::Hash, vec![&ready], 4),
            (
                SystemFingerprintMode::Custom("fp-1".to_string()),
                vec![&ready],
                4,
            ),
            (SystemFingerprintMode::None, vec![&ready], 4),
        ];
        let fingerprints = rows.map(|(mode, engines, dp)| {
            format!("{mode:?} dp{dp}: {:?}", mode.fingerprint(&engines, dp))
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

    #[test]
    fn fingerprint_hashes_the_engine_with_the_smallest_max_model_len() {
        // Dense DP ranks auto-fit `max_model_len` independently; the
        // fingerprint uses the hash of the engine with the smallest one.
        let engine = |max_model_len, config_hash: &str| EngineCoreReadyResponse {
            vllm_version: "0.33.0".to_string(),
            max_model_len,
            config_hash: Some(config_hash.to_string()),
            ..default_ready_response()
        };
        let engines = [engine(8192, "cf9be74a00"), engine(4096, "e380051600")];
        let fingerprint = SystemFingerprintMode::Hash.fingerprint(&[&engines[0], &engines[1]], 2);
        assert_eq!(fingerprint.as_deref(), Some("vllm-0.33.0-e3800516"));
    }
}
