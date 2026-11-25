// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;

/// The engine that the `system_fingerprint` of responses describes: the inputs
/// of Python's `build_system_fingerprint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmFingerprint {
    /// vLLM version of the engine.
    pub vllm_version: String,
    /// Hash of the engine config (Python's `hash_config`), or `None` when it is
    /// not reported.
    pub config_hash: Option<String>,
    /// Tensor-parallel size of each engine.
    pub tensor_parallel_size: u32,
    /// Pipeline-parallel size of each engine.
    pub pipeline_parallel_size: u32,
    /// Data-parallel size of the whole deployment.
    pub data_parallel_size: usize,
    /// Whether experts are spread across the engine's devices.
    pub enable_expert_parallel: bool,
}

impl LlmFingerprint {
    /// Describe `engines`, in a deployment of `data_parallel_size` ranks.
    pub(crate) fn from_engines(
        engines: &[&EngineCoreReadyResponse],
        data_parallel_size: usize,
    ) -> Self {
        // Dense DP ranks auto-fit `max_model_len` independently; describe the
        // engine with the smallest one, the deployment's effective limit (the
        // value Python's frontend keeps).
        let ready = engines
            .iter()
            .min_by_key(|ready| ready.max_model_len)
            .expect("engine core client requires at least one engine");
        Self {
            vllm_version: ready.vllm_version.clone(),
            config_hash: ready.config_hash.clone(),
            tensor_parallel_size: ready.tensor_parallel_size,
            pipeline_parallel_size: ready.pipeline_parallel_size,
            data_parallel_size,
            enable_expert_parallel: ready.enable_expert_parallel,
        }
    }
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::mock_engine::default_ready_response;

    use super::*;

    #[test]
    fn fingerprint_describes_the_engine_with_the_smallest_max_model_len() {
        let engine =
            |max_model_len, vllm_version: &str, config_hash: &str| EngineCoreReadyResponse {
                max_model_len,
                vllm_version: vllm_version.to_string(),
                config_hash: Some(config_hash.to_string()),
                ..default_ready_response()
            };
        let engines = [
            engine(8192, "0.33.0", "cf9be74a00"),
            engine(4096, "0.33.1", "e380051600"),
        ];
        let fingerprint = LlmFingerprint::from_engines(&[&engines[0], &engines[1]], 2);
        // The version and the hash come from the same engine.
        assert_eq!(fingerprint.vllm_version, "0.33.1");
        assert_eq!(fingerprint.config_hash.as_deref(), Some("e380051600"));
        assert_eq!(fingerprint.data_parallel_size, 2);
    }
}
