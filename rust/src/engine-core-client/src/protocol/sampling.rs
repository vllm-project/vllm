use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::protocol::structured_outputs::StructuredOutputsParams;

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> u32 {
    16
}

/// Engine-core-facing sampling parameters for text generation.
///
/// This is the normalized southbound subset used by the Rust frontend when it
/// talks to Python engine-core over the wire. User-facing request semantics
/// such as `stop` strings, `n`, `ignore_eos`, and output aggregation mode are
/// intentionally handled by higher layers before values reach this DTO.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
// Python's SamplingParams is `omit_defaults=True`, so msgpack drops
// default-valued keys; default the whole struct. Per-field fns cover the
// non-zero defaults.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DefaultFromSerde)]
#[serde(default)]
pub struct EngineCoreSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Cumulative probability threshold for nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Maximum number of top tokens to consider. `0` means all tokens.
    pub top_k: u32,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate per output sequence.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: u32,
    /// Maximum number of reasoning ("thinking") tokens to emit before the
    /// reasoning section is force-closed. `None` means unlimited; the
    /// user-facing `-1` sentinel is normalized to `None` by the frontend before
    /// reaching this DTO, so only non-negative values are sent. Enforced
    /// engine-side (and only when a reasoning parser is configured).
    pub thinking_token_budget: Option<u64>,
    /// Number of log probabilities to return per generated token.
    ///
    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.
    ///
    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
    /// Minimum probability threshold for token sampling.
    pub min_p: f32,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f32,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f32,
    /// Repetition penalty applied by the sampler.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Token IDs that stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Primary EOS token ID used by engine-core's dedicated EOS stop path.
    ///
    /// This mirrors Python's internal `_eos_token_id` field and is derived by
    /// the frontend from tokenizer/model metadata rather than supplied directly
    /// by end users.
    #[serde(rename = "_eos_token_id")]
    pub eos_token_id: Option<u32>,
    /// Complete stop-token set used by engine-core for `min_tokens` masking.
    ///
    /// This mirrors Python's internal `_all_stop_token_ids` field and should
    /// contain explicit `stop_token_ids` plus any frontend-derived EOS token
    /// IDs.
    #[serde(rename = "_all_stop_token_ids")]
    pub all_stop_token_ids: BTreeSet<u32>,
    /// Logit biases to apply during sampling.
    /// Keys are token IDs
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Tokenized bad words to avoid during generation.
    #[serde(rename = "_bad_words_token_ids")]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    /// Parameters for configuring structured outputs (guided decoding).
    pub structured_outputs: Option<StructuredOutputsParams>,
    /// Specific token IDs for which log probabilities should be returned at
    /// each position.
    ///
    /// When set, the engine returns logprobs for exactly these tokens in
    /// addition to the sampled/scored token. Mutually exclusive with the
    /// `logprobs` count field in practice.
    pub logprob_token_ids: Option<Vec<u32>>,
    /// If `Some(true)`, the request will not attempt to read from the prefix
    /// cache; newly computed blocks may still populate the cache. `None`
    /// defers to engine-core defaults.
    pub skip_reading_prefix_cache: Option<bool>,
    /// Additional request parameters for custom extensions (from `vllm_xargs`).
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
}

impl EngineCoreSamplingParams {
    /// Constructs a default sampling params for testing purposes only.
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            max_tokens: 65536,
            min_tokens: 0,
            thinking_token_budget: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            all_stop_token_ids: BTreeSet::new(),
            logit_bias: None,
            allowed_token_ids: None,
            bad_words_token_ids: None,
            structured_outputs: None,
            logprob_token_ids: None,
            skip_reading_prefix_cache: None,
            extra_args: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use crate::protocol::decode_msgpack;
    use crate::protocol::request::EngineCoreRequest;

    /// A real `sampling_params` is a sparse `omit_defaults` map; absent fields
    /// must fall back to defaults. `python_compat` can't catch this since Rust
    /// encodes full maps (see `engine_core_request_serializes_as_full_array`).
    #[test]
    fn decodes_sampling_params_with_omitted_defaults() {
        let sampling_params = Value::Map(vec![
            (
                Value::from("stop_token_ids"),
                Value::Array(vec![Value::from(151643u32)]),
            ),
            (Value::from("skip_reading_prefix_cache"), Value::from(false)),
        ]);
        let request = Value::Array(vec![
            Value::from("req-omit-defaults"),
            Value::Array(vec![
                Value::from(1u32),
                Value::from(2u32),
                Value::from(3u32),
            ]),
            Value::Nil,
            sampling_params,
            Value::Nil,
            Value::from(1.0f64),
        ]);

        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &request).unwrap();

        let decoded: EngineCoreRequest = decode_msgpack(&bytes)
            .expect("a real omit_defaults request must decode (regression: missing field)");

        assert_eq!(decoded.request_id, "req-omit-defaults");
        let sampling = decoded.sampling_params.expect("sampling params present");

        assert_eq!(sampling.stop_token_ids, vec![151643]);
        assert_eq!(sampling.skip_reading_prefix_cache, Some(false));

        // Omitted fields -> Python defaults.
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.top_k, 0);
        assert_eq!(sampling.seed, None);
        assert_eq!(sampling.max_tokens, 16);
        assert_eq!(sampling.min_tokens, 0);
        assert_eq!(sampling.min_p, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.logprobs, None);
        assert_eq!(sampling.prompt_logprobs, None);
        assert_eq!(sampling.eos_token_id, None);
        assert!(sampling.all_stop_token_ids.is_empty());
    }
}
