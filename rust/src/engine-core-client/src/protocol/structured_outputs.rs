use serde::{Deserialize, Serialize};

/// Structured-output backend selected for EngineCore grammar compilation.
///
/// Python vLLM stores this in `StructuredOutputsParams._backend` after request
/// validation. The Rust frontend currently always lowers structured-output
/// requests to guidance, while ignoring any user-supplied `_backend` value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StructuredOutputBackend {
    Xgrammar,
    #[default]
    Guidance,
    Outlines,
    LmFormatEnforcer,
}

/// Parameters for configuring structured outputs (guided decoding).
///
/// Exactly one constraint field (`json`, `regex`, `choice`, `grammar`,
/// `json_object`, or `structural_tag`) should be set. The engine-core
/// backend selects the appropriate grammar compiler based on which field
/// is present.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L36-L107>
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StructuredOutputsParams {
    /// JSON schema (as a dict/object or JSON string) constraining the output.
    pub json: Option<serde_json::Value>,
    /// Regular expression the output must match.
    pub regex: Option<String>,
    /// List of allowed output strings (the model must produce one of these).
    pub choice: Option<Vec<String>>,
    /// Context-free grammar (in EBNF-like notation) the output must conform to.
    pub grammar: Option<String>,
    /// When `true`, output must be valid JSON (free-form, no schema).
    pub json_object: Option<bool>,
    /// Disable any additional whitespace in guided JSON output.
    #[serde(skip_serializing_if = "crate::protocol::is_false")]
    pub disable_any_whitespace: bool,
    /// Disable `additionalProperties` in JSON schema output.
    #[serde(skip_serializing_if = "crate::protocol::is_false")]
    pub disable_additional_properties: bool,
    /// Custom whitespace pattern for guided JSON output.
    pub whitespace_pattern: Option<String>,
    /// Structural tag configuration (JSON-encoded string).
    pub structural_tag: Option<String>,
    /// Structured-output backend, mirroring Python's internal `_backend`.
    ///
    /// User-supplied values are ignored during deserialization. This matches
    /// Python's request boundary, where `_backend` is set by validation rather
    /// than accepted as a request-level backend selector.
    #[serde(
        default,
        rename = "_backend",
        deserialize_with = "serde_with::rust::deserialize_ignore_any"
    )]
    pub backend: StructuredOutputBackend,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_outputs_backend_ignores_deserialized_value() {
        let params: StructuredOutputsParams = serde_json::from_value(serde_json::json!({
            "json_object": true,
            "_backend": "xgrammar",
        }))
        .unwrap();

        assert_eq!(params.backend, StructuredOutputBackend::Guidance);

        let value = serde_json::to_value(params).unwrap();
        assert_eq!(value["_backend"], "guidance");
    }
}
