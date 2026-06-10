use serde::{Deserialize, Serialize};
use serde_json::Value;
use vllm_engine_core_client::protocol::StructuredOutputsParams;

use crate::error::ApiError;

/// JSON schema specification nested inside a `json_schema` response format.
///
/// Mirrors the Python vLLM `JsonSchemaResponseFormat` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// The actual JSON schema object.
    #[serde(alias = "json_schema")]
    pub schema: Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// Supported `response_format` types for chat and completion requests.
///
/// This is our own definition (rather than the `openai-protocol` crate's) so
/// that we can support the vLLM-specific `structural_tag` variant.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/entrypoints/openai/engine/protocol.py#L116-L157>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: JsonSchemaFormat,
    },
    /// vLLM-specific structural tag format. The entire object (including the
    /// `type` field) is JSON-serialized and passed as
    /// `StructuredOutputsParams.structural_tag`.
    ///
    /// We capture the payload as a catch-all map so both the legacy
    /// (`structures`/`triggers`) and current (`format`) shapes are
    /// preserved without needing typed structs.
    StructuralTag {
        #[serde(flatten)]
        extra: serde_json::Map<String, Value>,
    },
}

/// Convert an explicit `structured_outputs` JSON blob into
/// [`StructuredOutputsParams`].
fn deserialize_structured_outputs(
    raw: &serde_json::Value,
) -> Result<StructuredOutputsParams, ApiError> {
    serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid structured_outputs: {e}"),
            Some("structured_outputs"),
        )
    })
}

/// Convert a typed [`ResponseFormat`] and/or raw `structured_outputs` blob into
/// engine-core [`StructuredOutputsParams`].
///
/// Mirrors the Python vLLM conversion in
/// `ChatCompletionRequest.to_sampling_params()`: <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/entrypoints/openai/chat_completion/protocol.py#L457-L487>
pub fn convert_from_response_format(
    response_format: Option<&ResponseFormat>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputsParams>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(fmt) = response_format else {
        return Ok(None);
    };
    match fmt {
        ResponseFormat::Text => Ok(None),
        ResponseFormat::JsonObject => Ok(Some(StructuredOutputsParams {
            json_object: Some(true),
            ..Default::default()
        })),
        ResponseFormat::JsonSchema { json_schema } => Ok(Some(StructuredOutputsParams {
            json: Some(json_schema.schema.clone()),
            ..Default::default()
        })),
        ResponseFormat::StructuralTag { .. } => {
            // The Python frontend dumps the entire response_format object (including the
            // `type` field) as a JSON string for the engine-core backend.
            let tag_json = serde_json::to_string(fmt).map_err(|e| {
                ApiError::invalid_request(
                    format!("failed to serialize structural_tag: {e}"),
                    Some("response_format"),
                )
            })?;
            Ok(Some(StructuredOutputsParams {
                structural_tag: Some(tag_json),
                ..Default::default()
            }))
        }
    }
}

/// Convert raw `response_format` and/or `structured_outputs` JSON blobs into
/// engine-core [`StructuredOutputsParams`].
///
/// Used by the completions endpoint which keeps both fields as opaque
/// `serde_json::Value`.
pub fn convert_from_response_format_value(
    response_format: &Option<serde_json::Value>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputsParams>, ApiError> {
    if let Some(raw) = structured_outputs {
        return Ok(Some(deserialize_structured_outputs(raw)?));
    }

    let Some(raw) = response_format else {
        return Ok(None);
    };

    // Deserialize into our typed enum and delegate.
    let fmt: ResponseFormat = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid response_format: {e}"),
            Some("response_format"),
        )
    })?;
    convert_from_response_format(Some(&fmt), &None)
}
