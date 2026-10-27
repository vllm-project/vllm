// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::mem::discriminant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams;

use crate::error::{ApiError, bail_invalid_request};

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
/// <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/generate/base/protocol.py#L71-L112>
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
    /// `StructuredOutputConstraint::StructuralTag`.
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
/// A non-`text` `response_format` overrides the constraint of the same kind in
/// `structured_outputs` and keeps its options; a constraint of another kind is
/// rejected, since a request may use only one.
///
/// Mirrors the Python vLLM conversion in
/// `structured_outputs_from_response_format()`: <https://github.com/vllm-project/vllm/blob/7b569ec895/vllm/entrypoints/generate/base/protocol.py#L181-L210>
pub fn convert_from_response_format(
    response_format: Option<&ResponseFormat>,
    structured_outputs: &Option<serde_json::Value>,
) -> Result<Option<StructuredOutputsParams>, ApiError> {
    let structured_outputs =
        structured_outputs.as_ref().map(deserialize_structured_outputs).transpose()?;

    let Some(fmt) = response_format else {
        return Ok(structured_outputs);
    };
    let format_params = match fmt {
        ResponseFormat::Text => return Ok(structured_outputs),
        ResponseFormat::JsonObject => StructuredOutputsParams::json_object(),
        ResponseFormat::JsonSchema { json_schema } => {
            StructuredOutputsParams::json(json_schema.schema.clone())
        }
        ResponseFormat::StructuralTag { .. } => {
            // The Python frontend dumps the entire response_format object (including the
            // `type` field) as a JSON string for the engine-core backend.
            let tag_json = serde_json::to_string(fmt).map_err(|e| {
                ApiError::invalid_request(
                    format!("failed to serialize structural_tag: {e}"),
                    Some("response_format"),
                )
            })?;
            StructuredOutputsParams::structural_tag(tag_json)
        }
    };
    let params = match structured_outputs {
        None => format_params,
        Some(params)
            if discriminant(&params.constraint) == discriminant(&format_params.constraint) =>
        {
            StructuredOutputsParams {
                constraint: format_params.constraint,
                ..params
            }
        }
        Some(_) => bail_invalid_request!(
            "You can only use one kind of structured outputs constraint but multiple are \
             specified."
        ),
    };
    params.validate().map_err(|error| {
        ApiError::invalid_request(error.to_report_string(), Some("response_format"))
    })?;
    Ok(Some(params))
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
    let Some(raw) = response_format else {
        return convert_from_response_format(None, structured_outputs);
    };

    // Deserialize into our typed enum and delegate.
    let fmt: ResponseFormat = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid response_format: {e}"),
            Some("response_format"),
        )
    })?;
    convert_from_response_format(Some(&fmt), structured_outputs)
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;
    use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputConstraint;

    use super::*;

    #[test]
    fn response_format_overrides_same_kind_structured_outputs_and_keeps_options() {
        let params = convert_from_response_format_value(
            &Some(json!({
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "integer"}}
            })),
            &Some(json!({"json": {"type": "string"}, "disable_any_whitespace": true})),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Json(json!({"type": "integer"}))
        );
        assert!(params.options.disable_any_whitespace);
    }

    #[test]
    fn text_response_format_keeps_structured_outputs() {
        let params = convert_from_response_format_value(
            &Some(json!({"type": "text"})),
            &Some(json!({"regex": "a+"})),
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Regex("a+".to_string())
        );
    }

    #[test]
    fn response_format_rejects_structured_outputs_of_another_kind() {
        let error = convert_from_response_format_value(
            &Some(json!({"type": "json_object"})),
            &Some(json!({"regex": "a+"})),
        )
        .unwrap_err();

        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error.to_error_response().error.message,
            "You can only use one kind of structured outputs constraint but multiple are \
             specified."
        );
    }
}
