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

/// Validate that exactly one structured-output constraint field is set.
///
/// Mirrors the mutual-exclusivity check in Python's
/// `StructuredOutputsParams.__post_init__`:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/sampling_params.py#L90-L111>
fn validate_single_constraint(params: &StructuredOutputsParams) -> Result<(), ApiError> {
    let count = [
        params.json.is_some(),
        params.regex.is_some(),
        params.choice.is_some(),
        params.grammar.is_some(),
        params.json_object.is_some(),
        params.structural_tag.is_some(),
    ]
    .into_iter()
    .filter(|set| *set)
    .count();
    match count {
        1 => Ok(()),
        0 => Err(ApiError::invalid_request(
            "You must use one kind of structured outputs constraint but none \
             are specified",
            Some("structured_outputs"),
        )),
        _ => Err(ApiError::invalid_request(
            "You can only use one kind of structured outputs constraint but \
             multiple are specified",
            Some("structured_outputs"),
        )),
    }
}

/// Validate that a `structural_tag` payload matches one of the two shapes vLLM
/// accepts, mirroring Python's `AnyStructuralTagResponseFormat`:
///
/// - legacy: `structures` (array) and `triggers` (array)
/// - current: `format`
///
/// Python validates the shape at request time so malformed tags are rejected
/// as bad requests rather than surfacing later as opaque generation failures:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026/vllm/entrypoints/openai/engine/protocol.py#L167-L197>
fn validate_structural_tag_shape(extra: &serde_json::Map<String, Value>) -> Result<(), ApiError> {
    let invalid = || {
        ApiError::invalid_request(
            "Invalid response_format structural_tag specification",
            Some("response_format"),
        )
    };

    // Current shape: validate the `format` payload, not just its presence.
    // Python runs the xgrammar structural-tag validator here, so a malformed
    // tag such as `{"type":"structural_tag","format":{}}` is a 400 rather than
    // an opaque engine-core generation failure. We mirror that by deserializing
    // into the typed `xgrammar_structural_tag::StructuralTag`, which fails for
    // an unknown/empty `format`.
    if extra.contains_key("format") {
        let mut object = extra.clone();
        // `type` was consumed as the enum discriminator, so re-insert it for
        // the typed round-trip.
        object
            .entry("type".to_string())
            .or_insert_with(|| Value::from("structural_tag"));
        return serde_json::from_value::<xgrammar_structural_tag::StructuralTag>(Value::Object(
            object,
        ))
        .map(|_| ())
        .map_err(|_| invalid());
    }
    let legacy_shaped = extra.get("structures").is_some_and(Value::is_array)
        && extra.get("triggers").is_some_and(Value::is_array);
    if legacy_shaped {
        return Ok(());
    }
    Err(invalid())
}

/// Convert an explicit `structured_outputs` JSON blob into
/// [`StructuredOutputsParams`].
fn deserialize_structured_outputs(
    raw: &serde_json::Value,
) -> Result<StructuredOutputsParams, ApiError> {
    let params: StructuredOutputsParams = serde_json::from_value(raw.clone()).map_err(|e| {
        ApiError::invalid_request(
            format!("invalid structured_outputs: {e}"),
            Some("structured_outputs"),
        )
    })?;
    validate_single_constraint(&params)?;
    Ok(params)
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
        ResponseFormat::JsonSchema { json_schema } => {
            // Python maps a null schema to `json=None`, which then fails the
            // mutual-exclusivity check with "you must use one kind of
            // constraint". Mirror that by rejecting a null schema here rather
            // than forwarding `json=null` to engine-core.
            if json_schema.schema.is_null() {
                return Err(ApiError::invalid_request(
                    "You must use one kind of structured outputs constraint \
                     but none are specified",
                    Some("response_format"),
                ));
            }
            Ok(Some(StructuredOutputsParams {
                json: Some(json_schema.schema.clone()),
                ..Default::default()
            }))
        }
        ResponseFormat::StructuralTag { extra } => {
            validate_structural_tag_shape(extra)?;
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::*;

    /// Generate arbitrary, possibly-adversarial JSON values: deep nesting,
    /// mixed types, unicode keys, and numeric edge cases. Bounded so shrinking
    /// terminates quickly.
    fn arb_json() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            any::<f64>().prop_filter("finite", |f| f.is_finite()).prop_map(Value::from),
            ".*".prop_map(Value::from),
        ];
        leaf.prop_recursive(4, 32, 6, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..6).prop_map(Value::from),
                prop::collection::hash_map(".*", inner, 0..6)
                    .prop_map(|m| Value::Object(m.into_iter().collect())),
            ]
        })
    }

    proptest! {
        /// `convert_from_response_format_value` must never panic on arbitrary
        /// input: it either returns parsed params or a clean `ApiError`.
        #[test]
        fn convert_never_panics_on_arbitrary_json(
            response_format in arb_json(),
            structured_outputs in arb_json(),
        ) {
            let _ = convert_from_response_format_value(
                &Some(response_format),
                &Some(structured_outputs.clone()),
            );
            // Also exercise the response_format-only path (structured_outputs
            // takes precedence above, so cover it independently).
            let _ = convert_from_response_format_value(&Some(structured_outputs), &None);
        }

        /// Deserializing a `structured_outputs` blob and re-serializing it must
        /// round-trip the constraint fields. The `_backend` field is the one
        /// documented exception: it is ignored on input and always normalized
        /// to `guidance`, so we assert that separately rather than expecting
        /// it to survive untouched.
        #[test]
        fn structured_outputs_roundtrip_normalizes_backend(schema in arb_json()) {
            let input = json!({ "json": schema });
            let params: StructuredOutputsParams =
                serde_json::from_value(input.clone()).unwrap();
            let reserialized = serde_json::to_value(&params).unwrap();

            // The schema we put in must come back out byte-for-byte.
            prop_assert_eq!(&reserialized["json"], &input["json"]);
            // Backend is always normalized regardless of input.
            prop_assert_eq!(&reserialized["_backend"], &json!("guidance"));
        }
    }

    /// Mirroring Python, a blob naming several constraints must be rejected
    /// (mutual exclusivity), matching `StructuredOutputsParams.__post_init__`.
    #[test]
    fn multiple_constraints_rejected() {
        let blob = json!({
            "json": { "type": "object" },
            "regex": "[0-9]+",
            "choice": ["a", "b"],
        });
        let err = deserialize_structured_outputs(&blob).unwrap_err();
        assert!(err.to_error_response().error.message.contains("only use one"));
    }

    /// A blob naming no constraint is likewise rejected, matching Python's
    /// `count < 1` branch.
    #[test]
    fn no_constraint_rejected() {
        let err = deserialize_structured_outputs(&json!({})).unwrap_err();
        assert!(err.to_error_response().error.message.contains("must use one"));
    }

    /// A blob with exactly one constraint is accepted.
    #[test]
    fn single_constraint_accepted() {
        let params =
            deserialize_structured_outputs(&json!({ "json": { "type": "object" } })).unwrap();
        assert!(params.json.is_some());
    }

    /// A `json_schema` response format with a null schema is rejected, matching
    /// Python where the schema lowers to `json=None` and then fails the
    /// "must use one constraint" check (`sampling_params.py:107-111`).
    #[test]
    fn null_json_schema_rejected() {
        let fmt = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: "x".to_string(),
                description: None,
                schema: Value::Null,
                strict: None,
            },
        };
        let err = convert_from_response_format(Some(&fmt), &None).unwrap_err();
        assert!(err.to_error_response().error.message.contains("must use one"));
    }

    /// A `structural_tag` with neither the legacy (`structures`/`triggers`) nor
    /// the current (`format`) shape is rejected at request time, matching
    /// Python's `validate_structural_tag_response_format`.
    #[test]
    fn malformed_structural_tag_rejected() {
        for raw in [
            json!({ "type": "structural_tag" }),
            json!({ "type": "structural_tag", "structures": [] }),
            json!({ "type": "structural_tag", "triggers": [] }),
            // Present-but-malformed `format` payloads: Python rejects these via
            // the xgrammar validator, so key presence alone must not pass.
            json!({ "type": "structural_tag", "format": {} }),
            json!({ "type": "structural_tag", "format": { "type": "nope" } }),
        ] {
            let err = convert_from_response_format_value(&Some(raw.clone()), &None).unwrap_err();
            assert!(
                err.to_error_response().error.message.contains("structural_tag"),
                "expected rejection for {raw}"
            );
        }
    }

    /// Both accepted structural-tag shapes pass validation.
    #[test]
    fn well_formed_structural_tag_accepted() {
        // Current shape with a valid, typed `format` payload.
        let current = json!({ "type": "structural_tag", "format": { "type": "any_text" } });
        assert!(convert_from_response_format_value(&Some(current), &None).is_ok());

        let legacy = json!({
            "type": "structural_tag",
            "structures": [],
            "triggers": [],
        });
        assert!(convert_from_response_format_value(&Some(legacy), &None).is_ok());
    }
}
