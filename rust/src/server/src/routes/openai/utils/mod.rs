// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod logprobs;
pub mod structured_outputs;
pub mod types;
pub mod usage;
pub mod validated_json;

use vllm_text::{PromptTruncation, TruncationSide};

use crate::error::{ApiError, bail_invalid_request};

pub(crate) fn validate_generation_prompt_truncation(
    limit: Option<i64>,
    echo: bool,
) -> Result<(), ApiError> {
    if let Some(limit) = limit
        && limit < -1
    {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens must be >= -1."
        );
    }
    if echo && limit.is_some() {
        bail_invalid_request!(
            param = "echo",
            "`echo=true` is not supported with `truncate_prompt_tokens`."
        );
    }
    Ok(())
}

pub(crate) fn resolve_generation_prompt_truncation(
    limit: Option<i64>,
    side: Option<TruncationSide>,
) -> vllm_text::Result<Option<PromptTruncation>> {
    limit
        .map(|limit| PromptTruncation::from_wire(limit, side.unwrap_or(TruncationSide::Left)))
        .transpose()
}

/// Validate the effective KV metadata, including top-level override precedence.
pub(crate) fn validate_inline_hidden_states(
    kv_params: Option<&std::collections::HashMap<String, serde_json::Value>>,
    xargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<bool, ApiError> {
    let nested = xargs.and_then(|args| args.get("kv_transfer_params"));
    let get = |key: &str| match kv_params {
        Some(params) => params.get(key),
        None => nested.and_then(|params| params.get(key)),
    };
    if kv_params.is_none() && nested.is_some_and(|p| !p.is_null() && !p.is_object()) {
        bail_invalid_request!(
            param = "kv_transfer_params",
            "kv_transfer_params must be an object."
        );
    }
    let inline = match get("return_inline") {
        None | Some(serde_json::Value::Bool(false)) => return Ok(false),
        Some(serde_json::Value::Bool(true)) => true,
        _ => bail_invalid_request!(
            param = "kv_transfer_params.return_inline",
            "return_inline must be a boolean."
        ),
    };
    if get("hidden_states_path").is_some()
        || get("include_output_tokens").is_some_and(|v| match v {
            serde_json::Value::Null => false,
            serde_json::Value::Bool(v) => *v,
            serde_json::Value::Number(v) => v.as_f64() != Some(0.0),
            serde_json::Value::String(v) => !v.is_empty(),
            serde_json::Value::Array(v) => !v.is_empty(),
            serde_json::Value::Object(v) => !v.is_empty(),
        })
    {
        bail_invalid_request!(
            param = "kv_transfer_params.return_inline",
            "return_inline conflicts with include_output_tokens/hidden_states_path."
        );
    }
    Ok(inline)
}

pub(crate) fn validate_inline_hidden_states_backend(
    kv_params: Option<&std::collections::HashMap<String, serde_json::Value>>,
    xargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
    state: &crate::state::AppState,
) -> Result<(), ApiError> {
    if validate_inline_hidden_states(kv_params, xargs)?
        && !state
            .engine_core_client()
            .ready_responses()
            .iter()
            .all(|r| r.supports_inline_hidden_states)
    {
        bail_invalid_request!(
            param = "kv_transfer_params.return_inline",
            "return_inline is not supported by this engine configuration."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_inline_hidden_states;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn inline_hidden_states_validate_effective_metadata() {
        for (params, valid) in [
            (json!({"return_inline": true}), true),
            (json!({"return_inline": "true"}), false),
            (json!({"return_inline": null}), false),
            (
                json!({"return_inline": true, "include_output_tokens": true}),
                false,
            ),
            (
                json!({"return_inline": true, "hidden_states_path": null}),
                false,
            ),
            (
                json!({"return_inline": true, "include_output_tokens": false}),
                true,
            ),
        ] {
            let xargs = HashMap::from([("kv_transfer_params".to_string(), params.clone())]);
            let top = serde_json::from_value(params).unwrap();
            assert_eq!(
                validate_inline_hidden_states(None, Some(&xargs)).is_ok(),
                valid
            );
            assert_eq!(
                validate_inline_hidden_states(Some(&top), None).is_ok(),
                valid
            );
            assert_eq!(
                validate_inline_hidden_states(Some(&HashMap::new()), Some(&xargs)).unwrap(),
                false
            );
        }
    }
}
