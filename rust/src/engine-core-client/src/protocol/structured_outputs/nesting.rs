// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Nesting cap on structured-output JSON, mirrored from Python's
//! `check_json_nesting`.

use serde_json::Value;

use crate::error::{Error, Result};

/// Upper bound on nested objects and arrays in a structured-output JSON schema
/// or structural tag. EngineCore's grammar conversion cost grows steeply with
/// nesting. OpenAI allows 10 levels of schema nesting.
const MAX_JSON_NESTING: usize = 64;

/// Extra nesting allowed for a structural tag, which wraps its JSON schemas in
/// tag structure, such as the tool-call tags built by output parsers.
const STRUCTURAL_TAG_WRAPPER_NESTING: usize = 16;

/// Reject a JSON schema nested more than [`MAX_JSON_NESTING`] levels deep.
pub(super) fn check_json_schema(schema: &Value) -> Result<()> {
    ensure_within(exceeds(schema, MAX_JSON_NESTING))
}

/// Reject a structural tag nested more than [`MAX_JSON_NESTING`] plus
/// [`STRUCTURAL_TAG_WRAPPER_NESTING`] levels deep.
pub(super) fn check_structural_tag(tag: &Value) -> Result<()> {
    ensure_within(exceeds(
        tag,
        MAX_JSON_NESTING + STRUCTURAL_TAG_WRAPPER_NESTING,
    ))
}

fn ensure_within(exceeds: bool) -> Result<()> {
    if exceeds {
        return Err(Error::InvalidStructuredOutputsParams {
            message: format!(
                "The JSON schema is nested too deeply: objects and arrays may nest at most \
                 {MAX_JSON_NESTING} levels"
            ),
        });
    }
    Ok(())
}

/// Whether `value` nests objects and arrays deeper than `limit`.
fn exceeds(value: &Value, limit: usize) -> bool {
    let mut stack = vec![(value, 1)];
    while let Some((value, depth)) = stack.pop() {
        match value {
            Value::Object(_) | Value::Array(_) if depth > limit => return true,
            Value::Object(object) => stack.extend(object.values().map(|child| (child, depth + 1))),
            Value::Array(array) => stack.extend(array.iter().map(|child| (child, depth + 1))),
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A JSON document with `depth` nested containers, alternating objects and
    /// arrays.
    fn nested(depth: usize) -> Value {
        (0..depth).fold(json!("leaf"), |inner, level| {
            if level % 2 == 0 {
                json!({ "a": inner })
            } else {
                json!([inner])
            }
        })
    }

    #[test]
    fn exceeds_counts_objects_and_arrays() {
        for (depth, exceeds_limit) in [(0, false), (3, false), (4, true), (8, true)] {
            let value = nested(depth);
            assert_eq!(exceeds(&value, 3), exceeds_limit, "{value}");
        }
    }
}
