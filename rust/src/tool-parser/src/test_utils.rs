use serde_json::json;

use super::{ToolParser, ToolParserOutput};
use crate::{Tool, ToolParserTestExt as _};

/// Build a reusable set of function tools for parser unit tests.
pub fn test_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "city": { "type": "string" },
                    "state": { "type": "string" },
                    "unit": { "type": "string" },
                    "date": { "type": "string" },
                    "days": { "type": "integer" }
                }
            }),
            strict: None,
        },
        Tool {
            name: "add".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "integer" },
                    "y": { "type": "integer" }
                }
            }),
            strict: None,
        },
        Tool {
            name: "convert".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "whole": { "type": "number" },
                    "flag": { "type": "boolean" },
                    "payload": { "type": "object" },
                    "items": { "type": "array" },
                    "empty": { "type": "string" }
                }
            }),
            strict: None,
        },
        Tool {
            name: "calculate_area".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "shape": { "type": "string" },
                    "dimensions": { "type": "object" },
                    "precision": { "type": "integer" }
                }
            }),
            strict: None,
        },
        Tool {
            name: "update_record".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "data": {
                        "anyOf": [
                            { "type": "object" },
                            { "type": "null" }
                        ]
                    }
                }
            }),
            strict: None,
        },
    ]
}

/// Push chunks through a streaming parser and coalesce its tool-call deltas.
///
/// Panics if there are any parsing errors along the way.
pub fn collect_stream<T: ToolParser + ?Sized>(parser: &mut T, chunks: &[&str]) -> ToolParserOutput {
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        output.append(parser.parse_chunk(chunk).unwrap());
    }
    output.append(parser.finish().unwrap());
    output.coalesce_calls()
}

/// Split text into chunks containing at most `chunk_chars` Unicode scalar
/// values.
pub fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut count = 0;

    for (index, _) in text.char_indices() {
        if count == chunk_chars {
            chunks.push(&text[start..index]);
            start = index;
            count = 0;
        }
        count += 1;
    }

    if start < text.len() {
        chunks.push(&text[start..]);
    }

    chunks
}
