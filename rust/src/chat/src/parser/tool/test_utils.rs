use serde_json::json;

use super::{ToolParseResult, ToolParser};
use crate::request::ChatTool;

/// Build a reusable set of function tools for parser unit tests.
pub fn test_tools() -> Vec<ChatTool> {
    vec![
        ChatTool {
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
        ChatTool {
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
        ChatTool {
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
        ChatTool {
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
        ChatTool {
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
pub fn collect_stream<T: ToolParser + ?Sized>(parser: &mut T, chunks: &[&str]) -> ToolParseResult {
    let mut result = ToolParseResult::default();
    for chunk in chunks {
        result.append(parser.push(chunk).unwrap());
    }
    result.append(parser.finish().unwrap());
    result.coalesce_calls()
}

/// Split text into chunks containing at most `chunk_chars` Unicode scalar values.
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
