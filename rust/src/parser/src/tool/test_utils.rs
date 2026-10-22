// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::json;
use vllm_tokenizer::DecodedText;
#[cfg(test)]
use vllm_tokenizer::{DynTokenizer, Tokenizer};

use super::{ToolParser, ToolParserOutput};
use crate::tool::Tool;

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
            defer_loading: None,
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
            defer_loading: None,
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
            defer_loading: None,
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
            defer_loading: None,
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
            defer_loading: None,
        },
    ]
}

/// A tokenizer for parsers that do not resolve any tokens.
#[cfg(test)]
pub fn test_tokenizer() -> DynTokenizer {
    std::sync::Arc::new(vllm_tokenizer::test_utils::TestTokenizer::new())
}

/// Push chunks through a streaming parser and coalesce its tool-call deltas.
///
/// Panics if there are any parsing errors along the way.
pub fn collect_stream<T: ToolParser + ?Sized>(parser: &mut T, chunks: &[&str]) -> ToolParserOutput {
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(DecodedText::unattributed(*chunk), &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    output.coalesce()
}

/// Run token IDs through the real incremental decoder to obtain attributed
/// output, as the frontend does.
#[cfg(test)]
pub fn decode_ids(tokenizer: &dyn Tokenizer, ids: &[u32]) -> DecodedText {
    let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
    for &id in ids {
        decoder.push_token(id).unwrap();
    }
    decoder.flush(None).unwrap().1
}

/// Push chunks through a streaming parser as the frontend delivers them, and
/// coalesce its tool-call deltas.
///
/// The concatenated chunks are encoded with the added tokens of `tokenizer`
/// and decoded once, then split at the chunk boundaries, so a marker split
/// across chunks keeps its token anchor.
///
/// Panics if there are any parsing errors along the way.
#[cfg(test)]
pub fn collect_attributed_stream<T: ToolParser + ?Sized>(
    parser: &mut T,
    tokenizer: &dyn Tokenizer,
    chunks: &[&str],
) -> ToolParserOutput {
    let mut rest = decode_ids(
        tokenizer,
        &tokenizer.encode(&chunks.concat(), false).unwrap(),
    );
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(rest.drain_prefix(chunk.len()), &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    output.coalesce()
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
