use serde_json::json;

use super::{Result, ToolParser, ToolParserOutput};
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
        parser.parse_into(chunk, &mut output).unwrap();
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

/// Assert every tested streaming chunking matches whole-text parsing.
///
/// The comparison is made after `finish()` and `coalesce()`, so parsers may
/// still emit different intermediate deltas for different chunk boundaries.
pub fn assert_streaming_matches_complete<P>(tools: &[Tool], case_name: &str, text: &str)
where
    P: ToolParser + 'static,
{
    let complete = parse_complete_with_parser::<P>(tools, text);
    assert_chunking_matches::<P>(tools, case_name, text, &[text], &complete);

    let boundaries = internal_char_boundaries(text);
    for boundary in &boundaries {
        let chunks = [&text[..*boundary], &text[*boundary..]];
        assert_chunking_matches::<P>(tools, case_name, text, &chunks, &complete);
    }

    for window in boundaries.windows(2) {
        let left = window[0];
        let right = window[1];
        let chunks = [&text[..left], &text[left..right], &text[right..]];
        assert_chunking_matches::<P>(tools, case_name, text, &chunks, &complete);
    }
}

fn internal_char_boundaries(text: &str) -> Vec<usize> {
    text.char_indices().map(|(index, _)| index).filter(|index| *index > 0).collect()
}

fn parse_complete_with_parser<P>(tools: &[Tool], text: &str) -> Result<ToolParserOutput>
where
    P: ToolParser + 'static,
{
    let mut parser = P::create(tools).expect("parser should initialize");
    let mut output = ToolParserOutput::default();
    parser.parse_into(text, &mut output)?;
    output.append(parser.finish()?);
    Ok(output.coalesce())
}

fn assert_chunking_matches<P>(
    tools: &[Tool],
    case_name: &str,
    text: &str,
    chunks: &[&str],
    complete: &Result<ToolParserOutput>,
) where
    P: ToolParser + 'static,
{
    let mut parser = P::create(tools).expect("parser should initialize");
    let streamed = collect_stream_result(&mut *parser, chunks);

    match (&streamed, complete) {
        (Ok(streamed), Ok(complete)) => assert_eq!(
            streamed, complete,
            "streaming output differed for case {case_name:?}, chunks {chunks:?}, text {text:?}",
        ),
        (Err(_), Err(_)) => {}
        (Ok(streamed), Err(error)) => panic!(
            "streaming succeeded but complete parsing failed for case {case_name:?}, \
             chunks {chunks:?}, text {text:?}, streamed {streamed:?}, complete error {error:?}",
        ),
        (Err(error), Ok(complete)) => panic!(
            "streaming failed but complete parsing succeeded for case {case_name:?}, \
             chunks {chunks:?}, text {text:?}, streaming error {error:?}, complete {complete:?}",
        ),
    }
}

fn collect_stream_result<T: ToolParser + ?Sized>(
    parser: &mut T,
    chunks: &[&str],
) -> Result<ToolParserOutput> {
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output)?;
    }
    output.append(parser.finish()?);
    Ok(output.coalesce())
}
