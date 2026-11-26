// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;

use vllm_parser::tool::test_utils::split_by_chars;
use vllm_parser::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
use vllm_tokenizer::{DecodedText, Tokenizer};

/// Split `text` into token-attributed chunks of at most `chunk_chars`
/// characters, as the frontend streams them to a unified parser.
///
/// The whole text is encoded with special tokens and run through the
/// incremental decoder once before splitting, so a marker cut by a chunk
/// boundary keeps its special-token anchor. Prepare chunks outside the measured
/// routine.
pub fn attributed_chunks(
    tokenizer: &dyn Tokenizer,
    text: &str,
    chunk_chars: usize,
) -> Vec<DecodedText> {
    let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
    for token_id in tokenizer.encode(text, false).expect("fixture should encode") {
        decoder.push_token(token_id).expect("fixture token should decode");
    }
    let mut rest = decoder.flush(None).expect("decoder should flush").1;
    assert_eq!(
        rest.text, text,
        "fixture should round-trip through the tokenizer"
    );

    split_by_chars(text, chunk_chars)
        .into_iter()
        .map(|chunk| rest.drain_prefix(chunk.len()))
        .collect()
}

/// Event totals of one parsed stream, for benchmark sanity checks.
#[derive(Debug, Default)]
pub struct UnifiedStreamSummary {
    pub normal_text: String,
    pub reasoning_text: String,
    pub calls_len: usize,
}

/// Run one request stream through `parser`: initialize it from the prompt,
/// feed every chunk, and finish.
pub fn feed_unified_parser(
    parser: &mut dyn UnifiedParser,
    prompt_token_ids: &[u32],
    chunks: Vec<DecodedText>,
) -> UnifiedStreamSummary {
    parser.initialize(prompt_token_ids).expect("parser should initialize");

    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser.parse_into(chunk, &mut output).expect("chunk should parse");
    }
    output.append(parser.finish().expect("stream should finish"));

    let mut summary = UnifiedStreamSummary::default();
    let mut tool_indices = BTreeSet::new();
    for event in output.events {
        match event {
            UnifiedParserEvent::Text(text) => summary.normal_text.push_str(&text),
            UnifiedParserEvent::Reasoning(reasoning) => {
                summary.reasoning_text.push_str(&reasoning.text);
            }
            UnifiedParserEvent::ToolCall(call) => {
                tool_indices.insert(call.tool_index);
            }
        }
    }
    summary.calls_len = tool_indices.len();
    summary
}
