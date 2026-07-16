// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::ModalResult;
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use vllm_tokenizer::DynTokenizer;

use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::tool::json::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    tool_call_header_event,
};
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{
    JsonObjectScanState, parse_buffered_event, safe_text_len_mul, take_json_object,
};

const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const END_MESSAGE: &str = "<|end_message|>";

const IDLE_MARKERS: &[&str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TOOL_ERROR,
    END_MESSAGE,
];
const BLOCK_END_MARKERS: &[&str] = &[END_MESSAGE, CONTENT_MODEL_END_SAMPLING];

const INKLING_TOOL_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Inkling",
    start_marker: CONTENT_INVOKE_TOOL_JSON,
    end_marker: END_MESSAGE,
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["args"],
};

type InklingInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum InklingEvent {
    Text { len: usize },
    Reasoning { len: usize },
    TextStart,
    ReasoningStart,
    MessageStart,
    Header,
    ToolJsonStart,
    ToolJsonHeader { name: String },
    ToolJsonArgs { len: usize, complete: bool },
    BlockEnd,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum InklingMode {
    #[default]
    Idle,
    MessageHeader,
    Text,
    Reasoning,
    ToolJsonHeader,
    ToolJsonArgs {
        json_scan: JsonObjectScanState,
    },
    ToolJsonClose,
}

/// Unified parser for Inkling typed content blocks.
pub struct InklingUnifiedParser {
    buffer: String,
    mode: InklingMode,
    emitted_tool_count: usize,
    active_tool_index: Option<usize>,
    tokenizer: DynTokenizer,
    message_model_token_id: u32,
    content_text_token_id: u32,
    content_thinking_token_id: u32,
}

impl InklingUnifiedParser {
    /// Create a Inkling parser.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let message_model_token_id = token_id(tokenizer.as_ref(), MESSAGE_MODEL)?;
        let content_text_token_id = token_id(tokenizer.as_ref(), CONTENT_TEXT)?;
        let content_thinking_token_id = token_id(tokenizer.as_ref(), CONTENT_THINKING)?;

        Ok(Self {
            buffer: String::new(),
            mode: InklingMode::Idle,
            emitted_tool_count: 0,
            active_tool_index: None,
            tokenizer,
            message_model_token_id,
            content_text_token_id,
            content_thinking_token_id,
        })
    }

    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = InklingMode::Idle;
        for token_id in prompt_token_ids.iter().rev().copied() {
            if token_id == self.message_model_token_id {
                self.mode = InklingMode::MessageHeader;
                return;
            }
            if token_id == self.content_thinking_token_id {
                self.mode = InklingMode::Reasoning;
                return;
            }
            if token_id == self.content_text_token_id {
                self.mode = InklingMode::Text;
                return;
            }
            if self.tokenizer.is_special_id(token_id) {
                return;
            }
        }
    }

    fn apply_event(&mut self, event: InklingEvent, output: &mut UnifiedParserOutput) -> Result<()> {
        match event {
            InklingEvent::Text { len } => output.push_text(self.buffer[..len].to_string()),
            InklingEvent::Reasoning { len } => {
                output.push_reasoning(self.buffer[..len].to_string());
            }
            InklingEvent::MessageStart => self.mode = InklingMode::MessageHeader,
            InklingEvent::Header => {}
            InklingEvent::TextStart => self.mode = InklingMode::Text,
            InklingEvent::ReasoningStart => self.mode = InklingMode::Reasoning,
            InklingEvent::ToolJsonStart => self.mode = InklingMode::ToolJsonHeader,
            InklingEvent::ToolJsonHeader { name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = InklingMode::ToolJsonArgs {
                    json_scan: JsonObjectScanState::default(),
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments: String::new(),
                });
            }
            InklingEvent::ToolJsonArgs { len, complete } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Inkling arguments without an active tool call"
                    ));
                };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..len].to_string(),
                });
                if complete {
                    self.mode = InklingMode::ToolJsonClose;
                }
            }
            InklingEvent::BlockEnd => {
                self.mode = InklingMode::Idle;
                self.active_tool_index = None;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = InklingMode::Idle;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl UnifiedParser for InklingUnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.emitted_tool_count = 0;
        self.active_tool_index = None;
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_inkling_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            InklingMode::Idle | InklingMode::Text => {
                output.push_text(std::mem::take(&mut self.buffer))
            }
            InklingMode::MessageHeader => self.buffer.clear(),
            InklingMode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            InklingMode::ToolJsonHeader
            | InklingMode::ToolJsonArgs { .. }
            | InklingMode::ToolJsonClose => {
                return Err(parsing_failed!("incomplete Inkling tool call"));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        InklingUnifiedParser::reset(self)
    }
}

/// Parse one Inkling event from buffered streaming input.
fn parse_next_inkling_event(
    input: &mut InklingInput<'_>,
    mode: &mut InklingMode,
) -> ModalResult<InklingEvent> {
    match mode {
        InklingMode::Idle => parse_idle_event(input),
        InklingMode::MessageHeader => parse_message_header_event(input),
        InklingMode::Text => parse_text_event(input),
        InklingMode::Reasoning => parse_reasoning_event(input),
        InklingMode::ToolJsonHeader => parse_tool_json_header_event(input),
        InklingMode::ToolJsonArgs { json_scan } => parse_tool_json_args_event(input, json_scan),
        InklingMode::ToolJsonClose => parse_tool_json_close_event(input),
    }
}

/// Parse an event while waiting for a Inkling content kind.
fn parse_idle_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        message_start_event,
        reasoning_start_event,
        text_start_event,
        tool_json_start_event,
        raw_text_start_event,
        block_end_event,
        safe_idle_text_event,
    ))
    .parse_next(input)
}

/// Parse a Inkling model-authored message start marker.
fn message_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(MESSAGE_MODEL).value(InklingEvent::MessageStart).parse_next(input)
}

/// Parse an event while waiting for an Inkling message content kind.
fn parse_message_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        reasoning_start_event,
        text_start_event,
        tool_json_start_event,
        raw_text_start_event,
        block_end_event,
        safe_header_event,
    ))
    .parse_next(input)
}

/// Parse an event inside a Inkling text block.
fn parse_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((block_end_event, safe_text_event)).parse_next(input)
}

/// Parse an event inside a Inkling reasoning block.
fn parse_reasoning_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((block_end_event, safe_reasoning_event)).parse_next(input)
}

/// Parse a Inkling text start marker.
fn text_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_TEXT).value(InklingEvent::TextStart).parse_next(input)
}

/// Parse a Inkling reasoning start marker.
fn reasoning_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_THINKING).value(InklingEvent::ReasoningStart).parse_next(input)
}

/// Parse a Inkling JSON tool-call start marker.
fn tool_json_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    literal(CONTENT_INVOKE_TOOL_JSON)
        .value(InklingEvent::ToolJsonStart)
        .parse_next(input)
}

/// Parse a Inkling content kind treated as visible text.
fn raw_text_start_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((
        literal(CONTENT_INVOKE_TOOL_TEXT),
        literal(CONTENT_TOOL_ERROR),
    ))
    .value(InklingEvent::TextStart)
    .parse_next(input)
}

/// Parse a Inkling block end marker.
fn block_end_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    alt((literal(END_MESSAGE), literal(CONTENT_MODEL_END_SAMPLING)))
        .value(InklingEvent::BlockEnd)
        .parse_next(input)
}

/// Parse safe text while waiting for the next Inkling marker.
fn safe_idle_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|len| InklingEvent::Text { len })
}

/// Parse safe header text before the next Inkling marker.
fn safe_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, IDLE_MARKERS).map(|_| InklingEvent::Header)
}

/// Parse safe text before the end of a Inkling text block.
fn safe_text_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, BLOCK_END_MARKERS).map(|len| InklingEvent::Text { len })
}

/// Parse safe reasoning before the end of a Inkling reasoning block.
fn safe_reasoning_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    safe_text_len_mul(input, BLOCK_END_MARKERS).map(|len| InklingEvent::Reasoning { len })
}

/// Parse a Inkling JSON tool-call header.
fn parse_tool_json_header_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    match tool_call_header_event(input, INKLING_TOOL_CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => Ok(InklingEvent::ToolJsonHeader {
            name: function_name,
        }),
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse raw Inkling JSON tool-call argument bytes.
fn parse_tool_json_args_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<InklingEvent> {
    let len = take_json_object(input, json_scan)?;
    Ok(InklingEvent::ToolJsonArgs {
        len,
        complete: json_scan.complete(),
    })
}

/// Parse the close of a Inkling JSON tool-call block.
fn parse_tool_json_close_event(input: &mut InklingInput<'_>) -> ModalResult<InklingEvent> {
    seq!(
        _: ws0,
        _: literal("}"),
        _: ws0,
        _: literal(END_MESSAGE),
    )
    .value(InklingEvent::BlockEnd)
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{CONTENT_TEXT, CONTENT_THINKING, InklingUnifiedParser, MESSAGE_MODEL};
    use crate::tool::Tool;
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
    use thiserror_ext::AsReport;
    use vllm_tokenizer::Tokenizer;

    struct FakeTokenizer;

    impl Tokenizer for FakeTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> vllm_tokenizer::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_tokenizer::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                MESSAGE_MODEL => Some(200001),
                "<|content_text|>" => Some(200004),
                "<|content_model_end_sampling|>" => Some(200006),
                "<|content_thinking|>" => Some(200008),
                "<|end_message|>" => Some(200010),
                "<|content_tool_error|>" => Some(200022),
                "<|content_invoke_tool_json|>" => Some(200049),
                "<|content_invoke_tool_text|>" => Some(200057),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            let token = match id {
                200001 => MESSAGE_MODEL,
                200004 => "<|content_text|>",
                200006 => "<|content_model_end_sampling|>",
                200008 => "<|content_thinking|>",
                200010 => "<|end_message|>",
                200022 => "<|content_tool_error|>",
                200049 => "<|content_invoke_tool_json|>",
                200057 => "<|content_invoke_tool_text|>",
                _ => return None,
            };
            Some(token.to_string())
        }

        fn is_special_id(&self, token_id: u32) -> bool {
            (199999..=200057).contains(&token_id)
        }
    }

    trait UnifiedParserTestExt {
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput>;
        fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput>;
    }

    impl<T: UnifiedParser + ?Sized> UnifiedParserTestExt for T {
        fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            self.parse_into(chunk, &mut output)?;
            Ok(output)
        }

        fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput> {
            let mut output = self.parse_chunk(text)?;
            output.append(self.finish()?);
            Ok(output)
        }
    }

    trait UnifiedParserOutputTestExt {
        fn normal_text(&self) -> String;
        fn reasoning_text(&self) -> String;
        fn calls(&self) -> Vec<crate::tool::ToolCallDelta>;
    }

    impl UnifiedParserOutputTestExt for UnifiedParserOutput {
        fn normal_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect()
        }

        fn reasoning_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Reasoning(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect()
        }

        fn calls(&self) -> Vec<crate::tool::ToolCallDelta> {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect()
        }
    }

    fn test_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
            }),
            strict: None,
        }]
    }

    fn test_parser() -> InklingUnifiedParser {
        InklingUnifiedParser::new(&test_tools(), Arc::new(FakeTokenizer)).unwrap()
    }

    fn collect_stream(chunks: &[&str]) -> UnifiedParserOutput {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output
    }

    #[test]
    fn inkling_streaming_emits_reasoning_then_text() {
        let output = collect_stream(&[concat!(
            "<|content_thinking|>reason<|end_message|>",
            "<|message_model|><|content_text|>answer<|end_message|>",
            "<|content_model_end_sampling|>"
        )]);

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn inkling_streaming_holds_split_markers() {
        let output = collect_stream(&[
            "<|content_thin",
            "king|>rea",
            "son<|end_mes",
            "sage|><|message_",
            "model|><|content_text|>answer",
            "<|end_message|>",
        ]);

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn inkling_streaming_accepts_content_blocks_without_message_start() {
        let output = collect_stream(&[concat!(
            "<|content_thinking|>reason<|end_message|>",
            "<|content_text|>answer<|end_message|>",
        )]);

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn inkling_tool_json_streams_name_then_argument_deltas() {
        let mut parser = test_parser();
        let chunks = [
            "<|message_model|><|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":",
            "{\"city\":",
            "\"SF\"",
            "}}<|end_message|>",
        ];

        let mut output = UnifiedParserOutput::default();
        let mut observed_args = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_args.extend(
                next.calls()
                    .into_iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        let calls = output.calls();
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(observed_args, ["{\"city\":", "\"SF\"", "}"]);
        assert_eq!(
            calls.iter().map(|call| call.arguments.as_str()).collect::<String>(),
            "{\"city\":\"SF\"}"
        );
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn inkling_discards_tool_name_from_message_header() {
        let output = collect_stream(&[concat!(
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            "{\"name\":\"get_weather\",\"args\":{}}<|end_message|>"
        )]);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn inkling_streaming_handles_multiple_tool_blocks() {
        let output = collect_stream(&[concat!(
            "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}<|end_message|>",
            "<|message_model|><|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"NYC\"}}<|end_message|>"
        )]);

        let calls = output.calls();
        assert_eq!(calls.iter().filter(|call| call.name.is_some()).count(), 2);
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[2].tool_index, 1);
    }

    #[test]
    fn inkling_initialize_open_text_prompt_starts_in_text() {
        let mut parser = test_parser();
        parser.initialize(&[200004]).unwrap();

        let output = parser.parse_complete("answer<|end_message|>").unwrap();

        assert_eq!(output.normal_text(), "answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn inkling_initialize_open_reasoning_prompt_starts_in_reasoning() {
        let mut parser = test_parser();
        parser.initialize(&[200008]).unwrap();

        let output = parser.parse_complete("reason<|end_message|>").unwrap();

        assert_eq!(output.reasoning_text(), "reason");
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn inkling_initialize_model_opener_starts_in_message_header() {
        let mut parser = test_parser();
        parser.initialize(&[200001]).unwrap();

        let output = parser
            .parse_complete(concat!(
                "get_weather<|content_invoke_tool_json|>",
                "{\"name\":\"get_weather\",\"args\":{}}<|end_message|>"
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn inkling_plain_text_falls_through_as_text() {
        let output = collect_stream(&["plain ", "answer"]);

        assert_eq!(output.normal_text(), "plain answer");
        assert!(output.reasoning_text().is_empty());
    }

    #[test]
    fn inkling_raw_tool_text_and_tool_error_are_visible_text() {
        let output = collect_stream(&[concat!(
            "<|content_invoke_tool_text|>search SF<|end_message|>",
            "<|content_tool_error|>failed<|end_message|>"
        )]);

        assert_eq!(output.normal_text(), "search SFfailed");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn inkling_finish_fails_incomplete_tool_call() {
        let mut parser = test_parser();
        parser
            .parse_chunk("<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\"")
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert!(error.as_report().to_string().contains("incomplete Inkling tool call"));
    }

    #[test]
    fn inkling_rejects_non_object_args() {
        let mut parser = test_parser();
        let error = parser
            .parse_chunk("<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":42")
            .unwrap_err();

        assert!(error.as_report().to_string().contains("JSON object argument"));
    }

    #[test]
    fn inkling_missing_token_fails_create() {
        struct MissingTokenizer;

        impl Tokenizer for MissingTokenizer {
            fn encode(
                &self,
                _text: &str,
                _add_special_tokens: bool,
            ) -> vllm_tokenizer::Result<Vec<u32>> {
                Ok(vec![])
            }

            fn decode(
                &self,
                _token_ids: &[u32],
                _skip_special_tokens: bool,
            ) -> vllm_tokenizer::Result<String> {
                Ok(String::new())
            }

            fn token_to_id(&self, token: &str) -> Option<u32> {
                match token {
                    MESSAGE_MODEL => Some(200001),
                    CONTENT_TEXT => Some(200004),
                    _ => None,
                }
            }

            fn id_to_token(&self, id: u32) -> Option<String> {
                match id {
                    200001 => Some(MESSAGE_MODEL.to_string()),
                    200004 => Some(CONTENT_TEXT.to_string()),
                    _ => None,
                }
            }
        }

        let error = match InklingUnifiedParser::new(&[], Arc::new(MissingTokenizer)) {
            Ok(_) => panic!("expected parser creation to fail"),
            Err(error) => error,
        };

        assert!(error.as_report().to_string().contains(CONTENT_THINKING));
    }
}
