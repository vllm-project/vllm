//! Tests for the Kimi K3 unified parser, including the incremental tool-call
//! fragment contract (name first, argument fragments as they parse).

use std::sync::Arc;

use expect_test::expect;
use serde_json::{Value, json};
use thiserror_ext::AsReport;
use vllm_tokenizer::Tokenizer as _;
use vllm_tokenizer::test_utils::TestTokenizer;

use super::{
    END_OF_MSG, KimiK3UnifiedParser, OPEN, RESPONSE_CLOSE, RESPONSE_OPEN, SEP, THINK_CLOSE,
    THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};
use crate::tool::ToolCallDelta;
use crate::unified::{UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput};

const OPEN_ID: u32 = 256;
const CLOSE_ID: u32 = 257;
const SEP_ID: u32 = 258;
const END_OF_MSG_ID: u32 = 259;

fn tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_special_token(OPEN, OPEN_ID)
        .with_special_token("<|close|>", CLOSE_ID)
        .with_special_token(SEP, SEP_ID)
        .with_special_token(END_OF_MSG, END_OF_MSG_ID)
}

trait UnifiedParserTestExt {
    fn parse_chunk(&mut self, chunk: &str) -> super::Result<UnifiedParserOutput>;
    fn parse_complete(&mut self, text: &str) -> super::Result<UnifiedParserOutput>;
}

impl UnifiedParserTestExt for KimiK3UnifiedParser {
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

/// One tool call coalesced from its streamed fragments, mirroring how the
/// chat output layer concatenates argument deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CoalescedCall {
    tool_index: usize,
    name: String,
    arguments: String,
}

fn coalesced(tool_index: usize, name: &str, arguments: &str) -> CoalescedCall {
    CoalescedCall {
        tool_index,
        name: name.to_string(),
        arguments: arguments.to_string(),
    }
}

fn delta(tool_index: usize, name: Option<&str>, arguments: &str) -> ToolCallDelta {
    ToolCallDelta {
        tool_index,
        name: name.map(str::to_string),
        arguments: arguments.to_string(),
    }
}

trait UnifiedOutputTestExt {
    fn normal_text(&self) -> String;
    fn reasoning_text(&self) -> String;
    fn calls(&self) -> Vec<ToolCallDelta>;
    fn coalesced_calls(&self) -> Vec<CoalescedCall>;
}

impl UnifiedOutputTestExt for UnifiedParserOutput {
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

    fn calls(&self) -> Vec<ToolCallDelta> {
        self.events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }

    fn coalesced_calls(&self) -> Vec<CoalescedCall> {
        let mut calls = Vec::new();
        for delta in self.calls() {
            if let Some(name) = delta.name {
                assert_eq!(
                    delta.tool_index,
                    calls.len(),
                    "tool call indices must be dense per response"
                );
                calls.push(CoalescedCall {
                    tool_index: delta.tool_index,
                    name,
                    arguments: delta.arguments,
                });
            } else {
                let call = calls.last_mut().expect("arguments fragment before any call start");
                assert_eq!(
                    call.tool_index, delta.tool_index,
                    "arguments fragment for a different open call"
                );
                call.arguments.push_str(&delta.arguments);
            }
        }
        calls
    }
}

fn test_parser() -> KimiK3UnifiedParser {
    KimiK3UnifiedParser::new(&[], Arc::new(tokenizer())).unwrap()
}

fn collect_stream(parser: &mut KimiK3UnifiedParser, chunks: &[&str]) -> UnifiedParserOutput {
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        output.append(parser.parse_chunk(chunk).unwrap());
    }
    output.append(parser.finish().unwrap());
    output
}

/// Split `text` into small chunks to stress marker-split handling.
fn char_chunks(text: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars.chunks(size).map(|chunk| chunk.iter().collect()).collect()
}

fn arg_open(key: &str, arg_type: &str) -> String {
    format!("{OPEN}argument key=\"{key}\" type=\"{arg_type}\"{SEP}")
}

fn arg(key: &str, arg_type: &str, value: &str) -> String {
    format!(
        "{arg_open}{value}<|close|>argument{SEP}",
        arg_open = arg_open(key, arg_type)
    )
}

fn json_block(raw: &str) -> String {
    format!("{OPEN}json type=\"object\"{SEP}{raw}<|close|>json{SEP}")
}

fn call_open(attrs: &str) -> String {
    format!("{OPEN}call {attrs}{SEP}")
}

fn call_close() -> String {
    format!("<|close|>call{SEP}")
}

fn call(attrs: &str, body: &str) -> String {
    format!("{}{body}{}", call_open(attrs), call_close())
}

fn message_close() -> String {
    format!("<|close|>message{SEP}")
}

fn thinking_output(reasoning: &str, response: &str, tools_body: &str) -> String {
    let mut output = format!("{THINK_OPEN}{reasoning}{THINK_CLOSE}");
    output.push_str(&format!("{RESPONSE_OPEN}{response}{RESPONSE_CLOSE}"));
    if !tools_body.is_empty() {
        output.push_str(&format!("{TOOLS_OPEN}{tools_body}{TOOLS_CLOSE}"));
    }
    output.push_str(&message_close());
    output
}

fn first_call(output: &UnifiedParserOutput) -> CoalescedCall {
    output.coalesced_calls().into_iter().next().expect("expected one tool call")
}

#[test]
fn kimi_k3_create_requires_structural_tokens() {
    let error = match KimiK3UnifiedParser::new(&[], Arc::new(TestTokenizer::new())) {
        Ok(_) => panic!("expected missing token error"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        UnifiedParserError::MissingToken { token } if token == OPEN
    ));
}

#[test]
fn kimi_k3_parses_reasoning_response_and_typed_tool_call() {
    let body = [
        arg("city", "string", "Hangzhou"),
        arg("days", "number", "1.5"),
        arg("detailed", "boolean", "true"),
        arg("filters", "object", r#"{"kind":"rain"}"#),
        arg("hours", "array", "[8,20]"),
    ]
    .concat();
    let text = thinking_output(
        "Need the weather tool.",
        "I'll check.",
        &call("tool=\"get_weather\" index=\"1\"", &body),
    );

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(output.reasoning_text(), "Need the weather tool.");
    assert_eq!(output.normal_text(), "I'll check.");
    let call = first_call(&output);
    assert_eq!(call.tool_index, 0);
    assert_eq!(call.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(&call.arguments).unwrap(),
        json!({
            "city": "Hangzhou",
            "days": 1.5,
            "detailed": true,
            "filters": { "kind": "rain" },
            "hours": [8, 20],
        })
    );
}

#[test]
fn kimi_k3_tool_call_streams_fragments_incrementally() {
    let body = [
        arg("city", "string", "Hangzhou"),
        arg("days", "number", "1.5"),
    ]
    .concat();
    let text = thinking_output("t", "", &call("tool=\"get_weather\" index=\"1\"", &body));

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    // The streaming contract: the name first (as soon as the header parses),
    // the opening fragment when each argument begins, string value text as it
    // arrives, one fragment per buffered scalar at its block close, and the
    // closing brace when the call ends.
    expect![[r#"
        [
            ToolCallDelta {
                tool_index: 0,
                name: Some(
                    "get_weather",
                ),
                arguments: "",
            },
            ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "{\"city\":\"",
            },
            ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "Hangzhou",
            },
            ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "\"",
            },
            ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: ",\"days\":1.5",
            },
            ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "}",
            },
        ]
    "#]]
    .assert_debug_eq(&output.calls());
}

#[test]
fn kimi_k3_tool_call_streams_before_the_close_marker() {
    let mut parser = test_parser();

    let output = parser
        .parse_chunk(&format!(
            "{TOOLS_OPEN}{}",
            call_open("tool=\"calc\" index=\"1\"")
        ))
        .unwrap();
    // The function call is visible as soon as its header parses; no waiting
    // for the whole call (or even one argument) to complete.
    assert_eq!(output.calls(), [delta(0, Some("calc"), "")]);

    let output = parser.parse_chunk(&format!("{}Hel", arg_open("content", "string"))).unwrap();
    assert_eq!(
        output.calls(),
        [delta(0, None, "{\"content\":\""), delta(0, None, "Hel")]
    );

    let mut output = parser
        .parse_chunk(&format!("lo<|close|>argument{SEP}{}", call_close()))
        .unwrap();
    output.append(parser.finish().unwrap());
    assert_eq!(
        output.calls(),
        [
            delta(0, None, "lo"),
            delta(0, None, "\""),
            delta(0, None, "}")
        ]
    );
}

#[test]
fn kimi_k3_string_argument_streams_unicode_across_chunks() {
    let value = "Hángzhōu, €5 rain 🌧";
    let text = format!(
        "{TOOLS_OPEN}{}{TOOLS_CLOSE}{}",
        call(
            "tool=\"get_weather\" index=\"1\"",
            &arg("city", "string", value)
        ),
        message_close()
    );

    let whole = collect_stream(&mut test_parser(), &[&text]);
    let pieces = char_chunks(&text, 4);
    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let chunked = collect_stream(&mut test_parser(), &piece_refs);

    let expected = serde_json::to_string(&json!({ "city": value })).unwrap();
    assert_eq!(
        whole.coalesced_calls(),
        [coalesced(0, "get_weather", &expected)]
    );
    // Byte-exact and split-invariant: chunking only changes fragment counts.
    assert_eq!(chunked.coalesced_calls(), whole.coalesced_calls());
    assert!(chunked.calls().len() > whole.calls().len());
}

#[test]
fn kimi_k3_string_argument_escapes_are_split_invariant() {
    let value = "line one\nline \"two\" \\ end";
    let text = format!(
        "{TOOLS_OPEN}{}{TOOLS_CLOSE}{}",
        call(
            "tool=\"write\" index=\"1\"",
            &arg("content", "string", value)
        ),
        message_close()
    );

    let whole = collect_stream(&mut test_parser(), &[&text]);
    let pieces = char_chunks(&text, 2);
    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let chunked = collect_stream(&mut test_parser(), &piece_refs);

    assert_eq!(
        whole.coalesced_calls(),
        [coalesced(
            0,
            "write",
            &serde_json::to_string(&json!({ "content": value })).unwrap()
        )]
    );
    assert_eq!(chunked.coalesced_calls(), whole.coalesced_calls());
}

#[test]
fn kimi_k3_scalar_argument_buffers_until_block_close() {
    let mut parser = test_parser();
    let header = format!("{TOOLS_OPEN}{}", call_open("tool=\"calc\" index=\"1\""));
    parser.parse_chunk(&header).unwrap();

    // Partial scalars are never valid JSON, so a split scalar value emits
    // nothing until its block close parses.
    let output = parser.parse_chunk(&format!("{}4", arg_open("x", "number"))).unwrap();
    assert!(output.calls().is_empty(), "partial scalars must not stream");

    let output = parser.parse_chunk(&format!("2<|close|>argument{SEP}")).unwrap();
    // One fragment per closed scalar; the object's closing brace follows at
    // the call close.
    assert_eq!(output.calls(), [delta(0, None, "{\"x\":42")]);

    let mut output = parser
        .parse_chunk(&format!(
            "{}{}{}",
            call_close(),
            TOOLS_CLOSE,
            message_close()
        ))
        .unwrap();
    output.append(parser.finish().unwrap());
    assert_eq!(output.calls(), [delta(0, None, "}")]);
}

#[test]
fn kimi_k3_arguments_preserve_order_and_number_formatting() {
    let body = [
        arg("y", "number", "1.0"),
        arg("x", "number", "2"),
        arg("items", "array", r#"["left","right"]"#),
    ]
    .concat();
    let text = thinking_output("t", "", &call("tool=\"add\" index=\"1\"", &body));

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(
        first_call(&output).arguments,
        r#"{"y":1.0,"x":2,"items":["left","right"]}"#
    );
}

#[test]
fn kimi_k3_string_argument_passes_raw_text_through() {
    let value = "line one\nline two {\"not\": \"json\"} & <tags>";
    let text = thinking_output(
        "t",
        "",
        &call(
            "tool=\"write\" index=\"1\"",
            &arg("content", "string", value),
        ),
    );

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(
        serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
        json!({ "content": value })
    );
}

#[test]
fn kimi_k3_malformed_typed_argument_falls_back_to_raw_text() {
    let text = thinking_output(
        "t",
        "",
        &call(
            "tool=\"calc\" index=\"1\"",
            &arg("x", "number", "not a number"),
        ),
    );

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(first_call(&output).arguments, r#"{"x":"not a number"}"#);
}

#[test]
fn kimi_k3_json_block_arguments_pass_through_raw() {
    // Spacing and key order must survive unmodified: raw `json` blocks are
    // not validated or normalized.
    let raw = r#"{"b": 1,  "a": [2 , 3]}"#;
    let text = thinking_output("t", "", &call("tool=\"run\" index=\"1\"", &json_block(raw)));

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    // No braces are added around a raw json body, and no closing fragment
    // follows it at call end.
    assert_eq!(
        output.calls(),
        [delta(0, Some("run"), ""), delta(0, None, raw)]
    );
}

#[test]
fn kimi_k3_json_block_streams_verbatim_across_chunks() {
    let raw = r#"{"forecast": {"city": "Hangzhou", "days": [8, 20]}}"#;
    let text = format!(
        "{TOOLS_OPEN}{}{TOOLS_CLOSE}{}",
        call("tool=\"run\" index=\"1\"", &json_block(raw)),
        message_close()
    );

    let pieces = char_chunks(&text, 5);
    let piece_refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let chunked = collect_stream(&mut test_parser(), &piece_refs);

    assert_eq!(chunked.coalesced_calls(), [coalesced(0, "run", raw)]);
    assert!(chunked.calls().len() > 2);
}

#[test]
fn kimi_k3_mixed_json_and_typed_arguments_fail() {
    let typed_then_json = format!(
        "{}{}{}",
        arg("x", "number", "1"),
        json_block(r#"{"a":1}"#),
        ""
    );
    let text = thinking_output(
        "t",
        "",
        &format!(
            "{TOOLS_OPEN}{}{TOOLS_CLOSE}{}",
            call("tool=\"run\" index=\"1\"", &typed_then_json),
            message_close()
        ),
    );
    let error = test_parser().parse_complete(&text).unwrap_err();
    assert!(error.to_report_string().contains("mixed"));

    let json_then_typed = format!("{}{}", json_block(r#"{"a":1}"#), arg("x", "number", "1"));
    let text = format!(
        "{TOOLS_OPEN}{}{}{}",
        call_open("tool=\"run\" index=\"1\""),
        json_then_typed,
        call_close(),
    );
    let error = test_parser().parse_complete(&text).unwrap_err();
    assert!(error.to_report_string().contains("mixed"));
}

#[test]
fn kimi_k3_parses_multiple_tool_calls() {
    let tools_body = format!(
        "{}{}",
        call(
            "tool=\"get_weather\" index=\"1\"",
            &arg("city", "string", "SF")
        ),
        call("tool=\"get_time\" index=\"2\"", ""),
    );
    let text = thinking_output("t", "r", &tools_body);

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(
        output.coalesced_calls(),
        [
            coalesced(0, "get_weather", r#"{"city":"SF"}"#),
            coalesced(1, "get_time", "{}"),
        ]
    );
}

#[test]
fn kimi_k3_attribute_values_are_unescaped() {
    let text = thinking_output(
        "t",
        "",
        &call(
            "tool=\"a&quot;b&amp;c\" index=\"1\"",
            &arg("key", "string", "value"),
        ),
    );

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(first_call(&output).name, "a\"b&c");
}

#[test]
fn kimi_k3_call_without_tool_name_is_dropped() {
    let tools_body = format!(
        "{}{}",
        call("index=\"1\"", &arg("x", "number", "1")),
        call("tool=\"real\" index=\"2\"", ""),
    );
    let text = thinking_output("t", "", &tools_body);

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    // Dropped calls emit nothing at all and do not consume an output index.
    assert_eq!(
        output.calls(),
        [delta(0, Some("real"), ""), delta(0, None, "{}")]
    );
}

#[test]
fn kimi_k3_tool_indices_ignore_xtml_index_attribute() {
    let tools_body = format!(
        "{}{}{}",
        call("tool=\"first\" index=\"3\"", ""),
        call("tool=\"second\"", ""),
        call("tool=\"third\" index=\"x\"", ""),
    );
    let text = thinking_output("t", "", &tools_body);

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(
        output.coalesced_calls().iter().map(|call| call.tool_index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn kimi_k3_streaming_splits_markers_across_chunks() {
    let body = [
        arg("city", "string", "Hangzhou"),
        arg("days", "number", "1.5"),
        arg("x", "number", "42"),
    ]
    .concat();
    let text = thinking_output(
        "step by step",
        "the answer",
        &call("tool=\"calc\" index=\"1\"", &body),
    );

    let whole = collect_stream(&mut test_parser(), &[&text]);
    for size in [1, 3, 7] {
        let chunks = char_chunks(&text, size);
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let chunked = collect_stream(&mut test_parser(), &chunk_refs);

        assert_eq!(
            chunked.reasoning_text(),
            "step by step",
            "chunk size {size}"
        );
        assert_eq!(chunked.normal_text(), "the answer", "chunk size {size}");
        assert_eq!(chunked.coalesced_calls(), whole.coalesced_calls());
    }
}

#[test]
fn kimi_k3_streaming_emits_text_incrementally() {
    let mut parser = test_parser();
    let prompt = tokenizer().encode("<|open|>response<|sep|>", false).unwrap();
    parser.initialize(&prompt).unwrap();

    let first = parser.parse_chunk("Hel").unwrap();
    assert_eq!(first.normal_text(), "Hel");

    let second = parser.parse_chunk("lo<|close|>resp").unwrap();
    assert_eq!(second.normal_text(), "lo");

    let mut output = parser.parse_chunk("onse<|sep|>").unwrap();
    output.append(parser.finish().unwrap());
    assert_eq!(output.normal_text(), "");
}

#[test]
fn kimi_k3_initialize_think_prefill_starts_in_reasoning() {
    let mut parser = test_parser();
    let prompt = tokenizer()
        .encode(
            "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>",
            false,
        )
        .unwrap();
    parser.initialize(&prompt).unwrap();

    let output = parser
        .parse_complete(&format!(
            "reasoning{THINK_CLOSE}{RESPONSE_OPEN}answer{RESPONSE_CLOSE}{}",
            message_close()
        ))
        .unwrap();

    assert_eq!(output.reasoning_text(), "reasoning");
    assert_eq!(output.normal_text(), "answer");
}

#[test]
fn kimi_k3_initialize_response_prefill_starts_in_response() {
    let mut parser = test_parser();
    let prompt = tokenizer()
        .encode(
            "<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
            false,
        )
        .unwrap();
    parser.initialize(&prompt).unwrap();

    let output = parser
        .parse_complete(&format!("answer{RESPONSE_CLOSE}{}", message_close()))
        .unwrap();

    assert_eq!(output.normal_text(), "answer");
    assert!(output.reasoning_text().is_empty());
}

#[test]
fn kimi_k3_initialize_message_open_prefill_starts_idle() {
    let mut parser = test_parser();
    let prompt = tokenizer().encode("<|open|>message role=\"assistant\"<|sep|>", false).unwrap();
    parser.initialize(&prompt).unwrap();

    let output = parser
        .parse_complete(&format!("{THINK_OPEN}reason{THINK_CLOSE}{RESPONSE_OPEN}hi"))
        .unwrap();

    assert_eq!(output.reasoning_text(), "reason");
    assert_eq!(output.normal_text(), "hi");
}

#[test]
fn kimi_k3_plain_text_falls_through_as_text() {
    let output = collect_stream(&mut test_parser(), &["plain ", "answer"]);

    assert_eq!(output.normal_text(), "plain answer");
    assert!(output.reasoning_text().is_empty());
    assert!(output.calls().is_empty());
}

#[test]
fn kimi_k3_ignores_output_after_message_close() {
    let mut parser = test_parser();
    let output = parser
        .parse_complete(&format!(
            "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}{}junk{END_OF_MSG}",
            message_close()
        ))
        .unwrap();

    assert_eq!(output.normal_text(), "answer");
}

#[test]
fn kimi_k3_epilogue_noise_is_not_content() {
    let text = format!(
        "{RESPONSE_OPEN}answer{RESPONSE_CLOSE}\n{TOOLS_OPEN}{}{TOOLS_CLOSE}\n{}",
        call("tool=\"calc\" index=\"1\"", ""),
        message_close()
    );

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert_eq!(output.normal_text(), "answer");
    assert_eq!(output.coalesced_calls().len(), 1);
}

#[test]
fn kimi_k3_finish_flushes_unclosed_reasoning() {
    let mut parser = test_parser();
    let mut output = parser.parse_chunk(&format!("{THINK_OPEN}still thinking")).unwrap();
    output.append(parser.finish().unwrap());

    assert_eq!(output.reasoning_text(), "still thinking");
    assert!(output.normal_text().is_empty());
}

#[test]
fn kimi_k3_finish_flushes_partial_marker_as_text() {
    let mut parser = test_parser();
    let mut output = parser.parse_chunk("answer<|clo").unwrap();
    output.append(parser.finish().unwrap());

    assert_eq!(output.normal_text(), "answer<|clo");
}

#[test]
fn kimi_k3_finish_closes_in_flight_string_value() {
    let text = format!(
        "{TOOLS_OPEN}{}{}Hello wor",
        call_open("tool=\"write\" index=\"1\""),
        arg_open("content", "string")
    );

    let output = collect_stream(&mut test_parser(), &[&text]);

    assert_eq!(
        output.calls(),
        [
            delta(0, Some("write"), ""),
            delta(0, None, "{\"content\":\""),
            delta(0, None, "Hello wor"),
            delta(0, None, "\"}"),
        ]
    );
}

#[test]
fn kimi_k3_finish_drops_partial_scalar_value() {
    // A scalar truncated without its close marker is not guaranteed valid
    // JSON, so only the complete argument survives.
    let text = format!(
        "{TOOLS_OPEN}{}{}{}2",
        call_open("tool=\"calc\" index=\"1\""),
        arg("a", "number", "1"),
        arg_open("b", "number")
    );

    let output = collect_stream(&mut test_parser(), &[&text]);

    assert_eq!(
        output.coalesced_calls(),
        [coalesced(0, "calc", r#"{"a":1}"#)]
    );
}

#[test]
fn kimi_k3_finish_flushes_partial_json_block_verbatim() {
    let text = format!(
        "{TOOLS_OPEN}{}{}<|open|>json type=\"object\"{SEP}{{\"a\": 1",
        call_open("tool=\"run\" index=\"1\""),
        ""
    );

    let output = collect_stream(&mut test_parser(), &[&text]);

    assert_eq!(
        output.calls(),
        [delta(0, Some("run"), ""), delta(0, None, r#"{"a": 1"#)]
    );
}

#[test]
fn kimi_k3_finish_after_truncated_tools_keeps_complete_calls() {
    let mut parser = test_parser();
    let mut output = parser
        .parse_chunk(&format!(
            "{TOOLS_OPEN}{}",
            call("tool=\"calc\" index=\"1\"", &arg("x", "number", "1"))
        ))
        .unwrap();
    output.append(parser.finish().unwrap());

    assert_eq!(
        output.coalesced_calls(),
        [coalesced(0, "calc", r#"{"x":1}"#)]
    );
}

#[test]
fn kimi_k3_message_close_truncates_call_best_effort() {
    let text = format!(
        "{TOOLS_OPEN}{}{}{}TRAILING",
        call_open("tool=\"send\" index=\"1\""),
        arg("x", "string", "ok"),
        message_close()
    );

    let output = collect_stream(&mut test_parser(), &[&text]);

    assert_eq!(
        output.coalesced_calls(),
        [coalesced(0, "send", r#"{"x":"ok"}"#)]
    );
    assert!(output.normal_text().is_empty());
}

#[test]
fn kimi_k3_malformed_call_attributes_fail_fast() {
    let mut parser = test_parser();
    let error = parser
        .parse_chunk(&format!("{TOOLS_OPEN}<|open|>call garbage attrs{SEP}"))
        .unwrap_err();

    assert!(error.to_report_string().contains("XTML tag attributes"));
}

#[test]
fn kimi_k3_empty_response_channel_emits_nothing() {
    let text = thinking_output("t", "", &call("tool=\"calc\" index=\"1\"", ""));

    let mut parser = test_parser();
    let output = parser.parse_complete(&text).unwrap();

    assert!(output.normal_text().is_empty());
    assert_eq!(output.reasoning_text(), "t");
    assert_eq!(output.coalesced_calls().len(), 1);
}

#[test]
fn kimi_k3_reset_returns_buffered_text() {
    let mut parser = test_parser();
    let prompt = tokenizer().encode("<|open|>response<|sep|>", false).unwrap();
    parser.initialize(&prompt).unwrap();
    parser.parse_chunk("answer<|close|>resp").unwrap();

    let raw = parser.reset();

    assert_eq!(raw, "<|close|>resp");
}
