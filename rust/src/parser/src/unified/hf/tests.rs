// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Executor tests.
//!
//! Template fixtures and expected values are ported from Transformers
//! `tests/utils/test_chat_parsing.py` at 6d43ab4008, with the expected output
//! dict re-expressed as aggregated unified parser events (`thinking` becomes
//! `reasoning`).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution, Tokenizer as _};

use super::{HfTemplateError, HfUnifiedParser, ResponseTemplate};
use crate::tool::Tool;
use crate::tool::test_utils::split_by_chars;
use crate::unified::{UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput};

fn cohere_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "fields": {
            "content": {"open": "<|START_RESPONSE|>", "close": "<|END_RESPONSE|>", "content": "text"},
            "thinking": {"open": "<|START_THINKING|>", "close": "<|END_THINKING|>", "content": "text"},
            "tool_calls": {
                "open": "<|START_ACTION|>",
                "close": "<|END_ACTION|>",
                "content": "json",
                "transform_each": true,
                "transform": {"type": "function", "function": {"name": "{tool_name}", "arguments": "{parameters}"}},
            },
        },
    })
}

fn gpt_oss_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|start|>assistant",
        "fields": {
            "thinking": {"open": "<|channel|>analysis<|message|>", "close": "<|end|>", "content": "text"},
            "content": {"open": "<|channel|>final<|message|>", "close": "<|end|>", "content": "text"},
            "tool_calls": {
                "open_pattern": r"<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>",
                "close": "<|call|>",
                "repeats": true,
                "content": "json",
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
        },
    })
}

fn smollm_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
            "tool_calls": {
                "open": "<tool_call>",
                "close": "</tool_call>",
                "repeats": true,
                "content": "json",
                "transform": {"type": "function", "function": "{content}"},
            },
            "content": {"close": "<|im_end|>", "content": "text"},
        },
    })
}

fn qwen3_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
            "tool_calls": {
                "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
                "close": "</tool_call>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
                    "value_parser": {"name": "json", "args": {"allow_non_json": true}},
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
        },
    })
}

/// Identical to the template shipped in the Gemma 4 IT checkpoints.
fn gemma4_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": ["<|turn>model\n", "<tool_response|>"],
        "fields": {
            "thinking": {"open": "<|channel>thought\n", "close": "<channel|>", "content": "text"},
            "tool_calls": {
                "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
                "close": "<tool_call|>",
                "repeats": true,
                "content": "json",
                "content_args": {"unquoted_keys": true, "string_delims": [["<|\"|>", "<|\"|>"]]},
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
            "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"], "content": "text"},
        },
    })
}

/// A routed-message protocol with XML-style invocations: no implicit field
/// (text outside regions is discarded), Unicode word boundaries and lazy
/// quantifiers in the opener, `xml-inline` arguments, and joined content.
fn invoke_template() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|begin|>assistant",
        "fields": {
            "reasoning_content": {"open_pattern": r"to=self<\|msg\|>", "close": "<|pause|>", "content": "text"},
            "tool_calls": {
                "open_pattern": r#"<invoke\b[^>]*?\bname="(?P<name>[^"]+)">"#,
                "close": "</invoke>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": r#"<param\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</param>"#,
                    "value_parser": {"name": "json", "args": {"allow_non_json": true}},
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
            "content": {
                "open_pattern": r"to=(?:user|note)<\|msg\|>",
                "close": ["<|end|>", "<|pause|>"],
                "repeats": true,
                "join": "",
                "content": "text",
            },
        },
    })
}

fn compile(template: Value) -> Arc<ResponseTemplate> {
    Arc::new(ResponseTemplate::from_json(&template).unwrap())
}

fn tokenizer() -> Arc<TestTokenizer> {
    Arc::new(TestTokenizer::new())
}

/// Parse `chunks` after initializing from `prompt`, returning all events.
fn parse_events(
    template: &Arc<ResponseTemplate>,
    tools: &[Tool],
    prompt: &str,
    chunks: &[&str],
) -> Result<Vec<UnifiedParserEvent>, UnifiedParserError> {
    let tokenizer = tokenizer();
    let prompt_ids = tokenizer.encode(prompt, false).unwrap();
    let mut parser = HfUnifiedParser::new(template.clone(), tools, tokenizer);
    parser.initialize(&prompt_ids)?;
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        parser.parse_into(DecodedText::unattributed(*chunk), &mut output)?;
    }
    output.append(parser.finish()?);
    Ok(output.events)
}

/// Aggregate events into the shape of the Transformers output dict.
fn message(events: &[UnifiedParserEvent]) -> Value {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls: BTreeMap<usize, (String, String)> = BTreeMap::new();
    for event in events {
        match event {
            UnifiedParserEvent::Text(text) => content.push_str(text),
            UnifiedParserEvent::Reasoning(piece) => reasoning.push_str(&piece.text),
            UnifiedParserEvent::ToolCall(delta) => {
                let call = calls.entry(delta.tool_index).or_default();
                if let Some(name) = &delta.name {
                    call.0.clone_from(name);
                }
                call.1.push_str(&delta.arguments);
            }
        }
    }
    let mut out = serde_json::Map::new();
    if !reasoning.is_empty() {
        out.insert("reasoning".to_string(), reasoning.into());
    }
    if !content.is_empty() {
        out.insert("content".to_string(), content.into());
    }
    if !calls.is_empty() {
        let calls = calls
            .into_values()
            .map(|(name, arguments)| {
                let arguments =
                    serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
                json!({"name": name, "arguments": arguments})
            })
            .collect();
        out.insert("tool_calls".to_string(), Value::Array(calls));
    }
    Value::Object(out)
}

fn parse(template: Value, text: &str) -> Value {
    message(&parse_events(&compile(template), &[], "", &[text]).unwrap())
}

/// Split `text` at pseudo-random char boundaries (xorshift, seeded).
fn random_chunks(text: &str, seed: u64) -> Vec<&str> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut chunks = Vec::new();
    let mut start = 0;
    for (index, _) in text.char_indices().skip(1) {
        if next() % 3 == 0 {
            chunks.push(&text[start..index]);
            start = index;
        }
    }
    chunks.push(&text[start..]);
    chunks
}

#[test]
fn upstream_fixtures_match_transformers_values() {
    let cases = [
        (
            cohere_template(),
            "<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[\n    {\"tool_call_id\": \"0\", \"tool_name\": \"simple_tool\", \"parameters\": {\"temperature_format\": \"Celsius\"}}\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>",
            json!({
                "reasoning": "I should call a tool.",
                "tool_calls": [{"name": "simple_tool", "arguments": {"temperature_format": "Celsius"}}],
            }),
        ),
        (
            gpt_oss_template(),
            "<|channel|>analysis<|message|>We will call function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\n  \"location\": \"San Francisco, CA\"\n}",
            json!({
                "reasoning": "We will call function get_current_weather.",
                "tool_calls": [{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}}],
            }),
        ),
        (
            gpt_oss_template(),
            "<|channel|>analysis<|message|>User asks a simple math question: 2+2 = 4. Provide answer.<|end|><|start|>assistant<|channel|>final<|message|>2",
            json!({
                "reasoning": "User asks a simple math question: 2+2 = 4. Provide answer.",
                "content": "2",
            }),
        ),
        (
            smollm_template(),
            "<think>\nOkay, let me greet them.\n</think>\n\n<tool_call>{\"name\": \"greet_user\", \"arguments\": {\"greeting\": \"Hello!\"}}</tool_call>",
            json!({
                "reasoning": "Okay, let me greet them.",
                "tool_calls": [{"name": "greet_user", "arguments": {"greeting": "Hello!"}}],
            }),
        ),
        (
            smollm_template(),
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>",
            json!({"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}),
        ),
        (
            smollm_template(),
            "<think>\nOkay, gravity.</think>\nSome content about gravity goes here!",
            json!({"reasoning": "Okay, gravity.", "content": "Some content about gravity goes here!"}),
        ),
        (
            qwen3_template(),
            "<tool_call>\n<function=get_weather>\n<parameter=locations>\n[{\"country\": \"France\", \"city\": \"Paris\"}]\n</parameter>\n<parameter=temp_units>\ncelsius\n</parameter>\n</function>\n</tool_call>",
            json!({
                "tool_calls": [{
                    "name": "get_weather",
                    "arguments": {"locations": [{"country": "France", "city": "Paris"}], "temp_units": "celsius"},
                }],
            }),
        ),
        (
            gemma4_template(),
            "<|channel>thought\nThe user is asking for the temperature.<channel|><|tool_call>call:get_current_temperature{detail_level:0,location:<|\"|>Paris, France<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|><|tool_response>",
            json!({
                "reasoning": "The user is asking for the temperature.",
                "tool_calls": [{
                    "name": "get_current_temperature",
                    "arguments": {"detail_level": 0, "location": "Paris, France", "unit": "celsius"},
                }],
            }),
        ),
        (
            gemma4_template(),
            "<|channel>thought\nLet me call the tool.<channel|><|tool_call>call:foo{bool_value:true,list_value:[<|\"|>foo<|\"|>,<|\"|>bar<|\"|>],null_value:null,number_value:1,string_value:<|\"|>foo<|\"|>,struct_value:{foo:<|\"|>bar<|\"|>}}<tool_call|>",
            json!({
                "reasoning": "Let me call the tool.",
                "tool_calls": [{
                    "name": "foo",
                    "arguments": {
                        "bool_value": true,
                        "list_value": ["foo", "bar"],
                        "null_value": null,
                        "number_value": 1,
                        "string_value": "foo",
                        "struct_value": {"foo": "bar"},
                    },
                }],
            }),
        ),
    ];
    for (template, text, expected) in cases {
        assert_eq!(parse(template, text), expected, "{text}");
    }
}

/// Upstream fixtures whose patterns have no literal prefix are rejected at load.
#[test]
fn patterns_without_literal_prefix_are_unsupported() {
    let ernie = json!({
        "start_anchor": "Assistant:",
        "fields": {"thinking": {"open_pattern": r"(?:^|<think>\s*)", "close": "</think>"}},
    });
    let inkling = json!({
        "start_anchor": "<|message_model|>",
        "fields": {"content": {"open_pattern": r"(?:<\|message_model\|>)?[^<]*<\|content_text\|>", "close": "<|end_message|>"}},
    });
    for template in [ernie, inkling] {
        assert!(matches!(
            ResponseTemplate::from_json(&template),
            Err(HfTemplateError::Unsupported { .. })
        ));
    }
}

#[test]
fn streaming_matches_whole_string_for_every_chunking() {
    let fixtures = [
        (
            cohere_template(),
            "<|START_THINKING|>I should call a tool.<|END_THINKING|><|START_ACTION|>[{\"tool_call_id\": \"0\", \"tool_name\": \"simple_tool\", \"parameters\": {\"a\": 1}}]<|END_ACTION|>",
        ),
        (
            gpt_oss_template(),
            "<|channel|>analysis<|message|>thinking chunk<|end|><|channel|>final<|message|>done text",
        ),
        (
            gpt_oss_template(),
            "<|channel|>analysis<|message|>Let me check.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"location\": \"San Francisco, CA\"}<|call|>",
        ),
        (
            smollm_template(),
            "<think>thinking</think>\n<tool_call>{\"name\": \"fn\", \"arguments\": {\"x\": 1}}</tool_call>",
        ),
        (
            qwen3_template(),
            "<think>short thought</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        ),
        (
            gemma4_template(),
            "<|channel>thought\nhi 天气<channel|>前言 <|tool_call>call:foo{a:1,b:<|\"|>bar<|\"|>}<tool_call|> after",
        ),
        (
            invoke_template(),
            "header to=self<|msg|>plan it<|pause|>to=user<|msg|>Hello <|pause|><invoke id=\"1\" name=\"get\"><param name=\"city\">\"Paris\"</param><param name=\"days\">3</param></invoke>to=note<|msg|>world<|end|>",
        ),
    ];
    for (template, text) in fixtures {
        let template = compile(template);
        let expected = message(&parse_events(&template, &[], "", &[text]).unwrap());
        let mut chunkings: Vec<Vec<&str>> =
            [1, 2, 3, 5, 7, 13, 31].map(|step| split_by_chars(text, step)).into();
        chunkings.extend((0..30).map(|seed| random_chunks(text, 0xC0DE_5EED + seed)));
        for chunks in chunkings {
            assert_eq!(chunks.concat(), text);
            let actual = message(&parse_events(&template, &[], "", &chunks).unwrap());
            assert_eq!(actual, expected, "{chunks:?}");
        }
    }
}

#[test]
fn gemma4_streams_text_and_starts_calls_at_the_opener() {
    let template = compile(gemma4_template());
    let chunks = [
        "<|channel>thought\n",
        "Check ",
        "the weather. ",
        "<channel|>Sure",
        ". <|tool_call>call:get_weather",
        "{location:<|\"|>Paris<|\"|>}",
        "<tool_call|>",
    ];
    let tokenizer = tokenizer();
    let mut parser = HfUnifiedParser::new(template, &[], tokenizer);
    parser.initialize(&[]).unwrap();
    let steps: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            let mut output = UnifiedParserOutput::default();
            parser.parse_into(DecodedText::unattributed(*chunk), &mut output).unwrap();
            (*chunk, output.events)
        })
        .collect();
    expect_test::expect![[r#"
        [
            (
                "<|channel>thought\n",
                [],
            ),
            (
                "Check ",
                [
                    Reasoning(
                        DecodedText {
                            text: "Check",
                            attributions: [],
                        },
                    ),
                ],
            ),
            (
                "the weather. ",
                [
                    Reasoning(
                        DecodedText {
                            text: " the weather.",
                            attributions: [],
                        },
                    ),
                ],
            ),
            (
                "<channel|>Sure",
                [
                    Text(
                        "Sure",
                    ),
                ],
            ),
            (
                ". <|tool_call>call:get_weather",
                [
                    Text(
                        ".",
                    ),
                ],
            ),
            (
                "{location:<|\"|>Paris<|\"|>}",
                [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "",
                        },
                    ),
                ],
            ),
            (
                "<tool_call|>",
                [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "{\"location\":\"Paris\"}",
                        },
                    ),
                ],
            ),
        ]
    "#]]
    .assert_debug_eq(&steps);
}

#[test]
fn invoke_protocol_discards_unrouted_text_and_joins_content() {
    let text = "stray to=self<|msg|> plan <|pause|>to=user<|msg|>Hello, <|pause|>between<invoke name=\"get\"><param name=\"n\">x</param></invoke>to=note<|msg|>world<|end|>tail";
    expect_test::expect![[r#"
        Object {
            "reasoning": String("plan"),
            "content": String("Hello,world"),
            "tool_calls": Array [
                Object {
                    "name": String("get"),
                    "arguments": Object {
                        "n": String("x"),
                    },
                },
            ],
        }
    "#]]
    .assert_debug_eq(&parse(invoke_template(), text));

    // Unicode word boundaries: `é` is a word character, so `éname=` is not the `name` attribute.
    let text = "<invoke title=\"天气\" name=\"获取\"><param name=\"城市\">北京</param></invoke><invoke éname=\"x\">";
    assert_eq!(
        parse(invoke_template(), text),
        json!({"tool_calls": [{"name": "获取", "arguments": {"城市": "北京"}}]})
    );
}

#[test]
fn join_concatenates_repeated_matches() {
    let template = json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|assistant|>",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "repeats": true, "join": " "},
            "content": {"repeats": true, "join": " "},
        },
    });
    assert_eq!(
        parse(
            template.clone(),
            "<think>first</think>middle<think>second</think>done"
        ),
        json!({"reasoning": "first second", "content": "middle done"})
    );
    assert_eq!(
        parse(template, "<think>only</think>"),
        json!({"reasoning": "only"})
    );
}

#[test]
fn transforms_with_dotted_paths() {
    let template = json!({
        "start_anchor": "<|assistant|>",
        "fields": {
            "tool_calls": {
                "open": "<actions>",
                "close": "</actions>",
                "content": "json",
                "transform_each": true,
                "transform": {"type": "function", "function": {"name": "{fn.name}", "arguments": "{fn.args}"}},
            },
        },
    });
    assert_eq!(
        parse(
            template,
            r#"<actions>[{"fn": {"name": "a", "args": {"x": 1}}}, {"fn": {"name": "b", "args": {}}}]</actions>"#
        ),
        json!({"tool_calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]})
    );
}

#[test]
fn literal_boundaries() {
    let template = json!({
        "start_anchor": "<|assistant|>",
        "fields": {"content": {"open": ["<a>", "<bb>"], "close": ["</a>", "</bb>"]}},
    });
    for (opener, closer) in [("<a>", "</a>"), ("<bb>", "</bb>"), ("<a>", "</bb>")] {
        assert_eq!(
            parse(template.clone(), &format!("{opener}hi{closer}")),
            json!({"content": "hi"})
        );
    }

    // A literal that is a strict prefix of another in the same list defers at the buffer edge.
    let template = compile(json!({
        "start_anchor": "<|assistant|>",
        "fields": {"content": {"open": "<x>", "close": ["END", "ENDX"]}},
    }));
    let events = parse_events(&template, &[], "", &["<x>hiEND", " more"]).unwrap();
    assert_eq!(message(&events), json!({"content": "hi"}));

    // A field without a close runs to the end of the stream.
    let template =
        json!({"start_anchor": "<|assistant|>", "fields": {"content": {"open": "<resp>"}}});
    assert_eq!(
        parse(template, "<resp>hello world"),
        json!({"content": "hello world"})
    );
}

#[test]
fn required_field_missing_fails_at_finish() {
    let template = compile(json!({
        "start_anchor": "<|assistant|>",
        "fields": {"content": {"open": "<response>", "close": "</response>", "optional": false}},
    }));
    let error = parse_events(&template, &[], "", &["no response here"]).unwrap_err();
    expect_test::expect![[r#"
        ParsingFailed {
            message: "Required response_template fields missing from parsed output: [\"content\"]",
        }
    "#]]
    .assert_debug_eq(&error);
}

#[test]
fn prefix_sets_the_initial_state() {
    let template = compile(qwen3_template());

    // The prompt ends inside the thinking block; the prefilled newline is stripped.
    let prompt = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let events = parse_events(&template, &[], prompt, &["Let me think...</think>"]).unwrap();
    assert_eq!(message(&events), json!({"reasoning": "Let me think..."}));

    // Only the slice after the last anchor matters.
    let prompt = "<|im_start|>assistant\nEarlier reply<|im_end|>\n<|im_start|>user\nFollowup<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let events = parse_events(&template, &[], prompt, &["done</think>"]).unwrap();
    assert_eq!(message(&events), json!({"reasoning": "done"}));

    // A prefix ending mid-delimiter is completed by the generated text.
    let events = parse_events(
        &template,
        &[],
        "<|im_start|>assistant\n<thi",
        &["nk>real body</think>"],
    )
    .unwrap();
    assert_eq!(message(&events), json!({"reasoning": "real body"}));

    // Gemma 4 continues inside the same turn after a tool response.
    let template = compile(gemma4_template());
    let prompt = "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>response:f{}<tool_response|><|channel>thought\n";
    let events = parse_events(&template, &[], prompt, &["Done.<channel|>It is sunny."]).unwrap();
    assert_eq!(
        message(&events),
        json!({"reasoning": "Done.", "content": "It is sunny."})
    );

    // Prompt text is never emitted, but it counts for leading-whitespace stripping.
    let events = parse_events(&template, &[], "<|turn>model\nSure,", &[" here", " it is"]).unwrap();
    assert_eq!(message(&events), json!({"content": " here it is"}));
}

/// Diverges from Transformers, which runs the whole prompt through the parser
/// when the start anchor is absent.
#[test]
fn missing_start_anchor_starts_from_the_initial_state() {
    let template = compile(smollm_template());
    let events = parse_events(&template, &[], "<think>\n", &["hi</think>"]).unwrap();
    assert_eq!(message(&events), json!({"content": "hi</think>"}));
}

#[test]
fn tools_convert_string_arguments() {
    let template = compile(json!({
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "tool_calls": {
                "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
                "close": "</tool_call>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
                    "merge_duplicates": true,
                },
                "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
            },
        },
    }));
    let tools = [Tool {
        name: "set_alarm".to_string(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "hour": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "label": {"type": "string"},
                "weekday": {"type": "integer"},
            },
        }),
        strict: None,
        defer_loading: None,
    }];
    let text = "<tool_call>\n<function=set_alarm>\n<parameter=hour>\n7\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>\n<parameter=label>\nwake up\n</parameter>\n<parameter=weekday>1</parameter><parameter=weekday>2</parameter>\n</function>\n</tool_call>";
    assert_eq!(
        message(&parse_events(&template, &tools, "", &[text]).unwrap()),
        json!({"tool_calls": [{"name": "set_alarm", "arguments": {"hour": 7, "enabled": true, "label": "wake up", "weekday": [1, 2]}}]})
    );
    // Without tools, values stay strings.
    assert_eq!(
        message(&parse_events(&template, &[], "", &[text]).unwrap()),
        json!({"tool_calls": [{"name": "set_alarm", "arguments": {"hour": "7", "enabled": "true", "label": "wake up", "weekday": ["1", "2"]}}]})
    );
}

#[test]
fn stripped_reasoning_whitespace_keeps_its_tokens() {
    let template = compile(gemma4_template());
    let mut parser = HfUnifiedParser::new(template, &[], tokenizer());
    parser.initialize(&[]).unwrap();
    let text = "<|channel>thought\n \n plan \n<channel|>";
    let mut output = UnifiedParserOutput::default();
    for (index, c) in text.chars().enumerate() {
        let piece = DecodedText {
            text: c.to_string(),
            attributions: [TokenAttribution {
                token_id: u32::try_from(index).unwrap(),
                anchor: TokenAnchor::Visible { byte_offset: 0 },
            }]
            .into_iter()
            .collect(),
        };
        parser.parse_into(piece, &mut output).unwrap();
    }
    output.append(parser.finish().unwrap());
    let (reasoning, tokens) =
        output
            .events
            .iter()
            .fold((String::new(), 0), |(text, tokens), event| match event {
                UnifiedParserEvent::Reasoning(piece) => {
                    (text + &piece.text, tokens + piece.attributions.len())
                }
                _ => (text, tokens),
            });
    assert_eq!(reasoning, "plan");
    // Every character between the markers is a reasoning token, stripped or not.
    assert_eq!(tokens, " \n plan \n".chars().count());
}

#[test]
fn reset_returns_uncommitted_text() {
    let template = compile(gemma4_template());
    let mut parser = HfUnifiedParser::new(template, &[], tokenizer());
    parser.initialize(&[]).unwrap();
    let mut output = UnifiedParserOutput::default();
    parser
        .parse_into(
            DecodedText::unattributed("Hi <|tool_call>call:f{a:<|\"|>x"),
            &mut output,
        )
        .unwrap();
    assert_eq!(parser.reset(), "<|tool_call>call:f{a:<|\"|>x");
}

#[test]
fn malformed_tool_arguments_fail_the_parser() {
    let template = compile(gemma4_template());
    let error =
        parse_events(&template, &[], "", &["<|tool_call>call:f{a:}<tool_call|>"]).unwrap_err();
    assert!(
        matches!(error, UnifiedParserError::ParsingFailed { .. }),
        "{error:?}"
    );
}

#[test]
fn template_validation_errors() {
    let error = |template: Value| ResponseTemplate::from_json(&template).unwrap_err();
    expect_test::expect![[r#"
        [
            Invalid {
                message: "unsupported response_template version: 2",
            },
            Invalid {
                message: "At most one field may omit 'open'/'open_pattern' (that field becomes the implicit-open / leftover sink). Found: content, thinking",
            },
            Invalid {
                message: "response_template must define 'start_anchor' or 'start_anchor_pattern'.",
            },
            Invalid {
                message: "Field 'content': 'join' requires 'repeats': true",
            },
            Invalid {
                message: "Field 'tool_calls': open_pattern/close_pattern declares named group(s) [\"name\"], but the field has no 'transform'. Named captures are only surfaced through a 'transform' template (where they appear alongside 'content'). Either add a 'transform' that uses the captures, or remove the named groups from the pattern.",
            },
            Invalid {
                message: "Field 'tool_calls': transform string \"name: {content}\" mixes a {placeholder} with literal text. Use either a whole-string placeholder (e.g. \"{content}\") or a plain literal; string interpolation is not supported.",
            },
            Invalid {
                message: "Field 'content': 'open' literals cannot be empty strings",
            },
            Unsupported {
                message: "Field 'count': only 'content', 'reasoning_content'/'thinking', and 'tool_calls' fields can be reported",
            },
            Unsupported {
                message: "Field 'thinking': text and reasoning fields must use the 'text' content parser without a transform",
            },
            Invalid {
                message: "response_template.fields.content.content: unknown variant `yaml`, expected one of `text`, `int`, `float`, `bool`, `json`, `xml-inline`, `kv-lines`",
            },
            Invalid {
                message: "response_template.fields.content.repeats: invalid type: string \"yes\", expected a boolean",
            },
            Invalid {
                message: "Field 'content': cannot specify both 'open' and 'open_pattern'",
            },
            Invalid {
                message: "Field 'tool_calls': transform_each is set but no transform was provided",
            },
            Unsupported {
                message: "Field 'tool_calls': 'join' requires each match to parse to a string",
            },
            Invalid {
                message: "Field 'tool_calls': transform placeholder '{id}' is neither 'content' nor a named group of open_pattern",
            },
        ]
    "#]]
    .assert_debug_eq(&[
        error(json!({"version": 2, "start_anchor": "a", "fields": {"content": {}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {}, "thinking": {}}})),
        error(json!({"fields": {"content": {}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {"open": "<x>", "join": ""}}})),
        error(json!({"start_anchor": "a", "fields": {"tool_calls": {"open_pattern": r"<t (?P<name>\w+)>", "close": "</t>"}}})),
        error(json!({"start_anchor": "a", "fields": {"tool_calls": {"open": "<t>", "transform": {"label": "name: {content}"}}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {"open": [""]}}})),
        error(json!({"start_anchor": "a", "fields": {"count": {"open": "<n>", "content": "int"}}})),
        error(json!({"start_anchor": "a", "fields": {"thinking": {"open": "<n>", "content": "json"}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {"content": "yaml"}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {"repeats": "yes"}}})),
        error(json!({"start_anchor": "a", "fields": {"content": {"open": "<x>", "open_pattern": "<y>"}}})),
        error(json!({"start_anchor": "a", "fields": {"tool_calls": {"open": "<t>", "transform_each": true}}})),
        error(json!({"start_anchor": "a", "fields": {"tool_calls": {"open": "<t>", "repeats": true, "join": ""}}})),
        error(json!({
            "start_anchor": "a",
            "fields": {"tool_calls": {
                "open_pattern": r"<t (?P<name>\w+)>",
                "close_pattern": r"</t (?P<id>\d+)>",
                "content": "json",
                "transform": {"function": {"name": "{name}", "arguments": "{content}", "id": "{id}"}},
            }},
        })),
    ]);
}

/// Cases where the Rust parser intentionally differs from Transformers, keyed by
/// generated text.
const DIVERGENCES: &[(&str, &str)] = &[
    (
        // A field without `repeats` keeps only its last occurrence in Transformers;
        // streamed text cannot be retracted, so every occurrence is reported.
        r#"Before <|tool_call>call:f{x:<|"|>a, b<|"|>,y:[1,2.5,-3],z:{w:true}}<tool_call|> after"#,
        r#"{"content": "Before  after", "tool_calls": [{"name": "f", "arguments": {"x": "a, b", "y": [1, 2.5, -3], "z": {"w": true}}}]}"#,
    ),
    (
        // Arguments use the tool parsers' shared conversion, which keeps a JSON
        // number's spelling; Transformers turns `1e3` into the integer 1000.
        "<tool_call>\n<function=set_alarm>\nhour: seven\nratio: 1e3\nenabled: nope\n</tool_call>",
        r#"{"tool_calls": [{"name": "set_alarm", "arguments": {"hour": "seven", "ratio": 1000.0, "enabled": "nope"}}]}"#,
    ),
];

/// Compare against `fixtures/differential.json`, generated from Transformers
/// `parse_response` by `fixtures/generate_differential.py`.
#[test]
fn differential_against_transformers() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/unified/hf/fixtures/differential.json");
    let corpus: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let templates: BTreeMap<&str, Arc<ResponseTemplate>> = corpus["templates"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, template)| {
            (
                name.as_str(),
                Arc::new(ResponseTemplate::from_json(template).unwrap()),
            )
        })
        .collect();

    for case in corpus["cases"].as_array().unwrap() {
        let template = &templates[case["template"].as_str().unwrap()];
        let prefix = case["prefix"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let tools: Vec<Tool> = case["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| Tool {
                name: tool["function"]["name"].as_str().unwrap().to_string(),
                description: None,
                parameters: tool["function"]["parameters"].clone(),
                strict: None,
                defer_loading: None,
            })
            .collect();
        let expected = match DIVERGENCES.iter().find(|(divergent, _)| *divergent == text) {
            Some((_, expected)) => serde_json::from_str(expected).unwrap(),
            None => case["expected"].clone(),
        };

        for chunks in [vec![text], split_by_chars(text, 1)] {
            let actual = parse_events(template, &tools, prefix, &chunks);
            match (&actual, expected.get("error")) {
                (Err(_), Some(_)) => {}
                (Ok(events), None) => assert_eq!(message(events), expected, "{text:?}"),
                _ => panic!("{text:?}: expected {expected}, got {actual:?}"),
            }
        }
    }
}

#[test]
fn tool_call_value_shapes() {
    // No transform: the parsed JSON is the tool call itself.
    let template = compile(json!({
        "start_anchor": "<|assistant|>",
        "fields": {"tool_calls": {"open": "<tool_call>", "close": "</tool_call>", "content": "json"}},
    }));
    let parse =
        |text: &str| parse_events(&template, &[], "", &[text]).map(|events| message(&events));

    assert_eq!(
        parse(r#"<tool_call>{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}</tool_call>"#).unwrap(),
        json!({"tool_calls": [{"name": "f", "arguments": {"a": 1}}]})
    );
    assert_eq!(
        parse(r#"<tool_call>[{"name": "f", "arguments": null}, {"name": "g", "arguments": "raw"}]</tool_call>"#).unwrap(),
        json!({"tool_calls": [{"name": "f", "arguments": {}}, {"name": "g", "arguments": "raw"}]})
    );
    expect_test::expect![[r#"
        ParsingFailed {
            message: "tool call must be {\"function\": {\"name\": ..., \"arguments\": ...}} or {\"name\": ..., \"arguments\": ...} with a string name, got {\"name\":1}",
        }
    "#]]
        .assert_debug_eq(&parse(r#"<tool_call>{"name": 1}</tool_call>"#).unwrap_err());
}
