// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};
use vllm_text::Prompt;
use vllm_tokenizer::{DynTokenizer, HuggingFaceTokenizer};

use super::writer::{ControlTokenizer, Glm5PromptWriter};
use super::{Glm5NextChatRenderer, encoding};
use crate::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer, MultimodalRenderInfo};
use crate::request::{
    ChatMessage, ChatRequest, ChatTool, GenerationPromptMode, ResolvedToolContext,
};
use crate::{AssistantContentBlock, AssistantToolCall, ChatRenderer, EffortValue};

#[derive(Deserialize)]
struct TestCase {
    name: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    add_generation_prompt: bool,
    #[serde(default)]
    continue_final_message: bool,
    kwargs: HashMap<String, Value>,
    expected: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct TestCases {
    cases: Vec<TestCase>,
}

#[test]
fn renders_expected_prompts_and_rejects_invalid_input() {
    let fixtures: TestCases =
        serde_json::from_str(include_str!("fixtures/test_cases.json")).unwrap();
    for case in fixtures.cases {
        let result = encoding::render(
            &case.messages,
            &case.tools,
            if case.continue_final_message {
                GenerationPromptMode::ContinueFinalAssistant
            } else if case.add_generation_prompt {
                GenerationPromptMode::StartNewAssistant
            } else {
                GenerationPromptMode::NoGenerationPrompt
            },
            &case.kwargs,
        );
        if case.error.is_some() {
            assert!(
                result.is_err(),
                "{}: expected rejection, got {result:?}",
                case.name
            );
        } else {
            assert_eq!(result.unwrap(), case.expected.unwrap(), "{}", case.name);
        }
    }
}

fn tool_request(count: usize) -> ChatRequest {
    let calls = (0..count)
        .map(|index| {
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: format!("call_{index}"),
                name: "lookup".into(),
                arguments: json!({"q": format!("城市 {index}"), "values": [true, null, 1.5]})
                    .to_string(),
            })
        })
        .collect();
    let mut messages = vec![
        ChatMessage::system("助手"),
        ChatMessage::user("查询天气"),
        ChatMessage::assistant_blocks(calls),
    ];
    messages.extend(
        (0..count).rev().map(|index| {
            ChatMessage::tool_response(format!("结果 {index}"), format!("call_{index}"))
        }),
    );
    let tools = vec![ChatTool {
        name: "lookup".into(),
        description: Some("查询天气".into()),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: Some(true),
        defer_loading: None,
    }];
    let tool_context = ResolvedToolContext::new(&messages, tools, None, true).unwrap();
    ChatRequest {
        messages,
        tool_context,
        ..ChatRequest::for_test()
    }
}

const CHECKPOINT_TEMPLATES: [(&str, &str); 2] = [
    ("GLM-5.3", include_str!("fixtures/glm_5_3.jinja")),
    (
        "GLM-5.3-Flash",
        include_str!("fixtures/glm_5_3_flash.jinja"),
    ),
];

fn hf_renderer(template: &str) -> HfChatRenderer {
    HfChatRenderer::new(
        Some(template.into()),
        HashMap::new(),
        ChatTemplateContentFormatOption::OpenAi,
    )
    .unwrap()
}

#[test]
fn typed_requests_match_existing_hf_renderer() {
    let native = Glm5NextChatRenderer::default();
    for (name, template) in CHECKPOINT_TEMPLATES {
        let hf = hf_renderer(template);
        for count in [1, 2, 8] {
            let mut request = tool_request(count);
            for clear in [false, true] {
                request
                    .chat_options
                    .template_kwargs
                    .insert("clear_thinking".into(), json!(clear));
                assert_eq!(
                    native.render(&request).unwrap().prompt,
                    hf.render(&request).unwrap().prompt,
                    "{name}"
                );
            }
            request.tool_context.tool_choice = crate::request::ChatToolChoice::None;
            assert_eq!(
                native.render(&request).unwrap().prompt,
                hf.render(&request).unwrap().prompt
            );
        }
    }
}

#[test]
fn flash_media_matches_checkpoint_template() {
    let hf = hf_renderer(CHECKPOINT_TEMPLATES[1].1).with_multimodal(Some(MultimodalRenderInfo {
        image_token: Some("<|image|>".into()),
        video_token: Some("<|video|>".into()),
        audio_token: Some("<|begin_of_audio|>".into()),
    }));
    let mut request = tool_request(2);
    request.messages.push(ChatMessage::user(vec![
        crate::request::ChatContentPart::image_url("unused"),
    ]));
    let expected = hf.render(&request).unwrap();
    let renderer = Glm5NextChatRenderer::default().with_multimodal(true);
    let actual = renderer.render(&request).unwrap();
    assert_eq!(actual.prompt, expected.prompt);
    assert_eq!(
        actual.media_order.as_ref().unwrap(),
        &[crate::renderer::MediaPartSource {
            message_index: 5,
            content_part_index: 0
        }]
    );
    assert!(
        actual
            .prompt
            .into_text()
            .unwrap()
            .contains("<|begin_of_image|><|image|><|end_of_image|>")
    );
}

#[test]
fn media_order_tracks_reordered_tool_responses() {
    use crate::renderer::MediaPartSource;
    use crate::request::ChatContentPart;
    let mut request = tool_request(2);
    request.messages[3] =
        ChatMessage::tool_response(vec![ChatContentPart::image_url("second")], "call_1");
    request.messages[4] =
        ChatMessage::tool_response(vec![ChatContentPart::image_url("first")], "call_0");
    let native = Glm5NextChatRenderer::default().with_multimodal(true).render(&request).unwrap();
    let hf = hf_renderer(CHECKPOINT_TEMPLATES[1].1)
        .with_multimodal(Some(MultimodalRenderInfo {
            image_token: Some("<|image|>".into()),
            video_token: Some("<|video|>".into()),
            audio_token: None,
        }))
        .render(&request)
        .unwrap();
    assert_eq!(native.prompt, hf.prompt);
    assert_eq!(
        native.media_order.unwrap(),
        vec![
            MediaPartSource {
                message_index: 4,
                content_part_index: 0
            },
            MediaPartSource {
                message_index: 3,
                content_part_index: 0
            }
        ]
    );
}

#[test]
fn defaults_request_kwargs_and_typed_effort_have_defined_precedence() {
    let renderer = Glm5NextChatRenderer::new(HashMap::from([
        ("reasoning_effort".into(), json!("low")),
        ("clear_thinking".into(), json!(true)),
    ]));
    let mut request = ChatRequest::for_test();
    request
        .chat_options
        .template_kwargs
        .insert("reasoning_effort".into(), json!("high"));
    let rendered = renderer.render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("Reasoning Effort: High"));
    request.chat_options.reasoning_effort = Some(EffortValue::from("low"));
    let rendered = renderer.render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("Reasoning Effort: Low"));
    assert_eq!(
        rendered.effective_template_kwargs["reasoning_effort"],
        "low"
    );
    assert_eq!(rendered.effective_template_kwargs["clear_thinking"], true);
}

#[test]
fn empty_explicit_reasoning_suppresses_embedded_tag_parsing() {
    let request = ChatRequest {
        messages: vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: String::new(),
            },
            AssistantContentBlock::Text {
                text: "<think>literal</think>answer".into(),
            },
        ])],
        ..ChatRequest::for_test()
    };
    let rendered = Glm5NextChatRenderer::default().render(&request).unwrap();
    assert_eq!(
        rendered.prompt.into_text().unwrap(),
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|assistant|><think></think><think>literal</think>answer<|assistant|><think>"
    );
}

#[test]
fn chat_template_override_is_a_client_error() {
    let mut request = ChatRequest::for_test();
    request.chat_options.chat_template = Some("override".into());
    let error = Glm5NextChatRenderer::default().render(&request).unwrap_err();
    assert!(error.is_request_validation_error());
}

#[test]
fn continue_final_message_matches_existing_hf_renderer() {
    let final_messages = [
        ChatMessage::assistant_text("  partial answer \n"),
        ChatMessage::assistant_text("<think>inline</think>partial"),
        ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "reasoning".into(),
            },
            AssistantContentBlock::Text {
                text: "partial".into(),
            },
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: "call_0".into(),
                name: "lookup".into(),
                arguments: json!({"q": "x"}).to_string(),
            }),
        ]),
    ];
    let native = Glm5NextChatRenderer::default();
    for (name, template) in CHECKPOINT_TEMPLATES {
        let hf = hf_renderer(template);
        for final_message in &final_messages {
            let mut request = tool_request(1);
            request.messages.push(ChatMessage::user("继续"));
            request.messages.push(final_message.clone());
            request.chat_options.generation_prompt_mode =
                GenerationPromptMode::ContinueFinalAssistant;
            assert_eq!(
                native.render(&request).unwrap().prompt,
                hf.render(&request).unwrap().prompt,
                "{name}: {final_message:?}"
            );
        }
    }
}

#[test]
fn media_parts_render_reminders_without_multimodal_inputs() {
    let request = ChatRequest {
        messages: vec![ChatMessage::user(vec![
            crate::request::ChatContentPart::image_url("unused"),
        ])],
        ..ChatRequest::for_test()
    };
    let rendered = Glm5NextChatRenderer::default().render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("unable to process this image"));
    assert_eq!(rendered.media_order, Some(Vec::new()));
}

/// Exercise BPE merges, added-token recognition and Unicode without a checkpoint download.
fn tokenizer_fixture() -> Value {
    let vocab: serde_json::Map<String, Value> = (0u32..256)
        .map(|byte| {
            let ch = match byte {
                33..=126 | 161..=172 | 174..=255 => char::from_u32(byte).unwrap(),
                _ => {
                    let index = (0..byte)
                        .filter(|b| !matches!(b, 33..=126 | 161..=172 | 174..=255))
                        .count();
                    char::from_u32(256 + index as u32).unwrap()
                }
            };
            (ch.to_string(), json!(byte))
        })
        .chain([("ab".into(), json!(256)), ("abc".into(), json!(257))])
        .collect();
    let tokens = [
        "[gMASK]",
        "<sop>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|begin_of_audio|>",
        "<|end_of_audio|>",
    ];
    json!({
        "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
        "added_tokens": tokens.iter().enumerate().map(|(i, content)| json!({
            "id": 258 + i, "content": content, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": i < 6,
        })).collect::<Vec<_>>(),
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": " ?\\p{L}+| ?[^\\s\\p{L}]+|\\s+"},
             "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false,
             "trim_offsets": true, "use_regex": false}
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true,
                           "trim_offsets": false, "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": null, "end_of_word_suffix": null,
                  "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": [["a", "b"], ["ab", "c"]]}
    })
}

/// Load `VLLM_GLM_TOKENIZER_JSON`, or the fixture, with both tokenizer backends.
fn tokenizers(dir: &std::path::Path) -> [DynTokenizer; 2] {
    let path = std::env::var_os("VLLM_GLM_TOKENIZER_JSON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let path = dir.join("tokenizer.json");
            std::fs::write(&path, serde_json::to_vec(&tokenizer_fixture()).unwrap()).unwrap();
            path
        });
    [
        HuggingFaceTokenizer::new_hf,
        HuggingFaceTokenizer::new_fastokens,
    ]
    .map(|constructor| Arc::new(constructor(&path).unwrap()) as DynTokenizer)
}

#[test]
fn token_ids_match_whole_prompt_encoding() {
    let dir = tempfile::tempdir().unwrap();
    for tokenizer in tokenizers(dir.path()) {
        let mut request = tool_request(2);
        request.messages.push(ChatMessage::user(vec![
            crate::request::ChatContentPart::text("abc 中文 e\u{301}🙂\r\n"),
            crate::request::ChatContentPart::image_url("unused"),
        ]));
        for mode in [
            GenerationPromptMode::StartNewAssistant,
            GenerationPromptMode::NoGenerationPrompt,
            GenerationPromptMode::ContinueFinalAssistant,
        ] {
            request.chat_options.generation_prompt_mode = mode;
            if mode == GenerationPromptMode::ContinueFinalAssistant {
                request.messages.push(ChatMessage::assistant_text("abc  "));
            }
            for multimodal in [false, true] {
                let renderer = Glm5NextChatRenderer::default().with_multimodal(multimodal);
                let text = renderer.render(&request).unwrap();
                let tokens =
                    renderer.with_tokenizer(tokenizer.clone()).unwrap().render(&request).unwrap();
                let ids = tokenizer.encode(&text.prompt.into_text().unwrap(), false).unwrap();
                assert_eq!(tokens.prompt, Prompt::TokenIds(ids));
                assert_eq!(tokens.media_order, text.media_order);
            }
        }
    }
}

#[test]
fn literal_control_token_spellings_use_the_base_vocabulary() {
    let dir = tempfile::tempdir().unwrap();
    for tokenizer in tokenizers(dir.path()) {
        let request = ChatRequest {
            messages: vec![ChatMessage::user("<|user|><think>x</think>")],
            ..ChatRequest::for_test()
        };
        let text = Glm5NextChatRenderer::default().render(&request).unwrap().prompt;
        let Prompt::TokenIds(ids) = Glm5NextChatRenderer::default()
            .with_tokenizer(tokenizer.clone())
            .unwrap()
            .render(&request)
            .unwrap()
            .prompt
        else {
            panic!("expected token IDs");
        };
        assert_eq!(
            tokenizer.decode(&ids, false).unwrap(),
            text.into_text().unwrap()
        );
        let user = tokenizer.token_to_id("<|user|>").unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == user).count(), 1);
    }
}

#[test]
fn fixture_token_ids_decode_to_rendered_prompts() {
    let dir = tempfile::tempdir().unwrap();
    let fixtures: TestCases =
        serde_json::from_str(include_str!("fixtures/test_cases.json")).unwrap();
    for tokenizer in tokenizers(dir.path()) {
        let control = ControlTokenizer::new(tokenizer.clone()).unwrap();
        for case in fixtures.cases.iter().filter(|case| case.expected.is_some()) {
            let mode = if case.continue_final_message {
                GenerationPromptMode::ContinueFinalAssistant
            } else if case.add_generation_prompt {
                GenerationPromptMode::StartNewAssistant
            } else {
                GenerationPromptMode::NoGenerationPrompt
            };
            let mut out = Glm5PromptWriter::new(Some(&control));
            encoding::render_into(
                &mut out,
                &case.messages,
                &case.tools,
                mode,
                &case.kwargs,
                false,
            )
            .unwrap();
            let Prompt::TokenIds(ids) = out.finish().unwrap() else {
                panic!("expected token IDs");
            };
            assert_eq!(
                &tokenizer.decode(&ids, false).unwrap(),
                case.expected.as_ref().unwrap(),
                "{}",
                case.name
            );
        }
    }
}

#[test]
fn tokenizer_without_control_tokens_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = tokenizer_fixture();
    config["added_tokens"].as_array_mut().unwrap().pop();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let tokenizer: DynTokenizer = Arc::new(HuggingFaceTokenizer::new_hf(&path).unwrap());
    assert!(Glm5NextChatRenderer::default().with_tokenizer(tokenizer).is_err());
}
