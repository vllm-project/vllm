// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use futures::{StreamExt as _, stream};
use serde::Deserialize;
use serde_json::{Value, json};
use vllm_chat::ChatOutputProcessor;
use vllm_chat::ResolvedToolContext;
use vllm_chat::{AssistantContentBlock, ChatEvent};
use vllm_chat::{
    ChatRequest, ChatTool, ChatToolChoice, DefaultChatOutputProcessor, ParserSelection,
};
use vllm_parser::unified::{REGISTERED_UNIFIED_FAMILIES, canonical_unified_family};
use vllm_text::output::{DecodedTextEvent, Finished};
use vllm_text::{DecodedText, SampledDelta};
use vllm_tokenizer::test_utils::TestTokenizer;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    family: String,
    model: String,
    input: String,
    tools: Vec<ChatTool>,
    expected: Vec<Value>,
    call_only_input: String,
}

fn fixtures() -> Vec<Case> {
    serde_json::from_str::<Fixture>(include_str!("fixtures/dynamo_unified.json"))
        .unwrap()
        .cases
}

fn normalized(blocks: &[AssistantContentBlock]) -> Vec<Value> {
    blocks.iter().map(|block| match block {
        AssistantContentBlock::Text { text } => json!({"kind":"text", "text":text}),
        AssistantContentBlock::Reasoning { text } => json!({"kind":"reasoning", "text":text}),
        AssistantContentBlock::ToolCall(call) => json!({"kind":"tool_call", "name":call.name, "arguments":serde_json::from_str::<Value>(&call.arguments).unwrap()}),
    }).collect()
}

async fn run(
    case: &Case,
    tool: ParserSelection,
    reasoning: ParserSelection,
    chunks: &[&str],
    choice: ChatToolChoice,
) -> Vec<ChatEvent> {
    let mut request = ChatRequest {
        tool_context: ResolvedToolContext::new(&[], case.tools.clone(), Some(choice), true)
            .unwrap(),
        ..ChatRequest::for_test()
    };
    let mut processor = DefaultChatOutputProcessor::new(
        &mut request,
        &case.model,
        Arc::new(TestTokenizer::new()),
        &tool,
        &reasoning,
        None,
    )
    .unwrap();
    assert_eq!(request.decode_options.skip_special_tokens, Some(false));
    processor.initialize(&[]).unwrap();
    let events = chunks
        .iter()
        .map(|chunk| {
            Ok(DecodedTextEvent::TextDelta {
                decoded: DecodedText::unattributed(*chunk),
                sampled: SampledDelta::default(),
                finished: None,
            })
        })
        .chain([Ok(DecodedTextEvent::TextDelta {
            decoded: DecodedText::default(),
            sampled: SampledDelta::default(),
            finished: Some(Box::new(Finished {
                usage: vllm_llm::TokenUsage::default(),
                finish_reason: vllm_chat::FinishReason::Stop(None),
                kv_transfer_params: None,
                ec_transfer_params: None,
                routed_experts: None,
                spec_decode_metrics: None,
                weight_versions: None,
                timestamps: Default::default(),
            })),
        })])
        .collect::<Vec<_>>();
    Box::new(processor)
        .process(stream::iter(events).boxed())
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn assert_done(events: &[ChatEvent], expected: &[Value], label: &str) {
    let Some(ChatEvent::Done { message, usage, .. }) = events.last() else {
        panic!("missing Done: {label}");
    };
    assert_eq!(normalized(message), expected, "{label}");
    assert_eq!(
        usage.reasoning_tokens, None,
        "unattributed reasoning must not claim zero tokens"
    );
}

// The expected event order comes from the pinned frontend-crates corpus, while
// the assertions run vLLM selection, streaming assembly and its final message.
#[tokio::test]
#[ignore = "requires VLLM_USE_DYNAMO_RUST_PARSER=1"]
async fn all_registered_unified_names_reach_vllm_output() {
    assert_eq!(
        std::env::var("VLLM_USE_DYNAMO_RUST_PARSER").as_deref(),
        Ok("1"),
        "run this separate test binary with the experimental flag enabled"
    );
    let cases = fixtures();
    for name in
        REGISTERED_UNIFIED_FAMILIES
            .iter()
            .copied()
            .chain(["qwen3_xml", "muse", "musespark"])
    {
        let family = match name {
            "qwen3_xml" => "qwen3",
            "muse" | "musespark" => "muse_spark",
            _ => canonical_unified_family(name).unwrap(),
        };
        let case = cases.iter().find(|case| case.family == family).unwrap();
        let explicit = ParserSelection::Explicit(name.to_string());
        vllm_chat::validate_parser_overrides(&explicit, &explicit).unwrap();
        let events = run(
            case,
            explicit.clone(),
            explicit,
            &[&case.input],
            ChatToolChoice::Auto,
        )
        .await;
        assert_done(&events, &case.expected, name);
    }
}

#[tokio::test]
#[ignore = "requires VLLM_USE_DYNAMO_RUST_PARSER=1"]
async fn automatic_unified_selection_preserves_every_utf8_split() {
    for case in fixtures() {
        for at in case.input.char_indices().map(|(at, _)| at).chain([case.input.len()]) {
            let events = run(
                &case,
                ParserSelection::Auto,
                ParserSelection::Auto,
                &[&case.input[..at], &case.input[at..]],
                ChatToolChoice::Auto,
            )
            .await;
            assert_done(
                &events,
                &case.expected,
                &format!("{} split {at}", case.family),
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires VLLM_USE_DYNAMO_RUST_PARSER=1"]
async fn unified_tool_choice_none_preserves_reasoning_and_text() {
    for case in fixtures() {
        let expected = case
            .expected
            .iter()
            .filter(|item| item["kind"] != "tool_call")
            .cloned()
            .collect::<Vec<_>>();
        let events = run(
            &case,
            ParserSelection::Auto,
            ParserSelection::Auto,
            &[&case.input],
            ChatToolChoice::None,
        )
        .await;
        assert_done(&events, &expected, &case.family);
    }
}

#[tokio::test]
#[ignore = "requires VLLM_USE_DYNAMO_RUST_PARSER=1"]
async fn unified_grammars_build_at_the_vllm_request_boundary() {
    for case in fixtures() {
        let mut tools = case
            .tools
            .iter()
            .filter(|tool| tool.name == "get_weather")
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 1);
        tools[0].parameters = json!({"type":"object","properties":{"city":{"type":"string","enum":["Paris"]}},"required":["city"],"additionalProperties":false});
        let mut request = ChatRequest {
            tool_context: ResolvedToolContext::new(&[], tools, Some(ChatToolChoice::Auto), true)
                .unwrap(),
            ..ChatRequest::for_test()
        };
        let explicit = ParserSelection::Explicit(case.family.clone());
        let mut processor = DefaultChatOutputProcessor::new(
            &mut request,
            &case.model,
            Arc::new(TestTokenizer::new()),
            &explicit,
            &explicit,
            Some(vllm_chat::ToolStrictLevel::Parameter),
        )
        .unwrap();
        processor.initialize(&[]).unwrap();
        let result = processor.build_output_grammar();
        if [
            "gemma4",
            "muse_glimmer",
            "nemotron_v3",
            "hermes",
            "mistral_4",
        ]
        .contains(&case.family.as_str())
        {
            assert!(
                result.is_err(),
                "unsupported grammar must be rejected: {}",
                case.family
            );
        } else {
            let grammar = result.unwrap().unwrap();
            if let Ok(root) = std::env::var("DYNAMO_VLLM_GRAMMAR_DIR") {
                let path = std::path::Path::new(&root).join(format!("{}.json", case.family));
                std::fs::write(path,serde_json::to_vec(&json!({"family":case.family,"format":grammar.format,"input":case.call_only_input})).unwrap()).unwrap();
            }
        }
    }
}
