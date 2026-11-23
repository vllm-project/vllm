// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeSet, VecDeque};

use dynamo_parsers_v2::unified::{
    UnifiedParser as DynamoParser, UnifiedParserEvent as DynamoEvent, UnifiedParserInit,
    UnifiedParserOutput as DynamoOutput, UnifiedParserStartingState,
};
use dynamo_parsers_v2::{Tool as DynamoTool, create_unified_parser_for_family};
use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{Result, UnifiedParser, UnifiedParserError, UnifiedParserOutput};
use crate::output_grammar::{self, BuiltOutputGrammar, OutputGrammarContext, ToolStrictLevel};
use crate::tool::{Tool, ToolCallDelta};

pub use dynamo_parsers_v2::{REGISTERED_UNIFIED_FAMILIES, canonical_unified_family};

/// Bridges only Dynamo v2's ordered unified event stream.
pub struct DynamoUnifiedParser {
    family: &'static str,
    inner: Box<dyn DynamoParser>,
    tools: Vec<DynamoTool>,
    tokenizer: DynTokenizer,
    starting_state: UnifiedParserStartingState,
    token_input: Option<bool>,
    token_ids_compatible: bool,
    pending: VecDeque<DynamoEvent>,
    completed: BTreeSet<usize>,
    prompt_prefix: String,
}

impl DynamoUnifiedParser {
    pub fn new(family: &'static str, tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let tools = tools
            .iter()
            .map(|tool| DynamoTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            })
            .collect::<Vec<_>>();
        let inner = create_unified_parser_for_family(family, &tools).map_err(dynamo_error)?;
        let token_ids_compatible = inner.input_capabilities().token_ids
            && ["<|start|>", "<|message|>", "<|channel|>"].iter().all(|marker| {
                dynamo_parsers_v2::encode_harmony(marker).is_ok_and(|ids| {
                    ids.len() == 1 && tokenizer.token_to_id(marker) == Some(ids[0])
                })
            });
        Ok(Self {
            family,
            inner,
            tools,
            tokenizer,
            starting_state: UnifiedParserStartingState::None,
            token_input: None,
            token_ids_compatible,
            pending: VecDeque::new(),
            completed: BTreeSet::new(),
            prompt_prefix: String::new(),
        })
    }

    fn project(&mut self, raw: DynamoOutput, output: &mut UnifiedParserOutput) -> Result<()> {
        for event in raw.events {
            if let DynamoEvent::ToolCall(call) = &event
                && call.complete
            {
                self.completed.insert(call.tool_index);
            }
            self.pending.push_back(event);
        }
        self.drain(output)
    }

    fn drain(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        while let Some(event) = self.pending.front() {
            if let DynamoEvent::ToolCall(call) = event
                && !self.completed.contains(&call.tool_index)
            {
                break;
            }
            match self.pending.pop_front().unwrap() {
                DynamoEvent::Text(text) => output.push_text(text),
                DynamoEvent::Reasoning(text) => {
                    output.push_reasoning(DecodedText::unattributed(text))
                }
                DynamoEvent::ToolCall(first) => {
                    let mut call = ToolCallDelta {
                        tool_index: first.tool_index,
                        name: first.name,
                        arguments: first.arguments,
                    };
                    // vLLM cannot retract provisional calls. Keep their first-event
                    // position, but publish them only after Dynamo validates completion.
                    self.pending.retain(|event| {
                        if let DynamoEvent::ToolCall(delta) = event
                            && delta.tool_index == call.tool_index
                        {
                            if call.name.is_none() {
                                call.name = delta.name.clone();
                            }
                            call.arguments.push_str(&delta.arguments);
                            false
                        } else {
                            true
                        }
                    });
                    self.completed.remove(&call.tool_index);
                    output.push_call(call);
                }
                DynamoEvent::MessageStart(_) | DynamoEvent::MessageEnd => {
                    return Err(dynamo_error(anyhow::anyhow!(
                        "message phases are disabled for chat completions"
                    )));
                }
            }
        }
        Ok(())
    }

    fn drop_incomplete(&mut self, output: &mut UnifiedParserOutput) -> Result<()> {
        let before = self.pending.len();
        self.pending.retain(|event| !matches!(event, DynamoEvent::ToolCall(call) if !self.completed.contains(&call.tool_index)));
        if before != self.pending.len() {
            tracing::warn!(
                why = "dynamo_incomplete_call_not_published",
                family = self.family,
                "discarding unvalidated tool-call fragments"
            );
        }
        self.drain(output)?;
        self.completed.clear();
        Ok(())
    }
}

impl UnifiedParser for DynamoUnifiedParser {
    fn create(_tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>> {
        Err(UnifiedParserError::NoNamedConstructor {
            parser: "dynamo",
            built_from: "a registered v2 unified family",
        })
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        let markers = match self.family {
            "deepseek_v4" | "deepseek_v41" | "glm47" | "qwen3" | "qwen3_coder" | "nemotron_v3"
            | "nemotron3" | "kimi_k2" | "hermes" => Some(("<think>", "</think>")),
            "mistral_4" => Some(("[THINK]", "[/THINK]")),
            "minimax_m3" | "minimax-m3" => Some(("<mm:think>", "</mm:think>")),
            "gemma4" => Some(("<|channel>", "<channel|>")),
            _ => None,
        };
        self.starting_state = markers
            .and_then(|(open, close)| {
                let open = self.tokenizer.token_to_id(open)?;
                let close = self.tokenizer.token_to_id(close)?;
                crate::reasoning::last_reasoning_boundary(
                    prompt_token_ids,
                    open,
                    close,
                    self.tokenizer.as_ref(),
                )
                .map(|reasoning| {
                    if reasoning {
                        UnifiedParserStartingState::Reasoning
                    } else {
                        UnifiedParserStartingState::Response
                    }
                })
            })
            .unwrap_or(UnifiedParserStartingState::None);
        self.inner
            .initialize_request(UnifiedParserInit {
                prompt_token_ids: prompt_token_ids.to_vec(),
                starting_state: self.starting_state,
                ..Default::default()
            })
            .map_err(dynamo_error)?;

        // Envelope parsers own partial headers too. Prime that same state machine
        // with only the active prompt suffix; never emit prompt text as generation.
        let start = match self.family {
            "beam" | "muse_glimmer" | "muse_spark" | "muse_spark_open" => Some("<|start|>"),
            "kimi_k3" | "kimi-k3" => Some("<|open|>"),
            _ => None,
        };
        if let Some(start_id) = start.and_then(|start| self.tokenizer.token_to_id(start))
            && let Some(at) = prompt_token_ids.iter().rposition(|id| *id == start_id)
        {
            let ids = &prompt_token_ids[at..];
            let closed = [
                "<|return|>",
                "<|end|>",
                "<|eom|>",
                "<|eot|>",
                "<|end_of_msg|>",
            ]
            .iter()
            .filter_map(|end| self.tokenizer.token_to_id(end))
            .any(|end| ids.contains(&end));
            if !closed {
                let prefix = self
                    .tokenizer
                    .decode(ids, false)
                    .map_err(|error| dynamo_error(error.into()))?;
                if prefix.starts_with("<|start|>assistant")
                    || prefix.starts_with("<|open|>think")
                    || prefix.starts_with("<|open|>response")
                    || prefix.starts_with("<|open|>message role=\"assistant\"")
                {
                    let mut prompt_output = DynamoOutput::default();
                    self.inner.parse_into(&prefix, &mut prompt_output).map_err(dynamo_error)?;
                    self.pending.extend(prompt_output.events.into_iter().filter(
                        |event| matches!(event, DynamoEvent::ToolCall(call) if !call.complete),
                    ));
                    self.starting_state = if prefix.contains("<|channel|>analysis")
                        || prefix.contains("assistant to=self")
                        || prefix.starts_with("<|open|>think")
                    {
                        UnifiedParserStartingState::Reasoning
                    } else if prefix.contains("<|channel|>final")
                        || prefix.contains("assistant to=user")
                        || prefix.starts_with("<|open|>response")
                    {
                        UnifiedParserStartingState::Response
                    } else {
                        UnifiedParserStartingState::None
                    };
                    self.prompt_prefix = prefix;
                }
            }
        }
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn provides_reasoning_token_counts(&self) -> bool {
        false
    }

    fn default_tool_strict_level(&self) -> ToolStrictLevel {
        if self.family == "glm47" {
            ToolStrictLevel::Function
        } else {
            ToolStrictLevel::Auto
        }
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn build_output_grammar(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
        use dynamo_parsers_v2::structural_tag::{
            PromptChannel, StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
        };
        use xgrammar_structural_tag::ToolChoice;
        use xgrammar_structural_tag::tool::ToolChoiceValue;
        let Some(builder) = dynamo_parsers_v2::structural_tag_builder_for_family(self.family)
        else {
            if output_grammar::tool_grammar_applies(ctx) || ctx.answer.is_some() {
                return Err(output_grammar::OutputGrammarError::Dynamo {
                    error: anyhow::anyhow!(
                        "Dynamo unified family {} has no structural-tag builder",
                        self.family
                    ),
                });
            }
            return Ok(None);
        };
        let choice = match ctx.tool_choice {
            ToolChoice::Value(ToolChoiceValue::None) => StructuralTagToolChoice::None,
            ToolChoice::Value(ToolChoiceValue::Auto) => StructuralTagToolChoice::Auto,
            ToolChoice::Value(ToolChoiceValue::Required) => StructuralTagToolChoice::Required,
            ToolChoice::NamedFunction(function) => {
                StructuralTagToolChoice::Named(&function.function.name)
            }
            _ => {
                return Err(output_grammar::OutputGrammarError::Dynamo {
                    error: anyhow::anyhow!("unsupported Dynamo tool-choice constraint"),
                });
            }
        };
        if matches!(choice, StructuralTagToolChoice::Auto)
            && ctx.tool_strict_level == ToolStrictLevel::Auto
            && !self.tools.iter().any(|tool| tool.strict == Some(true))
            && ctx.answer.is_none()
            && !builder.constrains_every_turn()
        {
            return Ok(None);
        }
        let answer = ctx.answer.map(serde_json::to_value).transpose().map_err(|error| {
            output_grammar::OutputGrammarError::Dynamo {
                error: error.into(),
            }
        })?;
        let answer_schema = match &answer {
            Some(value) => Some(value.get("json_schema").ok_or_else(|| {
                output_grammar::OutputGrammarError::Dynamo {
                    error: anyhow::anyhow!("Dynamo requires a JSON schema answer constraint"),
                }
            })?),
            None => None,
        };
        let channel = match self.starting_state {
            UnifiedParserStartingState::None => PromptChannel::MessageStart,
            UnifiedParserStartingState::Reasoning => PromptChannel::OpenReasoning,
            UnifiedParserStartingState::Response => PromptChannel::OpenAnswer,
        };
        let tag = builder
            .build_turn(
                &StructuralTagContext {
                    tool_choice: choice,
                    tools: &self.tools,
                    parallel_tool_calls: Some(ctx.parallel_tool_calls),
                    schema_mode: if ctx.tool_strict_level == ToolStrictLevel::Parameter {
                        StructuralTagSchemaMode::Strict
                    } else {
                        StructuralTagSchemaMode::Auto
                    },
                    structured_output_schema: answer_schema,
                    starts_in_reasoning: self.starting_state
                        == UnifiedParserStartingState::Reasoning,
                },
                channel,
            )
            .map_err(|error| output_grammar::OutputGrammarError::Dynamo { error })?;
        tag.map(|tag| {
            let format = serde_json::from_value(tag.get("format").cloned().ok_or_else(|| {
                output_grammar::OutputGrammarError::Dynamo {
                    error: anyhow::anyhow!("Dynamo structural tag has no format"),
                }
            })?)
            .map_err(|error| output_grammar::OutputGrammarError::Dynamo {
                error: error.into(),
            })?;
            Ok(BuiltOutputGrammar::from_token_zero(format))
        })
        .transpose()
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        let tokens = *self
            .token_input
            .get_or_insert_with(|| self.token_ids_compatible && !delta.attributions.is_empty());
        let mut raw = DynamoOutput::default();
        let result = if tokens {
            let ids = delta.attributions.iter().map(|item| item.token_id).collect::<Vec<_>>();
            self.inner.parse_token_ids_into(&ids, &mut raw)
        } else {
            self.inner.parse_into(&delta.text, &mut raw)
        };
        self.project(raw, output)?;
        if result.is_err() {
            self.drop_incomplete(output)?;
        }
        result.map_err(dynamo_error)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let raw = self.inner.finish().map_err(dynamo_error)?;
        let mut output = UnifiedParserOutput::default();
        self.project(raw, &mut output)?;
        self.drop_incomplete(&mut output)?;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let mut recovered = self
            .pending
            .drain(..)
            .filter_map(|event| match event {
                DynamoEvent::Text(text) | DynamoEvent::Reasoning(text) => Some(text),
                _ => None,
            })
            .collect::<String>();
        self.completed.clear();
        self.token_input = None;
        self.starting_state = UnifiedParserStartingState::None;
        let raw = self.inner.reset();
        recovered.push_str(raw.strip_prefix(&self.prompt_prefix).unwrap_or(&raw));
        self.prompt_prefix.clear();
        recovered
    }
}

fn dynamo_error(error: anyhow::Error) -> UnifiedParserError {
    UnifiedParserError::Dynamo { error }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified::UnifiedParserEvent;
    use std::sync::Arc;
    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    fn parser(family: &'static str) -> DynamoUnifiedParser {
        DynamoUnifiedParser::new(
            family,
            &crate::tool::test_utils::test_tools(),
            Arc::new(
                TestTokenizer::new()
                    .with_regular_token("<think>", 1001)
                    .with_regular_token("</think>", 1002)
                    .with_special_token("<message>", 1003),
            ),
        )
        .unwrap()
    }

    #[test]
    fn prompt_initialization_stops_at_current_message_boundary() {
        for (prompt, expected) in [
            (
                vec![1001],
                vec![
                    UnifiedParserEvent::Reasoning(DecodedText::unattributed("thought")),
                    UnifiedParserEvent::Text("answer".into()),
                ],
            ),
            (
                vec![1001, 1003],
                vec![UnifiedParserEvent::Text("thoughtanswer".into())],
            ),
        ] {
            let mut parser = parser("qwen3");
            parser.initialize(&prompt).unwrap();
            let mut out = UnifiedParserOutput::default();
            parser
                .parse_into(DecodedText::unattributed("thought</think>answer"), &mut out)
                .unwrap();
            out.append(parser.finish().unwrap());
            assert_eq!(out.events, expected);
        }
    }

    #[test]
    fn completion_projection_does_not_publish_invalidated_or_truncated_calls() {
        let mut parser = parser("qwen3");
        parser.initialize(&[]).unwrap();
        let mut out = UnifiedParserOutput::default();
        parser
            .parse_into(
                DecodedText::unattributed(
                    "<tool_call><function=get_weather><parameter=location>first",
                ),
                &mut out,
            )
            .unwrap();
        assert!(out.events.is_empty());
        parser
            .parse_into(
                DecodedText::unattributed(
                    "</parameter><parameter=location>second</parameter></function></tool_call>",
                ),
                &mut out,
            )
            .unwrap();
        out.append(parser.finish().unwrap());
        assert!(out.events.is_empty());
        parser.reset();
        parser.initialize(&[]).unwrap();
        parser
            .parse_into(
                DecodedText::unattributed(
                    "<tool_call><function=get_weather><parameter=location>partial",
                ),
                &mut out,
            )
            .unwrap();
        out.append(parser.finish().unwrap());
        assert!(out.events.is_empty());
    }

    #[test]
    fn invalidated_call_does_not_hide_valid_successor_or_surrounding_events() {
        let mut parser = parser("qwen3");
        parser.initialize(&[]).unwrap();
        let mut out = UnifiedParserOutput::default();
        parser
            .parse_into(
                DecodedText::unattributed(
                    "<tool_call><function=get_weather><parameter=location>Paris",
                ),
                &mut out,
            )
            .unwrap();
        parser.parse_into(DecodedText::unattributed("</parameter><parameter=location>London</parameter></function></tool_call><think>retry</think><tool_call><function=get_weather><parameter=location>NYC</parameter></function></tool_call>done"), &mut out).unwrap();
        out.append(parser.finish().unwrap());
        let [
            UnifiedParserEvent::Reasoning(reason),
            UnifiedParserEvent::ToolCall(call),
            UnifiedParserEvent::Text(text),
        ] = out.events.as_slice()
        else {
            panic!("valid successor and surrounding events must survive");
        };
        assert_eq!(reason.text, "retry");
        assert_eq!(call.arguments, r#"{"location":"NYC"}"#);
        assert_eq!(text, "done");
    }

    #[test]
    fn rendered_prompt_prefills_reach_each_native_channel() {
        for (family, prefix, suffix) in [
            ("minimax_m3", "<mm:think>", "thought</mm:think>answer"),
            (
                "beam",
                "<|start|>assistant<|channel|>analysis<|message|>",
                "thought<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>",
            ),
            (
                "muse_glimmer",
                "<|start|>assistant to=self<|message|>",
                "thought<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            ),
            (
                "muse_spark",
                "<|start|>assistant to=self<|message|>",
                "thought<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            ),
            (
                "kimi_k3",
                "<|open|>think<|sep|>",
                "thought<|close|>think<|sep|><|open|>response<|sep|>answer<|close|>response<|sep|>",
            ),
        ] {
            let mut tokenizer = TestTokenizer::new();
            for (index, marker) in [
                "<mm:think>",
                "</mm:think>",
                "<|start|>",
                "<|channel|>",
                "<|message|>",
                "<|open|>",
                "<|sep|>",
            ]
            .iter()
            .enumerate()
            {
                tokenizer = tokenizer.with_regular_token(*marker, 2000 + index as u32);
            }
            let ids = tokenizer.encode(prefix, false).unwrap();
            let mut parser = DynamoUnifiedParser::new(family, &[], Arc::new(tokenizer)).unwrap();
            parser.initialize(&ids).unwrap();
            let mut out = UnifiedParserOutput::default();
            parser.parse_into(DecodedText::unattributed(suffix), &mut out).unwrap();
            out.append(parser.finish().unwrap());
            assert_eq!(
                out.events,
                vec![
                    UnifiedParserEvent::Reasoning(DecodedText::unattributed("thought")),
                    UnifiedParserEvent::Text("answer".into())
                ],
                "{family}"
            );
        }
    }

    #[test]
    fn unsupported_structured_answers_fail_before_generation() {
        let answer = serde_json::from_value(
            serde_json::json!({"type":"json_schema","json_schema":{"type":"object"}}),
        )
        .unwrap();
        let mut context = OutputGrammarContext {
            tools: &[],
            tool_choice: &xgrammar_structural_tag::ToolChoice::none(),
            tool_strict_level: ToolStrictLevel::Auto,
            parallel_tool_calls: true,
            answer: Some(&answer),
        };
        for family in ["gemma4", "muse_glimmer", "nemotron_v3"] {
            assert!(
                parser(family).build_output_grammar(&context).is_err(),
                "{family}"
            );
        }
        context.answer = None;
        for family in ["gemma4", "muse_glimmer", "nemotron_v3"] {
            assert!(
                parser(family).build_output_grammar(&context).unwrap().is_none(),
                "{family}"
            );
        }
    }

    #[test]
    fn reset_returns_raw_uncommitted_input_and_restarts_call_indices() {
        let mut parser = parser("qwen3");
        parser.initialize(&[]).unwrap();
        let partial = "<tool_call><function=get_weather";
        parser
            .parse_into(
                DecodedText::unattributed(partial),
                &mut UnifiedParserOutput::default(),
            )
            .unwrap();
        assert_eq!(parser.reset(), partial);
        parser.initialize(&[]).unwrap();
        let mut out = UnifiedParserOutput::default();
        parser.parse_into(DecodedText::unattributed("<tool_call><function=get_weather><parameter=location>fresh</parameter></function></tool_call>"), &mut out).unwrap();
        out.append(parser.finish().unwrap());
        let [UnifiedParserEvent::ToolCall(call)] = out.events.as_slice() else {
            panic!("expected one validated call");
        };
        assert_eq!(call.tool_index, 0);
        assert_eq!(call.arguments, r#"{"location":"fresh"}"#);
    }

    #[test]
    fn harmony_token_ingestion_matches_text_projection() {
        let input = "<|start|>assistant<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>";
        let mut tokenizer = TestTokenizer::new();
        for marker in ["<|start|>", "<|message|>", "<|channel|>"] {
            let ids = dynamo_parsers_v2::encode_harmony(marker).unwrap();
            assert_eq!(ids.len(), 1);
            tokenizer = tokenizer.with_regular_token(marker, ids[0]);
        }
        let mut tokens = DynamoUnifiedParser::new("gpt_oss", &[], Arc::new(tokenizer)).unwrap();
        tokens.initialize(&[]).unwrap();
        let mut token_output = UnifiedParserOutput::default();
        for id in dynamo_parsers_v2::encode_harmony(input).unwrap() {
            tokens
                .parse_into(
                    DecodedText {
                        text: String::new(),
                        attributions: [vllm_tokenizer::TokenAttribution {
                            token_id: id,
                            anchor: vllm_tokenizer::TokenAnchor::ZeroWidth { byte_offset: 0 },
                        }]
                        .into_iter()
                        .collect(),
                    },
                    &mut token_output,
                )
                .unwrap();
        }
        token_output.append(tokens.finish().unwrap());
        let mut text = parser("gpt_oss");
        text.initialize(&[]).unwrap();
        let mut text_output = UnifiedParserOutput::default();
        text.parse_into(DecodedText::unattributed(input), &mut text_output).unwrap();
        text_output.append(text.finish().unwrap());
        assert_eq!(token_output, text_output);
        assert_eq!(tokens.token_input, Some(true));
        assert_eq!(text.token_input, Some(false));
    }
}
