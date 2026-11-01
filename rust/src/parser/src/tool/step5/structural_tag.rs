// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use xgrammar_structural_tag::builders::{
    ReasoningMode, StructuralTagBuilder, StructuralTagContext,
};
use xgrammar_structural_tag::format::{Format, StructuralTag};
use xgrammar_structural_tag::{Model, Result};

/// Step-5 shares Qwen Coder's tool grammar and uses bare reasoning boundaries.
pub(super) struct Step5StructuralTagBuilder;

impl StructuralTagBuilder for Step5StructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        let tools = Model::Qwen3Coder.builder().build(StructuralTagContext {
            options: ctx.options.with_reasoning(false),
            ..ctx
        })?;
        if ctx.options.reasoning == ReasoningMode::Disabled {
            return Ok(tools);
        }

        let mut reasoning = Format::tag(
            if ctx.options.reasoning == ReasoningMode::Auto {
                "<think>"
            } else {
                ""
            },
            Format::any_text_excluding(if ctx.options.exclude_special_tokens {
                &["<think>", "</think>"]
            } else {
                &[]
            }),
            "</think>",
        );
        if ctx.options.reasoning == ReasoningMode::Auto {
            reasoning = Format::optional(reasoning);
        }
        Ok(StructuralTag::new(Format::sequence(vec![
            reasoning,
            tools.format,
        ])))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use xgrammar_structural_tag::builders::StructuralTagOptions;
    use xgrammar_structural_tag::{
        FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
    };

    use super::*;
    use crate::tool::test_utils::test_tokenizer;
    use crate::tool::{Step5ToolParser, ToolParser};

    #[test]
    fn structural_tag_keeps_qwen_tools_with_bare_reasoning_boundaries() {
        let parser = Step5ToolParser::create(&[], test_tokenizer()).unwrap();
        let builder = parser.structural_tag_builder().unwrap();
        for strict in [false, true] {
            let tools = [ToolParam::Function(FunctionToolParam::new(
                FunctionDefinition::new("lookup")
                    .with_parameters(json!({
                        "type": "object", "properties": {"count": {"type": "integer"}},
                        "required": ["count"], "additionalProperties": false
                    }))
                    .with_strict(strict),
            ))];
            for choice in [
                ToolChoice::auto(),
                ToolChoice::required(),
                ToolChoice::function("lookup"),
            ] {
                for parallel in [false, true] {
                    let options = StructuralTagOptions::default()
                        .with_reasoning(false)
                        .with_parallel_tool_calls(parallel);
                    let tool_tag =
                        build_structural_tag(Model::Qwen3Coder, &tools, choice.clone(), options)
                            .unwrap();
                    assert_eq!(
                        build_structural_tag(builder, &tools, choice.clone(), options).unwrap(),
                        tool_tag
                    );
                    for mode in [ReasoningMode::Enabled, ReasoningMode::Auto] {
                        let tag = build_structural_tag(
                            builder,
                            &tools,
                            choice.clone(),
                            options.with_reasoning(mode),
                        )
                        .unwrap();
                        let tag = serde_json::to_value(tag).unwrap();
                        let elements = tag["format"]["elements"].as_array().unwrap();
                        assert_eq!(elements.len(), 2);
                        let prefix = &elements[0];
                        let prefix = if mode == ReasoningMode::Auto {
                            assert_eq!(prefix["type"], "optional");
                            &prefix["content"]
                        } else {
                            prefix
                        };
                        assert_eq!(prefix["type"], "tag");
                        assert_eq!(
                            prefix["begin"],
                            if mode == ReasoningMode::Auto {
                                "<think>"
                            } else {
                                ""
                            }
                        );
                        assert_eq!(prefix["end"], "</think>");
                        assert_eq!(elements[1], serde_json::to_value(&tool_tag.format).unwrap());
                    }
                }
            }
        }
    }
}
