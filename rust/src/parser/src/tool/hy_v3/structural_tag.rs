// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for HY3 XML-style tool calls.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{StructuralTagBuilder, StructuralTagContext};
use xgrammar_structural_tag::format::{Format, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam};

use super::HyV3ToolMarkers;

/// HY3 structural-tag builder using tokenizer-specific structural markers.
#[derive(Debug, Clone)]
pub(super) struct HyV3StructuralTagBuilder {
    markers: Arc<HyV3ToolMarkers>,
}

impl HyV3StructuralTagBuilder {
    pub(super) fn new(markers: Arc<HyV3ToolMarkers>) -> Self {
        Self { markers }
    }

    fn argument_pair(&self, key: &str) -> Format {
        let excludes = self.markers.iter().collect::<Vec<_>>();
        Format::sequence(vec![
            Format::const_string(&self.markers.arg_key_start),
            Format::const_string(key),
            Format::const_string(&self.markers.arg_key_end),
            Format::const_string("\n"),
            Format::const_string(&self.markers.arg_value_start),
            Format::any_text_excluding(&excludes),
            Format::const_string(&self.markers.arg_value_end),
            Format::const_string("\n"),
        ])
    }

    fn tool_call(&self, tool: &FunctionToolParam) -> TagFormat {
        let (required_keys, optional_keys) = argument_keys(tool.function.parameters.as_ref());
        let mut elements =
            required_keys.into_iter().map(|key| self.argument_pair(key)).collect::<Vec<_>>();

        if !optional_keys.is_empty() {
            let mut pairs =
                optional_keys.into_iter().map(|key| self.argument_pair(key)).collect::<Vec<_>>();
            let optional = if pairs.len() == 1 {
                pairs.pop().unwrap()
            } else {
                Format::or(pairs)
            };
            elements.push(Format::star(optional));
        }

        let content = if elements.is_empty() {
            Format::any_text()
        } else {
            Format::sequence(elements)
        };
        TagFormat::new(
            format!(
                "{}{}{}\n",
                self.markers.tool_call_start, tool.function.name, self.markers.tool_sep
            ),
            content,
            self.markers.tool_call_end.clone(),
        )
    }

    fn tool_calls(&self, tools: &[FunctionToolParam], choice: BuilderToolChoice) -> Format {
        let mut calls = tools.iter().map(|tool| self.tool_call(tool)).collect::<Vec<_>>();
        let begin = format!("{}\n", self.markers.tool_calls_start);
        let end = format!("\n{}", self.markers.tool_calls_end);

        match choice {
            BuilderToolChoice::Auto if calls.is_empty() => Format::any_text(),
            BuilderToolChoice::Auto => {
                let outer = TagFormat::new(
                    begin,
                    Format::tags_with_separator(calls, "\n", true, false),
                    end,
                );
                Format::triggered_tags(&[&self.markers.tool_calls_start], vec![outer])
            }
            BuilderToolChoice::Forced => Format::sequence(vec![
                Format::const_string(begin),
                Format::Tag(calls.pop().unwrap()),
                Format::const_string(end),
            ]),
            BuilderToolChoice::Required => Format::sequence(vec![
                Format::const_string(begin),
                Format::tags_with_separator(calls, "\n", true, false),
                Format::const_string(end),
            ]),
        }
    }
}

impl StructuralTagBuilder for HyV3StructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        Ok(StructuralTag::new(
            self.tool_calls(ctx.function_tools, ctx.tool_choice),
        ))
    }
}

/// Split argument keys into required and optional declaration-order groups.
fn argument_keys(parameters: Option<&Value>) -> (Vec<&str>, Vec<&str>) {
    let Some(parameters) = parameters.and_then(Value::as_object) else {
        return (Vec::new(), Vec::new());
    };
    let properties = parameters.get("properties").and_then(Value::as_object);
    let required = parameters
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();

    let mut required_keys = properties
        .into_iter()
        .flat_map(|properties| properties.keys())
        .map(String::as_str)
        .filter(|key| required_set.contains(key))
        .collect::<Vec<_>>();
    required_keys.extend(
        required
            .into_iter()
            .filter(|key| properties.is_none_or(|properties| !properties.contains_key(*key))),
    );
    let optional_keys = properties
        .into_iter()
        .flat_map(|properties| properties.keys())
        .map(String::as_str)
        .filter(|key| !required_set.contains(key))
        .collect();
    (required_keys, optional_keys)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use xgrammar_structural_tag::builders::StructuralTagOptions;
    use xgrammar_structural_tag::{
        FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
    };

    use super::HyV3StructuralTagBuilder;
    use crate::tool::HyV3ToolMarkers;

    fn tool(name: &str, parameters: Value) -> ToolParam {
        ToolParam::Function(FunctionToolParam::new(
            FunctionDefinition::new(name).with_parameters(parameters),
        ))
    }

    fn build(
        suffix: &str,
        tools: &[ToolParam],
        choice: ToolChoice,
    ) -> xgrammar_structural_tag::format::StructuralTag {
        build_structural_tag(
            HyV3StructuralTagBuilder::new(Arc::new(HyV3ToolMarkers::new(suffix))),
            tools,
            choice,
            StructuralTagOptions::default().with_reasoning(false),
        )
        .unwrap()
    }

    #[test]
    fn required_uses_suffixed_hy3_skeleton_and_bounded_values() {
        let tag = build(
            ":opensource",
            &[tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "days": { "type": "integer" }
                    },
                    "required": ["city"]
                }),
            )],
            ToolChoice::required(),
        );

        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"<tool_calls:opensource>\n"},{"type":"tags_with_separator","tags":[{"begin":"<tool_call:opensource>get_weather<tool_sep:opensource>\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key:opensource>"},{"type":"const_string","value":"city"},{"type":"const_string","value":"</arg_key:opensource>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value:opensource>"},{"type":"any_text","excludes":["<tool_calls:opensource>","</tool_calls:opensource>","<tool_call:opensource>","</tool_call:opensource>","<tool_sep:opensource>","<arg_key:opensource>","</arg_key:opensource>","<arg_value:opensource>","</arg_value:opensource>"]},{"type":"const_string","value":"</arg_value:opensource>"},{"type":"const_string","value":"\n"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key:opensource>"},{"type":"const_string","value":"days"},{"type":"const_string","value":"</arg_key:opensource>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value:opensource>"},{"type":"any_text","excludes":["<tool_calls:opensource>","</tool_calls:opensource>","<tool_call:opensource>","</tool_call:opensource>","<tool_sep:opensource>","<arg_key:opensource>","</arg_key:opensource>","<arg_value:opensource>","</arg_value:opensource>"]},{"type":"const_string","value":"</arg_value:opensource>"},{"type":"const_string","value":"\n"}]}}]},"end":"</tool_call:opensource>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</tool_calls:opensource>"}]}}"#]].assert_eq(&tag.to_json_string().unwrap());
    }

    #[test]
    fn optional_only_schema_keeps_declared_key_alternatives() {
        let tag = build(
            "",
            &[tool(
                "lookup",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "integer" }
                    }
                }),
            )],
            ToolChoice::required(),
        );
        let value = serde_json::to_value(tag).unwrap();

        expect![[r#"{"type":"sequence","elements":[{"type":"star","content":{"type":"or","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key>"},{"type":"const_string","value":"query"},{"type":"const_string","value":"</arg_key>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value>"},{"type":"any_text","excludes":["<tool_calls>","</tool_calls>","<tool_call>","</tool_call>","<tool_sep>","<arg_key>","</arg_key>","<arg_value>","</arg_value>"]},{"type":"const_string","value":"</arg_value>"},{"type":"const_string","value":"\n"}]},{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key>"},{"type":"const_string","value":"limit"},{"type":"const_string","value":"</arg_key>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value>"},{"type":"any_text","excludes":["<tool_calls>","</tool_calls>","<tool_call>","</tool_call>","<tool_sep>","<arg_key>","</arg_key>","<arg_value>","</arg_value>"]},{"type":"const_string","value":"</arg_value>"},{"type":"const_string","value":"\n"}]}]}}]}"#]].assert_eq(
            &serde_json::to_string(
                &value["format"]["elements"][1]["tags"][0]["content"],
            )
            .unwrap(),
        );
    }

    #[test]
    fn auto_and_forced_preserve_tool_choice_shape() {
        let tools = [
            tool("search", json!({ "type": "object" })),
            tool("lookup", json!({ "type": "object" })),
        ];
        let auto = build("", &tools, ToolChoice::auto()).to_json_string().unwrap();
        let forced = build("", &tools, ToolChoice::function("lookup")).to_json_string().unwrap();

        expect![[r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<tool_calls>"],"tags":[{"begin":"<tool_calls>\n","content":{"type":"tags_with_separator","tags":[{"begin":"<tool_call>search<tool_sep>\n","content":{"type":"any_text","excludes":[]},"end":"</tool_call>"},{"begin":"<tool_call>lookup<tool_sep>\n","content":{"type":"any_text","excludes":[]},"end":"</tool_call>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},"end":"\n</tool_calls>"}],"at_least_one":false,"stop_after_first":false,"excludes":[]}}"#]].assert_eq(&auto);
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"<tool_calls>\n"},{"type":"tag","begin":"<tool_call>lookup<tool_sep>\n","content":{"type":"any_text","excludes":[]},"end":"</tool_call>"},{"type":"const_string","value":"\n</tool_calls>"}]}}"#]].assert_eq(&forced);
    }

    #[test]
    fn missing_required_property_is_still_emitted() {
        let tag = build(
            "",
            &[tool(
                "search",
                json!({
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query", "tenant"]
                }),
            )],
            ToolChoice::required(),
        );
        let value = serde_json::to_value(tag).unwrap();

        expect![[r#"{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key>"},{"type":"const_string","value":"query"},{"type":"const_string","value":"</arg_key>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value>"},{"type":"any_text","excludes":["<tool_calls>","</tool_calls>","<tool_call>","</tool_call>","<tool_sep>","<arg_key>","</arg_key>","<arg_value>","</arg_value>"]},{"type":"const_string","value":"</arg_value>"},{"type":"const_string","value":"\n"}]},{"type":"sequence","elements":[{"type":"const_string","value":"<arg_key>"},{"type":"const_string","value":"tenant"},{"type":"const_string","value":"</arg_key>"},{"type":"const_string","value":"\n"},{"type":"const_string","value":"<arg_value>"},{"type":"any_text","excludes":["<tool_calls>","</tool_calls>","<tool_call>","</tool_call>","<tool_sep>","<arg_key>","</arg_key>","<arg_value>","</arg_value>"]},{"type":"const_string","value":"</arg_value>"},{"type":"const_string","value":"\n"}]}]}"#]].assert_eq(
            &serde_json::to_string(
                &value["format"]["elements"][1]["tags"][0]["content"],
            )
            .unwrap(),
        );
    }
}
