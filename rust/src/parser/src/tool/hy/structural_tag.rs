// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for HY XML-style tool calls.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{StructuralTagBuilder, StructuralTagContext};
use xgrammar_structural_tag::format::{Format, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam};

use super::{HyDialect, HyToolMarkers};

/// HY structural-tag builder using tokenizer-specific structural markers.
#[derive(Debug, Clone)]
pub(super) struct HyStructuralTagBuilder {
    markers: Arc<HyToolMarkers>,
    dialect: HyDialect,
}

impl HyStructuralTagBuilder {
    pub(super) fn new(markers: Arc<HyToolMarkers>, dialect: HyDialect) -> Self {
        Self { markers, dialect }
    }

    fn argument_pair(&self, key: &str) -> Format {
        let excludes = self.markers.iter().collect::<Vec<_>>();
        let separator = self.dialect.separator();
        let mut elements = vec![
            Format::const_string(&self.markers.arg_key_start),
            Format::const_string(key),
            Format::const_string(&self.markers.arg_key_end),
        ];
        if !separator.is_empty() {
            elements.push(Format::const_string(separator));
        }
        elements.extend([
            Format::const_string(&self.markers.arg_value_start),
            Format::any_text_excluding(&excludes),
            Format::const_string(&self.markers.arg_value_end),
        ]);
        if !separator.is_empty() {
            elements.push(Format::const_string(separator));
        }
        Format::sequence(elements)
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
            match self.dialect {
                HyDialect::V3 => Format::any_text(),
                HyDialect::V4 => Format::const_string(""),
            }
        } else {
            Format::sequence(elements)
        };
        let tool_sep = self.markers.tool_sep.as_deref().unwrap_or_default();
        TagFormat::new(
            format!(
                "{}{}{}{}",
                self.markers.tool_call_start,
                tool.function.name,
                tool_sep,
                self.dialect.separator(),
            ),
            content,
            self.markers.tool_call_end.clone(),
        )
    }

    fn tool_calls(&self, tools: &[FunctionToolParam], choice: BuilderToolChoice) -> Format {
        let mut calls = tools.iter().map(|tool| self.tool_call(tool)).collect::<Vec<_>>();
        let separator = self.dialect.separator();
        let begin = format!("{}{separator}", self.markers.tool_calls_start);
        let end = format!("{separator}{}", self.markers.tool_calls_end);

        match choice {
            BuilderToolChoice::Auto if calls.is_empty() => Format::any_text(),
            BuilderToolChoice::Auto => {
                let outer = TagFormat::new(
                    begin,
                    Format::tags_with_separator(calls, separator, true, false),
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
                Format::tags_with_separator(calls, separator, true, false),
                Format::const_string(end),
            ]),
        }
    }
}

impl StructuralTagBuilder for HyStructuralTagBuilder {
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

    use super::HyStructuralTagBuilder;
    use crate::output_grammar::test_utils::outline;
    use crate::tool::{HyDialect, HyToolMarkers};

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
        build_for_dialect(HyDialect::V3, suffix, tools, choice)
    }

    fn build_for_dialect(
        dialect: HyDialect,
        suffix: &str,
        tools: &[ToolParam],
        choice: ToolChoice,
    ) -> xgrammar_structural_tag::format::StructuralTag {
        build_structural_tag(
            HyStructuralTagBuilder::new(Arc::new(HyToolMarkers::new(suffix, dialect)), dialect),
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

        expect![[r#"
            sequence
              `<tool_calls:opensource>\n`
              tags_with_separator `\n` at_least_one
                tag `<tool_call:opensource>get_weather<tool_sep:opensource>\n` .. `</tool_call:opensource>`
                  sequence
                    sequence
                      `<arg_key:opensource>`
                      `city`
                      `</arg_key:opensource>`
                      `\n`
                      `<arg_value:opensource>`
                      text excluding [`<tool_calls:opensource>`, `</tool_calls:opensource>`, `<tool_call:opensource>`, `</tool_call:opensource>`, `<tool_sep:opensource>`, `<arg_key:opensource>`, `</arg_key:opensource>`, `<arg_value:opensource>`, `</arg_value:opensource>`]
                      `</arg_value:opensource>`
                      `\n`
                    star
                      sequence
                        `<arg_key:opensource>`
                        `days`
                        `</arg_key:opensource>`
                        `\n`
                        `<arg_value:opensource>`
                        text excluding [`<tool_calls:opensource>`, `</tool_calls:opensource>`, `<tool_call:opensource>`, `</tool_call:opensource>`, `<tool_sep:opensource>`, `<arg_key:opensource>`, `</arg_key:opensource>`, `<arg_value:opensource>`, `</arg_value:opensource>`]
                        `</arg_value:opensource>`
                        `\n`
              `\n</tool_calls:opensource>`
        "#]].assert_eq(&outline(&tag.format));
    }

    #[test]
    fn required_uses_suffixed_compact_hy4_skeleton_and_bounded_values() {
        let tag = build_for_dialect(
            HyDialect::V4,
            ":opensource",
            &[tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            )],
            ToolChoice::required(),
        );

        expect![[r#"
            sequence
              `<tool_calls:opensource>`
              tags_with_separator `` at_least_one
                tag `<tool_call:opensource>get_weather` .. `</tool_call:opensource>`
                  sequence
                    sequence
                      `<arg_key:opensource>`
                      `city`
                      `</arg_key:opensource>`
                      `<arg_value:opensource>`
                      text excluding [`<tool_calls:opensource>`, `</tool_calls:opensource>`, `<tool_call:opensource>`, `</tool_call:opensource>`, `<arg_key:opensource>`, `</arg_key:opensource>`, `<arg_value:opensource>`, `</arg_value:opensource>`]
                      `</arg_value:opensource>`
              `</tool_calls:opensource>`
        "#]].assert_eq(&outline(&tag.format));
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
        expect![[r#"
            sequence
              `<tool_calls>\n`
              tags_with_separator `\n` at_least_one
                tag `<tool_call>lookup<tool_sep>\n` .. `</tool_call>`
                  sequence
                    star
                      or
                        sequence
                          `<arg_key>`
                          `query`
                          `</arg_key>`
                          `\n`
                          `<arg_value>`
                          text excluding [`<tool_calls>`, `</tool_calls>`, `<tool_call>`, `</tool_call>`, `<tool_sep>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`]
                          `</arg_value>`
                          `\n`
                        sequence
                          `<arg_key>`
                          `limit`
                          `</arg_key>`
                          `\n`
                          `<arg_value>`
                          text excluding [`<tool_calls>`, `</tool_calls>`, `<tool_call>`, `</tool_call>`, `<tool_sep>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`]
                          `</arg_value>`
                          `\n`
              `\n</tool_calls>`
        "#]].assert_eq(&outline(&tag.format));
    }

    #[test]
    fn auto_and_forced_preserve_tool_choice_shape() {
        let tools = [
            tool("search", json!({ "type": "object" })),
            tool("lookup", json!({ "type": "object" })),
        ];
        let auto = build("", &tools, ToolChoice::auto());
        let forced = build("", &tools, ToolChoice::function("lookup"));

        expect![[r#"
            triggered_tags [`<tool_calls>`]
              tag `<tool_calls>\n` .. `\n</tool_calls>`
                tags_with_separator `\n` at_least_one
                  tag `<tool_call>search<tool_sep>\n` text `</tool_call>`
                  tag `<tool_call>lookup<tool_sep>\n` text `</tool_call>`
        "#]]
        .assert_eq(&outline(&auto.format));
        expect![[r#"
            sequence
              `<tool_calls>\n`
              tag `<tool_call>lookup<tool_sep>\n` text `</tool_call>`
              `\n</tool_calls>`
        "#]]
        .assert_eq(&outline(&forced.format));
    }

    #[test]
    fn hy_v4_zero_argument_tool_has_empty_compact_body() {
        let tag = build_for_dialect(
            HyDialect::V4,
            "",
            &[tool(
                "get_current_date",
                json!({ "type": "object", "properties": {} }),
            )],
            ToolChoice::function("get_current_date"),
        );
        expect![[r#"
            sequence
              `<tool_calls>`
              tag `<tool_call>get_current_date` `` `</tool_call>`
              `</tool_calls>`
        "#]]
        .assert_eq(&outline(&tag.format));
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
        expect![[r#"
            sequence
              `<tool_calls>\n`
              tags_with_separator `\n` at_least_one
                tag `<tool_call>search<tool_sep>\n` .. `</tool_call>`
                  sequence
                    sequence
                      `<arg_key>`
                      `query`
                      `</arg_key>`
                      `\n`
                      `<arg_value>`
                      text excluding [`<tool_calls>`, `</tool_calls>`, `<tool_call>`, `</tool_call>`, `<tool_sep>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`]
                      `</arg_value>`
                      `\n`
                    sequence
                      `<arg_key>`
                      `tenant`
                      `</arg_key>`
                      `\n`
                      `<arg_value>`
                      text excluding [`<tool_calls>`, `</tool_calls>`, `<tool_call>`, `</tool_call>`, `<tool_sep>`, `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`]
                      `</arg_value>`
                      `\n`
              `\n</tool_calls>`
        "#]].assert_eq(&outline(&tag.format));
    }
}
