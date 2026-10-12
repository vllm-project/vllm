// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use super::parameters::ToolSchemas;
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const FUNCTION_START: &str = "<function";
const FUNCTION_END: &str = "</function>";
const PARAM_START: &str = "<param";
const PARAM_END: &str = "</param>";

#[derive(Debug, Clone, Default)]
struct ToolShape {
    allowed: BTreeSet<String>,
    required: BTreeSet<String>,
}

impl ToolShape {
    fn from_tool(tool: &Tool) -> Self {
        let allowed = tool
            .parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default();
        let required = tool
            .parameters
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        Self { allowed, required }
    }
}

/// Tool parser for MiniCPM5 XML-style function calls.
///
/// MiniCPM5 emits one or more blocks shaped like:
///
/// ```text
/// <function name="get_weather"><param name="city">Shanghai</param></function>
/// ```
///
/// The parser buffers incomplete tags, so markers and parameter values may be
/// split at arbitrary decoded chunk boundaries.
pub struct MiniCPM5ToolParser {
    buffer: String,
    emitted_tool_count: usize,
    schemas: ToolSchemas,
    shapes: BTreeMap<String, ToolShape>,
}

impl MiniCPM5ToolParser {
    /// Create a MiniCPM5 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            emitted_tool_count: 0,
            schemas: ToolSchemas::from_tools(tools),
            shapes: tools
                .iter()
                .map(|tool| (tool.name.clone(), ToolShape::from_tool(tool)))
                .collect(),
        }
    }

    fn normalize_buffer(&mut self) {
        self.buffer = self
            .buffer
            .replace('\u{0120}', " ")
            .replace('\u{010a}', "\n")
            .replace("<functionname=", "<function name=")
            .replace("<paramname=", "<param name=");
    }

    fn emit_block(&mut self, block: &str, output: &mut ToolParserOutput) -> Result<()> {
        let Some((name, arguments)) = self.parse_block(block) else {
            output.push_text(block);
            return Ok(());
        };
        let arguments = serde_json::to_string(&arguments)
            .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;
        output.push_call(ToolCallDelta {
            tool_index: self.emitted_tool_count,
            name: Some(name),
            arguments,
        });
        self.emitted_tool_count += 1;
        Ok(())
    }

    fn parse_block(&self, block: &str) -> Option<(String, Map<String, Value>)> {
        let open_end = block.find('>')?;
        let source_name = attribute_value(&block[..=open_end], "name")?.trim().to_string();
        if source_name.is_empty() {
            return None;
        }

        let source_shape = self.shapes.get(&source_name);
        let mut seen = BTreeSet::new();
        let mut raw_params = Vec::new();
        let mut wrapped = Map::new();
        let mut cursor = open_end + 1;
        let body_end = block.rfind(FUNCTION_END)?;

        while let Some(relative_start) = block[cursor..body_end].find(PARAM_START) {
            let start = cursor + relative_start;
            let tag_end = start + block[start..body_end].find('>')?;
            let name = attribute_value(&block[start..=tag_end], "name")?.trim().to_string();
            if name.is_empty() || !seen.insert(name.clone()) {
                return None;
            }
            let value_end = tag_end + 1 + block[tag_end + 1..body_end].find(PARAM_END)?;
            let value = param_text(&block[tag_end + 1..value_end]);
            cursor = value_end + PARAM_END.len();

            let allowed = source_shape.is_some_and(|shape| shape.allowed.contains(&name));
            if matches!(name.as_str(), "properties" | "arguments")
                && source_shape.is_some_and(|shape| !shape.allowed.is_empty() && !allowed)
            {
                for (wrapped_name, wrapped_value) in parse_jsonish_object(&value)? {
                    if source_shape.is_some_and(|shape| shape.allowed.contains(&wrapped_name)) {
                        if !seen.insert(wrapped_name.clone()) {
                            return None;
                        }
                        wrapped.insert(wrapped_name, wrapped_value);
                    }
                }
                continue;
            }
            if source_shape.is_some_and(|shape| !shape.allowed.contains(&name)) {
                continue;
            }
            raw_params.push((name, value));
        }

        let mut arguments = self.schemas.convert_params_with_schema(&source_name, raw_params);
        arguments.extend(wrapped);
        let (name, mut arguments) = self.normalize_alias(source_name, arguments);
        let shape = self.shapes.get(&name)?;
        if !shape.required.iter().all(|key| arguments.contains_key(key)) {
            return None;
        }
        arguments.retain(|key, _| shape.allowed.contains(key));
        Some((name, arguments))
    }

    fn normalize_alias(
        &self,
        source_name: String,
        arguments: Map<String, Value>,
    ) -> (String, Map<String, Value>) {
        if self.shapes.contains_key(&source_name) {
            return (source_name, arguments);
        }
        let pick = |keys: &[&str]| keys.iter().find_map(|key| arguments.get(*key).cloned());
        let mapped = match source_name.as_str() {
            "get_details_by_phone" if self.shapes.contains_key("get_customer_by_phone") => {
                pick(&["phone_number", "phone"])
                    .map(|value| ("get_customer_by_phone", vec![("phone_number", value)]))
            }
            "get_line_details" | "get_line_status" | "get_roaming_status"
                if self.shapes.contains_key("get_details_by_id") =>
            {
                pick(&["line_id", "id"]).map(|value| ("get_details_by_id", vec![("id", value)]))
            }
            "get_plan_details" if self.shapes.contains_key("get_details_by_id") => {
                pick(&["plan_id", "id"]).map(|value| ("get_details_by_id", vec![("id", value)]))
            }
            "enable_roaming" | "disable_roaming" if self.shapes.contains_key("toggle_roaming") => {
                pick(&["line_id", "id"]).map(|value| {
                    (
                        "toggle_roaming",
                        vec![
                            ("line_id", value),
                            ("enabled", Value::Bool(source_name == "enable_roaming")),
                        ],
                    )
                })
            }
            _ => None,
        };
        let Some((name, values)) = mapped else {
            return (source_name, arguments);
        };
        (
            name.to_string(),
            values.into_iter().map(|(key, value)| (key.to_string(), value)).collect(),
        )
    }

    fn reset_state(&mut self) -> String {
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for MiniCPM5ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        self.normalize_buffer();
        loop {
            let Some(start) = self.buffer.find(FUNCTION_START) else {
                let keep = partial_marker_suffix_len(&self.buffer, FUNCTION_START);
                let emit_len = self.buffer.len() - keep;
                if emit_len > 0 {
                    output.push_text(&self.buffer[..emit_len]);
                    self.buffer.drain(..emit_len);
                }
                break;
            };
            if start > 0 {
                output.push_text(&self.buffer[..start]);
                self.buffer.drain(..start);
                continue;
            }
            let Some(relative_end) = self.buffer.find(FUNCTION_END) else {
                break;
            };
            let end = relative_end + FUNCTION_END.len();
            let block = self.buffer[..end].to_string();
            self.emit_block(&block, output)?;
            self.buffer.drain(..end);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        output.push_text(&self.buffer);
        let _ = self.reset_state();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

fn partial_marker_suffix_len(text: &str, marker: &str) -> usize {
    marker
        .char_indices()
        .map(|(index, _)| &marker[..index])
        .filter(|prefix| !prefix.is_empty() && text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

fn attribute_value<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let mut cursor = 0;
    while let Some(relative) = tag[cursor..].find(name) {
        let start = cursor + relative;
        let before = tag[..start].chars().next_back();
        if before.is_some_and(|ch| ch.is_alphanumeric() || ch == '_') {
            cursor = start + name.len();
            continue;
        }
        let after = tag[start + name.len()..].trim_start();
        let after = after.strip_prefix('=')?.trim_start();
        let quote = after.chars().next()?;
        if !matches!(quote, '\'' | '"') {
            return None;
        }
        let value = &after[quote.len_utf8()..];
        return value.find(quote).map(|end| &value[..end]);
    }
    None
}

fn param_text(text: &str) -> String {
    text.strip_prefix("<![CDATA[")
        .and_then(|value| value.strip_suffix("]]>"))
        .map(str::to_string)
        .unwrap_or_else(|| text.trim().to_string())
}

fn parse_jsonish_object(text: &str) -> Option<Map<String, Value>> {
    serde_json::from_str::<Value>(text)
        .ok()
        .or_else(|| serde_json::from_str::<Value>(&pythonish_to_json(text)?).ok())?
        .as_object()
        .cloned()
}

fn pythonish_to_json(text: &str) -> Option<String> {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;
    while let Some(ch) = chars.next() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                output.push('"');
            }
            '"' if !in_single => {
                in_double = !in_double;
                output.push(ch);
            }
            '\\' if in_single => {
                let next = chars.next()?;
                if next == '\'' {
                    output.push('\'');
                } else {
                    output.push('\\');
                    output.push(next);
                }
            }
            _ => output.push(ch),
        }
    }
    (!in_single && !in_double).then_some(output)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::MiniCPM5ToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars};
    use crate::tool::{Tool, ToolParser, ToolParserTestExt as _};

    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".into(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
                strict: None,
            },
            Tool {
                name: "sum_values".into(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "nums": {"type": "array"}, "exact": {"type": "boolean"}
                    },
                    "required": ["nums"]
                }),
                strict: None,
            },
        ]
    }

    #[test]
    fn parses_schema_typed_call() {
        let mut parser = MiniCPM5ToolParser::new(&tools());
        let output = parser
            .parse_complete(concat!(
                "before <function name=\"sum_values\">",
                "<param name=\"nums\">[1,2,3]</param>",
                "<param name=\"exact\">true</param></function> after"
            ))
            .unwrap();
        assert_eq!(output.normal_text(), "before  after");
        assert_eq!(output.calls()[0].name.as_deref(), Some("sum_values"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"exact": true, "nums": [1, 2, 3]})
        );
    }

    #[test]
    fn streams_every_character_boundary_and_multiple_calls() {
        let input = concat!(
            "head <functionname=\"get_weather\"><paramname=\"city\">上海</param></function>",
            " middle <function name=\"sum_values\"><param name=\"nums\">[7,8]</param>",
            "<param name=\"exact\">false</param></function> tail"
        );
        let mut parser = MiniCPM5ToolParser::new(&tools());
        let output = collect_stream(&mut parser, &split_by_chars(input, 1));
        assert_eq!(output.normal_text(), "head  middle  tail");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].arguments, r#"{"city":"上海"}"#);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({"exact": false, "nums": [7, 8]})
        );
    }

    #[test]
    fn preserves_invalid_and_incomplete_blocks() {
        let input = concat!(
            "<function name=\"missing\"><param name=\"x\">1</param></function>",
            "<function name=\"get_weather\"><param name=\"city\">Paris</param>"
        );
        let mut parser = MiniCPM5ToolParser::new(&tools());
        let output = parser.parse_complete(input).unwrap();
        assert_eq!(output.normal_text(), input);
        assert!(output.calls().is_empty());
    }

    #[test]
    fn unwraps_python_style_arguments() {
        let mut parser = MiniCPM5ToolParser::new(&tools());
        let output = parser
            .parse_complete(concat!(
                "<function name=\"get_weather\"><param name=\"arguments\">",
                "{'city': 'Paris'}",
                "</param></function>"
            ))
            .unwrap();
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn maps_alias_and_preserves_special_tokens() {
        let tool = Tool {
            name: "toggle_roaming".into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"}, "enabled": {"type": "boolean"}
                },
                "required": ["line_id", "enabled"]
            }),
            strict: None,
        };
        let mut parser = MiniCPM5ToolParser::new(&[tool]);
        let output = parser
            .parse_complete(concat!(
                "<function name=\"enable_roaming\">",
                "<param name=\"line_id\">L1</param></function>"
            ))
            .unwrap();
        assert!(parser.preserve_special_tokens());
        assert_eq!(output.calls()[0].name.as_deref(), Some("toggle_roaming"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"enabled": true, "line_id": "L1"})
        );
    }
}
