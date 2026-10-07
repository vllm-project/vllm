// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

const QWEN_XML_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Qwen XML",
    tool_call_start: "<tool_call>",
    first_tool_call_start: "\n\n<tool_call>",
    next_tool_call_start: "\n<tool_call>",
    tool_call_end: "</tool_call>",
    trim_parameter_newlines: true,
};

/// Tool parser for Qwen XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=location>Tokyo</parameter>
/// </function>
/// </tool_call>
/// ```
pub struct Qwen3XmlToolParser {
    inner: Qwen3CoderToolParser,
}

impl Qwen3XmlToolParser {
    /// Create a Qwen XML tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            inner: Qwen3CoderToolParser::with_config(tools, QWEN_XML_CONFIG),
        }
    }
}

impl ToolParser for Qwen3XmlToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Qwen3Coder.builder())
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::super::super::qwen_coder::Qwen3CoderToolParser;
    use super::Qwen3XmlToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<parameter={name}>{value}</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>\n<function={function_name}>\n{params}\n</function>\n</tool_call>")
    }

    #[test]
    fn qwen_xml_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn qwen_xml_parse_complete_extracts_single_tool_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2026-04-29")],
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2026-04-29",
            })
        );
    }

    #[test]
    fn qwen_xml_parse_complete_preserves_prefix_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Thinking... {}",
                build_tool_call("get_weather", &[("location", "NYC")])
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn qwen_xml_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "{}\n{}",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let chunks = split_by_chars(&text, 7);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn qwen_xml_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        parser
            .parse_chunk("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert_eq!(
            error.to_report_string(),
            "tool parser parsing failed: incomplete Qwen XML tool call"
        );
    }

    #[test]
    fn qwen_xml_matches_qwen_coder_parser() {
        let inputs = [
            "Hello, world!".to_string(),
            build_tool_call("get_weather", &[("location", "SF"), ("date", "2026-04-29")]),
            format!(
                "Thinking... {}",
                build_tool_call("get_weather", &[("location", "NYC")])
            ),
            format!(
                "{}\n{}",
                build_tool_call("get_weather", &[("location", "SF")]),
                build_tool_call("add", &[("x", "1"), ("y", "2")]),
            ),
        ];
        for input in &inputs {
            let chunks = split_by_chars(input, 7);
            let mut xml = Qwen3XmlToolParser::new(&test_tools());
            let mut coder = Qwen3CoderToolParser::create(&test_tools()).unwrap();
            let mut xml_output = ToolParserOutput::default();
            let mut coder_output = ToolParserOutput::default();
            for chunk in &chunks {
                xml.parse_into(chunk, &mut xml_output).unwrap();
                coder.parse_into(chunk, &mut coder_output).unwrap();
            }
            xml_output.append(xml.finish().unwrap());
            coder_output.append(coder.finish().unwrap());
            let xml_output = xml_output.coalesce();
            let coder_output = coder_output.coalesce();
            assert_eq!(xml_output.normal_text(), coder_output.normal_text());
            assert_eq!(
                xml_output.calls().len(),
                coder_output.calls().len(),
                "input: {input}"
            );
            for (xml_call, coder_call) in xml_output.calls().iter().zip(coder_output.calls().iter())
            {
                assert_eq!(xml_call.name, coder_call.name, "input: {input}");
                assert_eq!(
                    serde_json::from_str::<Value>(&xml_call.arguments).unwrap(),
                    serde_json::from_str::<Value>(&coder_call.arguments).unwrap(),
                    "input: {input}"
                );
            }
        }
    }

    #[test]
    fn tool_framing_preserves_body_whitespace_across_chunk_boundaries() {
        assert_tool_framing_preserves_body_whitespace::<Qwen3XmlToolParser>(
            "\n\n",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        );
    }
}
