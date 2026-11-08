// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};

use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V4 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜tool_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜tool_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// V4 reuses the V3.2 DSML invoke/parameter grammar but wraps calls in
/// `<｜DSML｜tool_calls>` instead of `<｜DSML｜function_calls>`.
///
/// DeepSeek V4 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
///
/// A marker counts only where its `｜DSML｜` is the dedicated token, so DSML
/// spelled with ordinary tokens (the model quoting the format) stays text,
/// as in DeepSeek's reference output parser.
pub struct DeepSeekV4ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV4ToolParser {
    /// Create a DeepSeek V4 tool parser.
    fn new(tools: &[Tool], tokenizer: &dyn Tokenizer) -> Self {
        Self(DeepSeekDsmlToolParser::new(
            tools,
            tokenizer,
            DsmlTokens::V4,
        ))
    }
}

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools, tokenizer.as_ref())))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::DeepSeekV4.builder())
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.0.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.0.finish()
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use std::sync::Arc;

    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::super::test_utils::{
        DsmlParserTestExt as _, collect_stream, decode, dedicated, ordinary, tokenizer,
    };
    use super::DeepSeekV4ToolParser;
    use crate::tool::test_utils::{decode_ids, test_tools};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace_with;
    use crate::tool::{ToolParser, ToolParserOutput};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!(
                    r#"<｜DSML｜parameter name="{name}" string="true">{value}</｜DSML｜parameter>"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"{function_name}\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    }

    #[test]
    fn deepseek_v4_exposes_structural_tag_builder() {
        let parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());

        assert!(parser.structural_tag_builder().is_some());
    }

    #[test]
    fn deepseek_v4_parse_complete_reuses_dsml_parser_with_tool_calls_token() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2024-01-16")],
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2024-01-16"
            })
        );
    }

    #[test]
    fn deepseek_v4_streaming_handles_tool_calls_token_split_across_chunks() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
        let output = collect_stream(
            &mut parser,
            &[
                "Thinking... ",
                "<｜DSML｜",
                "tool",
                "_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜",
                "tool_calls>",
            ],
        );

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "Beijing" })
        );
    }

    #[test]
    fn deepseek_v4_streaming_keeps_parameter_without_string_attr() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
        let output = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"loc",
                "ation\">Par",
                "is</｜DSML｜parameter>\n",
                "<｜DSML｜parameter name=\"date\" string=\"true\">tomorrow</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "Paris", "date": "tomorrow" })
        );
    }

    #[test]
    fn tool_framing_preserves_body_whitespace_across_chunk_boundaries() {
        assert_tool_framing_preserves_body_whitespace_with::<DeepSeekV4ToolParser>(
            tokenizer(),
            "\n\n",
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\"></｜DSML｜invoke></｜DSML｜tool_calls>",
        );
    }

    /// Feed `ids` decoded once and split at the byte offsets `splits`, then
    /// finish and coalesce.
    fn stream_ids(ids: &[u32], splits: &[usize]) -> ToolParserOutput {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
        let mut rest = decode(ids);
        let mut output = ToolParserOutput::default();
        let mut consumed = 0;
        for &split in splits {
            parser.parse_into(rest.drain_prefix(split - consumed), &mut output).unwrap();
            consumed = split;
        }
        parser.parse_into(rest, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        output.coalesce()
    }

    fn location(output: &ToolParserOutput) -> Value {
        serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap()["location"].clone()
    }

    #[test]
    fn deepseek_v4_ordinary_tool_calls_marker_stays_text() {
        let call = build_tool_call("get_weather", &[("location", "SF")]);
        let ids = [dedicated("Write "), ordinary(&call), dedicated(" here.")].concat();

        let output = stream_ids(&ids, &[]);
        assert_eq!(output.normal_text(), format!("Write {call} here."));
        assert!(output.calls().is_empty());
    }

    #[test]
    fn deepseek_v4_ordinary_partial_tool_calls_marker_streams_immediately() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
        let mut output = ToolParserOutput::default();
        parser.parse_into(decode(&ordinary("text <｜DS")), &mut output).unwrap();

        // Not held back: an ordinary `｜` cannot grow into the DSML token.
        assert_eq!(output.normal_text(), "text <｜DS");
    }

    #[test]
    fn deepseek_v4_dedicated_markers_split_mid_marker_across_chunks() {
        let wire = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "SF")])
        );
        // Every char boundary, including right after the `<` before the DSML
        // token and inside its spelling, in every marker of the call.
        for split in wire.char_indices().map(|(i, _)| i) {
            let mut parser = DeepSeekV4ToolParser::new(&test_tools(), &tokenizer());
            let output = collect_stream(&mut parser, &[&wire[..split], &wire[split..]]);

            assert_eq!(output.normal_text(), "Thinking... ", "split {split}");
            assert_eq!(output.calls().len(), 1, "split {split}");
            assert_eq!(location(&output), json!("SF"), "split {split}");
        }
    }

    #[test]
    fn deepseek_v4_ordinary_block_closers_inside_parameter_value_stay_value() {
        let opener = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">";
        let quoted = "</｜DSML｜invoke></｜DSML｜tool_calls>";
        let closer = "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let ids = [
            dedicated(opener),
            dedicated("a"),
            ordinary(quoted),
            dedicated("b"),
            dedicated(closer),
        ]
        .concat();

        let text = decode(&ids).text;
        let splits = [0, text.len() / 3, text.len() / 2, text.len() - 1];
        for split in splits.map(|at| text.floor_char_boundary(at)) {
            let output = stream_ids(&ids, &[split]);
            assert!(output.normal_text().is_empty(), "split {split}");
            assert_eq!(output.calls().len(), 1, "split {split}");
            assert_eq!(
                location(&output),
                json!(format!("a{quoted}b")),
                "split {split}"
            );
        }
    }

    #[test]
    fn deepseek_v4_without_dsml_token_matches_markers_by_spelling() {
        let tokenizer = TestTokenizer::new();
        let mut parser =
            DeepSeekV4ToolParser::create(&test_tools(), Arc::new(tokenizer.clone())).unwrap();
        let call = build_tool_call("get_weather", &[("location", "SF")]);
        let ids = tokenizer.encode_ordinary(&call).unwrap();

        let mut output = ToolParserOutput::default();
        parser.parse_into(decode_ids(&tokenizer, &ids), &mut output).unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(location(&output.coalesce()), json!("SF"));
    }
}
