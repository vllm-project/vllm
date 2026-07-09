// This module is shared by multiple benchmark targets.
// There could be false positives for unused code or imports, and fixing them would lead to some other benchmarks failing to compile.
#![allow(dead_code)]
#![allow(unused_imports)]

mod adapter;

pub(super) use adapter::UnifiedToolParserAdapter;
use futures::FutureExt as _;
use openai_protocol::common::{Function as OpenAiFunction, Tool as OpenAiTool};
use tool_parser::traits::ToolParser as ExternalToolParser;
use vllm_parser::tool::test_utils::collect_stream;
use vllm_parser::tool::{Tool, ToolParser};

pub(super) fn openai_tools(tools: &[Tool]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|tool| OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            },
        })
        .collect()
}

pub(super) fn feed_parser(parser: &mut dyn ToolParser, chunks: &[&str]) -> (String, usize) {
    let result = collect_stream(parser, chunks);
    (result.normal_text(), result.calls().len())
}

pub(super) fn feed_external_parser(
    parser: &mut impl ExternalToolParser,
    tools: &[OpenAiTool],
    chunks: &[&str],
) -> (String, usize) {
    ExternalToolParser::reset(parser);

    let mut normal_text = String::new();
    let mut calls_len = 0;
    for chunk in chunks {
        let delta = parser
            .parse_incremental(chunk, tools)
            .now_or_never()
            .expect("external parser should not suspend")
            .expect("chunk should parse");
        normal_text.push_str(&delta.normal_text);
        calls_len += delta.calls.len();
    }
    calls_len += parser.get_unstreamed_tool_args().unwrap_or_default().len();
    (normal_text, calls_len)
}
