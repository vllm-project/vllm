// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use expect_test::expect;
use serde_json::json;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
};

use super::super::test_utils::{collect_stream, tokenizer};
use super::DeepSeekV41ToolParser;
use crate::tool::test_utils::split_by_chars;
use crate::tool::{Tool, ToolParser};

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "lookup".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "data": {"type": "object"},
                "items": {"type": "array"},
                "label": {"type": "string"}
            }
        }),
        strict: None,
        defer_loading: None,
    }]
}

#[test]
fn spaced_dsml_streaming_passes_json_values_through() {
    // `string="true"` values are raw text and every other value is JSON text
    // as the encoder wrote it, regardless of the schema (`label` is a string
    // parameter), or a string when it is not valid JSON. A repeated parameter
    // keeps its first value.
    let wire = concat!(
        "summary\n\n<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">  value\n</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"enabled\" string=\"false\">true</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"data\" string=\"false\">{\"x\":1}</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"items\" string=\"false\">[1,2]</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"label\" string=\"false\">42</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"unknown\" string=\"false\">42</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"nullish\" string=\"false\">null</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"city\" string=\"false\">Hangzhou</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">again</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">second</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    );
    let expected = [
        json!({"query":"  value\n","limit":2,"enabled":true,"data":{"x":1},"items":[1,2],"label":42,"unknown":42,"nullish":null,"city":"Hangzhou"}),
        json!({"query":"second"}),
    ];
    let two_chunks = wire
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(wire.len()))
        .map(|split| vec![&wire[..split], &wire[split..]]);
    for chunks in two_chunks.chain([split_by_chars(wire, 1)]) {
        let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
        let output = collect_stream(parser.as_mut(), &chunks);
        let chunks = chunks.len();
        // Protocol framing follows the current shared DSML behavior.
        assert_eq!(output.normal_text(), "summary", "{chunks} chunks");
        assert_eq!(output.calls().len(), 2, "{chunks} chunks");
        for (index, call) in output.calls().iter().enumerate() {
            assert_eq!(call.tool_index, index, "{chunks} chunks");
            assert_eq!(call.name.as_deref(), Some("lookup"), "{chunks} chunks");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
                expected[index],
                "{chunks} chunks"
            );
        }
    }
}

#[test]
fn spaced_dsml_keeps_parameter_without_string_attr() {
    let wire = concat!(
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\">value</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"limit\">2</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    );
    let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();

    let output = collect_stream(parser.as_mut(), &[wire]);

    assert!(output.normal_text().is_empty());
    assert_eq!(output.calls().len(), 1);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
        json!({ "query": "value", "limit": 2 })
    );
}

#[test]
fn v4_tags_remain_text_in_v41_dialect() {
    let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
    let output = collect_stream(parser.as_mut(), &["before<｜DSML｜tool_calls>after"]);
    expect!["before<｜DSML｜tool_calls>after"].assert_eq(&output.normal_text());
    assert!(output.calls().is_empty());
}

#[test]
fn structural_tag_respects_parameter_schemas_and_strict() {
    let parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
    let builder = parser.structural_tag_builder().unwrap();
    let options = StructuralTagOptions::default().with_reasoning(false);
    for choice in [
        ToolChoice::auto(),
        ToolChoice::required(),
        ToolChoice::function("lookup"),
    ] {
        let build = |function| {
            build_structural_tag(
                builder,
                &[ToolParam::Function(FunctionToolParam::new(function))],
                choice.clone(),
                options,
            )
            .unwrap()
        };
        let expected = build(FunctionDefinition::new("lookup"));
        for strict in [None, Some(false), Some(true)] {
            for parameters in [
                json!(false),
                json!({"type":"object", "properties":{"query":{"const":"fixed"}}, "required":["query"]}),
            ] {
                let mut function = FunctionDefinition::new("lookup").with_parameters(parameters);
                function.strict = strict;
                let tag = build(function);
                if strict == Some(false) {
                    assert_eq!(tag, expected);
                } else {
                    assert_ne!(tag, expected);
                }
                assert!(tag.to_json_string().unwrap().contains("deepseek_v4_1_xml"));
            }
        }
    }
}
