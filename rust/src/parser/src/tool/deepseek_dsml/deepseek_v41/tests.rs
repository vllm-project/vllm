// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use expect_test::expect;
use serde_json::json;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
};

use super::DeepSeekV41ToolParser;
use crate::tool::test_utils::collect_stream;
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
    }]
}

#[test]
fn spaced_dsml_streaming_preserves_shared_schema_coercion() {
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
        "</｜DSML｜ invoke>\n",
        "<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">second</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    );
    let expected = [
        json!({"query":"  value\n","limit":2,"enabled":true,"data":{"x":1},"items":[1,2],"label":"42","unknown":"42","nullish":null}),
        json!({"query":"second"}),
    ];
    for split in wire.char_indices().map(|(i, _)| i).chain(std::iter::once(wire.len())) {
        let mut parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
        let output = collect_stream(parser.as_mut(), &[&wire[..split], &wire[split..]]);
        // Protocol framing follows the current shared DSML behavior.
        assert_eq!(output.normal_text(), "summary", "split {split}");
        assert_eq!(output.calls().len(), 2, "split {split}");
        for (index, call) in output.calls().iter().enumerate() {
            assert_eq!(call.tool_index, index, "split {split}");
            assert_eq!(call.name.as_deref(), Some("lookup"), "split {split}");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
                expected[index],
                "split {split}"
            );
        }
    }
}

#[test]
fn v4_tags_remain_text_in_v41_dialect() {
    let mut parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
    let output = collect_stream(parser.as_mut(), &["before<｜DSML｜tool_calls>after"]);
    expect!["before<｜DSML｜tool_calls>after"].assert_eq(&output.normal_text());
    assert!(output.calls().is_empty());
}

#[test]
fn structural_tag_ignores_parameter_schemas_and_strict() {
    let parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
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
                assert_eq!(build(function), expected);
            }
        }
    }
}
