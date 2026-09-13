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

#[test]
fn spaced_dsml_streaming_emits_arguments_before_invoke_closes() {
    let mut parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
    let mut output = crate::tool::ToolParserOutput::default();
    parser
        .parse_into(
            concat!(
                "<｜DSML｜ calls>\n",
                "<｜DSML｜ invoke name=\"lookup\">\n",
                "<｜DSML｜ parameter name=\"query\" string=\"true\">hello",
                "</｜DSML｜ parameter>\n",
            ),
            &mut output,
        )
        .unwrap();

    let calls = output.calls();
    assert!(calls.iter().any(|call| call.name.as_deref() == Some("lookup")));
    let arguments = calls.iter().map(|call| call.arguments.as_str()).collect::<String>();
    assert_eq!(arguments, r#"{"query":"hello""#);
    assert!(parser.finish().is_err());
}

#[test]
fn spaced_dsml_streaming_matches_pr42879_mixed_argument_semantics() {
    let tool = Tool {
        name: "plan_trip".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "days": {"type": "integer"},
                "flexible": {"type": "boolean"},
                "cities": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"}
            }
        }),
        strict: None,
    };
    let wire = concat!(
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"plan_trip\">\n",
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"flexible\" string=\"false\">false</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"cities\" string=\"false\">[\"Beijing\",\"Shanghai\",\"Tokyo\",\"New York\"]</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"notes\" string=\"true\">靠窗座位</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    );
    let chunks = crate::tool::test_utils::split_by_chars(wire, 4);
    let mut parser = DeepSeekV41ToolParser::create(&[tool]).unwrap();
    let mut output = crate::tool::ToolParserOutput::default();
    let mut non_empty_argument_deltas = 0;
    for chunk in chunks {
        let before = output.events.len();
        parser.parse_into(chunk, &mut output).unwrap();
        non_empty_argument_deltas += output.events[before..]
            .iter()
            .filter(|event| match event {
                crate::tool::ToolParserEvent::ToolCall(call) => !call.arguments.is_empty(),
                crate::tool::ToolParserEvent::Text(_) => false,
            })
            .count();
    }
    output.append(parser.finish().unwrap());
    let output = output.coalesce();

    assert!(non_empty_argument_deltas > 2);
    assert_eq!(output.calls().len(), 1);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
        json!({
            "days": 3,
            "flexible": false,
            "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
            "notes": "靠窗座位"
        })
    );
}
