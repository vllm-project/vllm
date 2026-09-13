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

fn build_tag(
    function: FunctionDefinition,
    choice: ToolChoice,
) -> xgrammar_structural_tag::format::StructuralTag {
    let parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
    let builder = parser.structural_tag_builder().unwrap();
    build_structural_tag(
        builder,
        &[ToolParam::Function(FunctionToolParam::new(function))],
        choice,
        StructuralTagOptions::default().with_reasoning(false),
    )
    .unwrap()
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
fn structural_tag_constrains_parameters_by_schema() {
    let function = FunctionDefinition::new("lookup").with_parameters(json!({
        "$defs": {
            "window": {
                "type": "object",
                "properties": {"start": {"type": "integer"}},
                "required": ["start"]
            }
        },
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
            "unit": {"type": "string", "enum": ["ids", "titles"]},
            "window": {"$ref": "#/$defs/window"},
            "note": true
        },
        "required": ["query", "limit"],
        "additionalProperties": false
    }));

    let tag = build_tag(function, ToolChoice::required());

    expect![[r##"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"\n\n<｜DSML｜ calls>\n"},{"type":"tags_with_separator","tags":[{"begin":"<｜DSML｜ invoke name=\"lookup\">\n","content":{"type":"sequence","elements":[{"type":"tag","begin":"<｜DSML｜ parameter name=\"query","content":{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\""},{"type":"or","elements":[{"type":"const_string","value":"true"},{"type":"const_string","value":"false"}]},{"type":"const_string","value":"\">"},{"type":"any_text","excludes":["</｜DSML｜ parameter>","</｜DSML｜ invoke>","</｜DSML｜ calls>"]}]},"end":"</｜DSML｜ parameter>\n"},{"type":"tag","begin":"<｜DSML｜ parameter name=\"limit","content":{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\""},{"type":"or","elements":[{"type":"const_string","value":"true"},{"type":"const_string","value":"false"}]},{"type":"const_string","value":"\">"},{"type":"json_schema","json_schema":{"type":"integer","$defs":{"window":{"type":"object","properties":{"start":{"type":"integer"}},"required":["start"]}}},"style":"json","any_order":false,"max_whitespace_cnt":null}]},"end":"</｜DSML｜ parameter>\n"},{"type":"optional","content":{"type":"tag","begin":"<｜DSML｜ parameter name=\"unit","content":{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\""},{"type":"or","elements":[{"type":"const_string","value":"true"},{"type":"const_string","value":"false"}]},{"type":"const_string","value":"\">"},{"type":"or","elements":[{"type":"const_string","value":"ids"},{"type":"const_string","value":"titles"}]}]},"end":"</｜DSML｜ parameter>\n"}},{"type":"optional","content":{"type":"tag","begin":"<｜DSML｜ parameter name=\"window","content":{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\""},{"type":"or","elements":[{"type":"const_string","value":"true"},{"type":"const_string","value":"false"}]},{"type":"const_string","value":"\">"},{"type":"json_schema","json_schema":{"$ref":"#/$defs/window","$defs":{"window":{"type":"object","properties":{"start":{"type":"integer"}},"required":["start"]}}},"style":"json","any_order":false,"max_whitespace_cnt":null}]},"end":"</｜DSML｜ parameter>\n"}},{"type":"optional","content":{"type":"tag","begin":"<｜DSML｜ parameter name=\"note","content":{"type":"or","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\"true\">"},{"type":"any_text","excludes":["</｜DSML｜ parameter>","</｜DSML｜ invoke>","</｜DSML｜ calls>"]}]},{"type":"sequence","elements":[{"type":"const_string","value":"\" string=\"false\">"},{"type":"json_schema","json_schema":true,"style":"json","any_order":false,"max_whitespace_cnt":null}]}]},"end":"</｜DSML｜ parameter>\n"}}]},"end":"</｜DSML｜ invoke>\n"}],"separator":"","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"</｜DSML｜ calls>"}]}}"##]].assert_eq(&tag.to_json_string().unwrap());
}

#[test]
fn structural_tag_strict_false_matches_unconstrained_parameters() {
    let parameters = json!({
        "type": "object",
        "properties": {"query": {"const": "fixed"}},
        "required": ["query"],
        "additionalProperties": false
    });
    for choice in [
        ToolChoice::auto(),
        ToolChoice::required(),
        ToolChoice::function("lookup"),
    ] {
        let strict_off = build_tag(
            FunctionDefinition::new("lookup")
                .with_parameters(parameters.clone())
                .with_strict(false),
            choice.clone(),
        );
        let unconstrained = build_tag(
            FunctionDefinition::new("lookup").with_parameters(json!(true)),
            choice,
        );
        assert_eq!(strict_off, unconstrained);
    }
}

#[test]
fn structural_tag_without_parameters_stays_permissive() {
    let tag = build_tag(FunctionDefinition::new("lookup"), ToolChoice::required());
    let value = serde_json::to_value(tag).unwrap();
    let invoke_content = &value["format"]["elements"][1]["tags"][0]["content"];
    assert_eq!(invoke_content["type"], "star");
}

#[test]
fn structural_tag_auto_without_tools_allows_any_text() {
    let parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
    let builder = parser.structural_tag_builder().unwrap();

    let tag = build_structural_tag(
        builder,
        &[],
        ToolChoice::auto(),
        StructuralTagOptions::default().with_reasoning(false),
    )
    .unwrap();
    let value = serde_json::to_value(tag).unwrap();

    assert_eq!(value["format"]["type"], "any_text");
    assert_eq!(value["format"]["excludes"], json!(["<think>", "</think>"]));
}

#[test]
fn structural_tag_named_choice_keeps_only_named_tool() {
    let parser = DeepSeekV41ToolParser::create(&tools()).unwrap();
    let builder = parser.structural_tag_builder().unwrap();
    let tools = vec![
        ToolParam::Function(FunctionToolParam::new(
            FunctionDefinition::new("search").with_parameters(json!({
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"]
            })),
        )),
        ToolParam::Function(FunctionToolParam::new(
            FunctionDefinition::new("lookup").with_parameters(json!({
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"]
            })),
        )),
    ];

    let tag = build_structural_tag(
        builder,
        &tools,
        ToolChoice::function("lookup"),
        StructuralTagOptions::default().with_reasoning(false),
    )
    .unwrap()
    .to_json_string()
    .unwrap();

    assert!(tag.contains("lookup"));
    assert!(!tag.contains("search"));
    // The named tool's parameter schema is still enforced.
    assert!(tag.contains(r#"<｜DSML｜ parameter name=\"id"#), "{tag}");
}
