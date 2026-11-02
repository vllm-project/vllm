// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use expect_test::expect;
use serde_json::json;
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::{
    FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
};

use super::super::test_utils::{collect_stream, decode, dedicated, ordinary, tokenizer};
use super::DeepSeekV41ToolParser;
use crate::tool::{Tool, ToolParser, ToolParserError, ToolParserOutput};

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
        let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
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

/// Feed the structural encoding of the concatenated `chunks`, split at the
/// chunk boundaries, then return the parsed output and the `finish()` result.
fn parse_then_finish(
    parser: &mut dyn ToolParser,
    chunks: &[&str],
) -> (ToolParserOutput, crate::tool::Result<ToolParserOutput>) {
    let mut rest = decode(&dedicated(&chunks.concat()));
    let mut output = ToolParserOutput::default();
    for chunk in chunks {
        parser.parse_into(rest.drain_prefix(chunk.len()), &mut output).unwrap();
    }
    let finished = parser.finish();
    (output.coalesce(), finished)
}

#[test]
fn spaced_dsml_reports_tool_call_cut_by_max_tokens() {
    let wires = [
        concat!(
            "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n",
            "<｜DSML｜ parameter name=\"query\" string=\"true\">Paris is the ca",
        ),
        "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invo",
        "Checking.\n\n<｜DSML｜ ca",
        "Checking.\n\n<｜DSML｜",
    ];
    for wire in wires {
        for split in wire.char_indices().map(|(i, _)| i).chain(std::iter::once(wire.len())) {
            let mut parser =
                DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
            let (output, finished) =
                parse_then_finish(parser.as_mut(), &[&wire[..split], &wire[split..]]);
            assert_eq!(output.normal_text(), "Checking.", "{wire:?} split {split}");
            assert!(output.calls().is_empty(), "{wire:?} split {split}");
            assert!(
                matches!(finished, Err(ToolParserError::IncompleteToolCall { .. })),
                "{wire:?} split {split}"
            );
            let unfinished = parser.reset();
            assert!(
                unfinished.contains("<｜"),
                "{wire:?} split {split}: {unfinished:?}"
            );
            assert!(
                !unfinished.contains("Checking"),
                "{wire:?} split {split}: {unfinished:?}"
            );
        }
    }
}

#[test]
fn spaced_dsml_keeps_completed_call_before_truncated_one() {
    let wire = concat!(
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">Paris</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"lookup\">\n",
        "<｜DSML｜ parameter name=\"query\" string=\"true\">Tok",
    );
    for split in wire.char_indices().map(|(i, _)| i).chain(std::iter::once(wire.len())) {
        let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
        let (output, finished) =
            parse_then_finish(parser.as_mut(), &[&wire[..split], &wire[split..]]);
        let calls = output.calls();
        assert_eq!(calls.len(), 1, "split {split}");
        assert_eq!(calls[0].name.as_deref(), Some("lookup"), "split {split}");
        assert_eq!(calls[0].arguments, r#"{"query":"Paris"}"#, "split {split}");
        assert_eq!(output.normal_text(), "", "split {split}");
        assert!(
            matches!(finished, Err(ToolParserError::IncompleteToolCall { .. })),
            "split {split}"
        );
        let unfinished = parser.reset();
        assert!(
            unfinished.contains("Tok") && !unfinished.contains("Paris"),
            "split {split}: {unfinished:?}"
        );
    }
}

#[test]
fn spaced_dsml_finish_keeps_ordinary_trailing_text() {
    // A partial `｜DSML｜` spelling cannot come from the dedicated token, which
    // is emitted whole, so it is text.
    for wire in [
        "Done.\n\n",
        "a < b",
        "Use <｜User｜> tags",
        "Checking.\n\n<｜DS",
    ] {
        let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
        let output = collect_stream(parser.as_mut(), &[wire]);
        assert_eq!(output.normal_text(), wire);
        assert!(output.calls().is_empty());
    }
}

#[test]
fn spaced_dsml_finish_keeps_ordinary_token_opener_as_text() {
    // An opener spelled with ordinary tokens is text, even when it is cut.
    for wire in [
        "Checking.\n\n<｜",
        "Checking.\n\n<｜DSML｜ ca",
        "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invo",
    ] {
        let mut parser = DeepSeekV41ToolParser::create(&tools(), Arc::new(tokenizer())).unwrap();
        let mut output = ToolParserOutput::default();
        parser.parse_into(decode(&ordinary(wire)), &mut output).unwrap();
        output.append(parser.finish().unwrap());
        let output = output.coalesce();
        assert_eq!(output.normal_text(), wire);
        assert!(output.calls().is_empty());
    }
}
