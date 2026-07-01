use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, seq};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::{ParamElement, ParamInput, ToolSchemas};
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::Tool;

const NAMESPACE: &str = "]<]minimax[>[";
const TOOL_CALL_START: &str = "]<]minimax[>[<tool_call>";
const TOOL_CALL_END: &str = "]<]minimax[>[</tool_call>";
const INVOKE_START: &str = "]<]minimax[>[<invoke";
const INVOKE_END: &str = "]<]minimax[>[</invoke>";
const ELEMENT_START: &str = "]<]minimax[>[<";
const ELEMENT_END_START: &str = "]<]minimax[>[</";
const MIXED_TEXT_FIELD: &str = "$text";

type MinimaxM3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Mode {
    Text,
    ToolBlock { invoke_end_scan: MarkerScanState },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    Invoke {
        name: String,
        params: Vec<(String, ParamInput)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for MiniMax M3 namespace-delimited XML-style tool calls.
///
/// Example tool call content with recursive parameters:
///
/// ```text
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="create_order">
/// ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
/// ]<]minimax[>[<shipping>
/// ]<]minimax[>[<city>Singapore]<]minimax[>[</city>
/// ]<]minimax[>[<zip>018956]<]minimax[>[</zip>
/// ]<]minimax[>[</shipping>
/// ]<]minimax[>[<items>
/// ]<]minimax[>[<item>
/// ]<]minimax[>[<sku>book-001]<]minimax[>[</sku>
/// ]<]minimax[>[<qty>2]<]minimax[>[</qty>
/// ]<]minimax[>[</item>
/// ]<]minimax[>[</items>
/// ]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
///
/// With a schema where `shipping` is an object and `items` is an array of
/// objects, recursive parameter conversion produces:
///
/// ```json
/// {
///   "user_id": 42,
///   "shipping": {
///     "city": "Singapore",
///     "zip": 18956
///   },
///   "items": [
///     {
///       "sku": "book-001",
///       "qty": 2
///     }
///   ]
/// }
/// ```
///
/// MiniMax M3 emits the namespace marker `]<]minimax[>[` before each structural
/// tag. Arguments are emitted only after a full `<invoke>` block is parsed.
pub struct MinimaxM3ToolParser {
    buffer: String,
    mode: MinimaxM3Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MinimaxM3ToolParser {
    /// Create a MiniMax M3 tool parser.
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM3Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniMax M3 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MinimaxM3Event::ToolBlockStart => {
                self.mode = MinimaxM3Mode::ToolBlock {
                    invoke_end_scan: MarkerScanState::default(),
                };
            }
            MinimaxM3Event::Invoke { name, params } => {
                let arguments = self.tool_parameters.convert_params_with_schema(&name, params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            MinimaxM3Event::ToolBlockEnd => self.mode = MinimaxM3Mode::Done,
            MinimaxM3Event::IgnoredRest => {}
        }
        Ok(())
    }
}

impl ToolParser for MinimaxM3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minimax_m3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            MinimaxM3Mode::Text => {
                output.push_text(&self.buffer);
            }
            MinimaxM3Mode::ToolBlock { .. } => {
                if !self.buffer.trim_start().is_empty() {
                    return Err(parsing_failed!("incomplete MiniMax M3 tool call"));
                }
            }
            MinimaxM3Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM3Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a MiniMax M3 event for the current parser mode.
fn parse_next_minimax_m3_event(
    input: &mut MinimaxM3Input<'_>,
    mode: &mut MinimaxM3Mode,
) -> ModalResult<MinimaxM3Event> {
    match mode {
        MinimaxM3Mode::Text => parse_text_event(input),
        MinimaxM3Mode::ToolBlock { invoke_end_scan } => {
            parse_tool_block_event(input, invoke_end_scan)
        }
        MinimaxM3Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M3 event.
fn parse_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M3 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    literal(TOOL_CALL_START).value(MinimaxM3Event::ToolBlockStart).parse_next(input)
}

/// Parse a safe text run before the next MiniMax M3 marker.
fn safe_text_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    safe_text_len(input, TOOL_CALL_START).map(|len| MinimaxM3Event::Text { len })
}

/// Parse one event inside a MiniMax M3 tool block.
fn parse_tool_block_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_end_event, |input: &mut MinimaxM3Input<'_>| {
        invoke_event(input, invoke_end_scan)
    }))
    .parse_next(input)
}

/// Parse a MiniMax M3 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM3Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete MiniMax M3 invoke block.
fn invoke_event(
    input: &mut MinimaxM3Input<'_>,
    invoke_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM3Event> {
    let (name, body) = seq!(
        _: ws0,
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
        take_until_marker(INVOKE_END, invoke_end_scan),
        _: literal(INVOKE_END),
    )
    .parse_next(input)?;
    let params = parse_invoke_params(body)?;

    Ok(MinimaxM3Event::Invoke {
        name: name.trim().to_string(),
        params,
    })
}

/// Parse all parameter elements inside a complete MiniMax M3 invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<(String, ParamInput)>> {
    let mut input = invoke_body;
    let mut elements = Vec::new();

    loop {
        let _ = ws0.parse_next(&mut input)?;
        if input.is_empty() {
            break;
        }
        if input.starts_with(ELEMENT_START) {
            elements.push(parameter_element(&mut input)?);
            continue;
        }
        if input.starts_with(NAMESPACE) {
            return malformed();
        }
        // Be tolerant: ordinary text at an invokeparameter boundary ends this invoke.
        // Keep parsed parameters and drop the remaining invoke body.
        break;
    }

    Ok(elements.into_iter().map(|element| (element.name, element.value)).collect())
}

/// Parse a MiniMax M3 parameter element.
fn parameter_element(input: &mut &str) -> ModalResult<ParamElement> {
    let name = open_element_tag(input)?.to_string();
    let value = element_body(input, &name)?;
    close_element_tag(input, &name)?;
    Ok(ParamElement { name, value })
}

/// Parse a MiniMax M3 opening element tag.
fn open_element_tag<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    let name = seq!(
        _: literal(ELEMENT_START),
        take_until(1.., ">"),
        _: literal(">"),
    )
    .parse_next(input)?;

    let name = name.0;
    if name.starts_with('/') || name.trim().is_empty() {
        return malformed();
    }

    Ok(name)
}

/// Parse a MiniMax M3 closing element tag.
fn close_element_tag(input: &mut &str, name: &str) -> ModalResult<()> {
    literal(ELEMENT_END_START).void().parse_next(input)?;
    literal(name).void().parse_next(input)?;
    literal(">").void().parse_next(input)
}

/// Parse the body of one MiniMax M3 element.
fn element_body(input: &mut &str, closing_name: &str) -> ModalResult<ParamInput> {
    let close_tag = format!("{ELEMENT_END_START}{closing_name}>");
    let mut text = String::new();
    let mut elements = Vec::new();

    loop {
        text.push_str(text_until_namespace(input)?);

        if input.starts_with(&close_tag) {
            // Close tag reached, end of element body.
            break;
        }
        if input.starts_with(ELEMENT_START) {
            // Child element start reached, parse child element recursively.
            elements.push(parameter_element(input)?);
            continue;
        }
        if input.starts_with(NAMESPACE) {
            // Unexpected namespace marker.
            return malformed();
        }
    }

    if elements.is_empty() {
        Ok(ParamInput::Text(text))
    } else {
        if !text.trim().is_empty() {
            push_mixed_text_element(&mut elements, text);
        }
        Ok(ParamInput::Elements(elements))
    }
}

/// Parse text until the next MiniMax M3 namespace marker.
fn text_until_namespace<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    take_until(0.., NAMESPACE).parse_next(input)
}

/// Preserve mixed text content under a reserved object field.
///
/// By default, the field name is `$text`, but if that collides with an existing
/// child element name, prepend `$` until there is no collision.
fn push_mixed_text_element(elements: &mut Vec<ParamElement>, text: String) {
    let mut name = MIXED_TEXT_FIELD.to_string();
    while elements.iter().any(|element| element.name == name) {
        name.insert(0, '$');
    }
    elements.push(ParamElement {
        name,
        value: ParamInput::Text(text),
    });
}

/// Parse a quoted or unquoted XML attribute value from partial streaming input.
fn partial_attr_value<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M3 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    rest.value(MinimaxM3Event::IgnoredRest).parse_next(input)
}

fn malformed<T>() -> ModalResult<T> {
    Err(ErrMode::Cut(ContextError::new()))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{
        ELEMENT_END_START, ELEMENT_START, INVOKE_END, INVOKE_START, MinimaxM3ToolParser,
        TOOL_CALL_END, TOOL_CALL_START, ToolParser,
    };
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{Tool, ToolParserEvent, ToolParserTestExt as _};

    fn element(name: &str, body: &str) -> String {
        format!("{ELEMENT_START}{name}>{body}{ELEMENT_END_START}{name}>")
    }

    fn invoke(function_name: &str, body: &str) -> String {
        format!("{INVOKE_START} name=\"{function_name}\">{body}{INVOKE_END}")
    }

    fn build_tool_block(invokes: &[(&str, String)]) -> String {
        let invokes = invokes
            .iter()
            .map(|(function_name, body)| invoke(function_name, body))
            .collect::<Vec<_>>()
            .join("\n");
        format!("{TOOL_CALL_START}\n{invokes}\n{TOOL_CALL_END}")
    }

    fn m3_test_tools() -> Vec<Tool> {
        let mut tools = test_tools();
        tools.push(Tool {
            name: "create_order".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "user_id": { "type": "integer" },
                    "urgent": { "type": "boolean" },
                    "note": { "type": "string" },
                    "shipping": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": "integer" }
                        }
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": { "type": "string" },
                                "qty": { "type": "integer" }
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": { "type": "integer" }
                    },
                    "duplicate_demo": {
                        "type": "object",
                        "properties": {
                            "tag": { "type": "string" }
                        }
                    },
                    "schema_mismatch_array": {
                        "type": "array",
                        "items": { "type": "integer" }
                    }
                }
            }),
            strict: None,
        });
        tools
    }

    fn order_arguments() -> String {
        let shipping = element(
            "shipping",
            &format!(
                "{}{}",
                element("city", "Singapore"),
                element("zip", "018956")
            ),
        );
        let first_item = element(
            "item",
            &format!("{}{}", element("sku", "book-001"), element("qty", "2")),
        );
        let second_item = element(
            "item",
            &format!("{}{}", element("sku", "pen-007"), element("qty", "5")),
        );
        let items = element("items", &format!("{first_item}{second_item}"));
        let metadata = element(
            "metadata",
            &format!("{}{}", element("score", "42"), element("rank", "7")),
        );
        let duplicate_demo = element(
            "duplicate_demo",
            &format!("{}{}", element("tag", "a"), element("tag", "b")),
        );
        let schema_mismatch_array = element(
            "schema_mismatch_array",
            &format!("{}{}", element("x", "1"), element("x", "2")),
        );

        [
            element("user_id", "42"),
            element("urgent", "true"),
            element("note", "Please leave at front desk."),
            shipping,
            items,
            metadata,
            duplicate_demo,
            schema_mismatch_array,
            element(
                "unknown_struct",
                &format!("{}{}", element("a", "1"), element("a", "2")),
            ),
        ]
        .join("")
    }

    #[test]
    fn minimax_m3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_parse_complete_extracts_single_tool_call() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                format!("{}{}", element("city", "Seattle"), element("days", "5")),
            )]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle", "days": 5 })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_preserves_prefix_and_ignores_trailing_text() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = format!(
            "Let me check. {} This trailing text is ignored.",
            build_tool_block(&[("get_weather", element("city", "Seattle"))])
        );
        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_parse_complete_extracts_multiple_invokes() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[
                ("get_weather", element("city", "Seattle")),
                ("get_weather", element("city", "NYC")),
            ]))
            .unwrap();

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "city": "NYC" })
        );
    }

    #[test]
    fn minimax_m3_invoke_body_junk_drops_rest_of_invoke() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                [
                    element("city", "Seattle"),
                    "I need to use the city above.".to_string(),
                    element("days", "5"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_converts_schema_types() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "convert",
                [
                    element("whole", "5.0"),
                    element("flag", "true"),
                    element("payload", r#"{"nested":true}"#),
                    element("items", "[1,2]"),
                    element("empty", "42"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_converts_nested_arguments() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[("create_order", order_arguments())]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "user_id": 42,
                "urgent": true,
                "note": "Please leave at front desk.",
                "shipping": {
                    "city": "Singapore",
                    "zip": 18956
                },
                "items": [
                    {
                        "sku": "book-001",
                        "qty": 2
                    },
                    {
                        "sku": "pen-007",
                        "qty": 5
                    }
                ],
                "metadata": {
                    "score": 42,
                    "rank": 7
                },
                "duplicate_demo": {
                    "tag": ["a", "b"]
                },
                "schema_mismatch_array": [1, 2],
                "unknown_struct": {
                    "a": ["1", "2"]
                }
            })
        );
    }

    #[test]
    fn minimax_m3_parse_complete_handles_multiline_leaf_parameters() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "calculate_area",
                [
                    element("shape", "\nrectangle\n"),
                    element("dimensions", r#"{"width":10,"height":20}"#),
                    element("precision", "2"),
                ]
                .join(""),
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "shape": "\nrectangle\n",
                "dimensions": { "width": 10, "height": 20 },
                "precision": 2,
            })
        );
    }

    #[test]
    fn minimax_m3_streaming_extracts_single_tool_call() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_streaming_preserves_prefix_text() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_streaming_preserves_ordered_events() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                TOOL_CALL_START,
                &invoke("get_weather", &element("city", "Seattle")),
                TOOL_CALL_END,
            ],
        );

        assert_eq!(output.events.len(), 2);
        assert_eq!(
            output.events[0],
            ToolParserEvent::Text("Let me check. ".to_string())
        );
        let ToolParserEvent::ToolCall(call) = &output.events[1] else {
            panic!("expected tool-call event");
        };
        assert_eq!(call.name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_streaming_handles_marker_split_across_chunks() {
        let text = build_tool_block(&[("get_weather", element("city", "Seattle"))]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn minimax_m3_streaming_extracts_multiple_invokes_in_order() {
        let text = build_tool_block(&[
            ("get_weather", element("city", "Seattle")),
            ("get_weather", element("city", "NYC")),
        ]);
        let chunks = split_by_chars(&text, 7);
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
    }

    #[test]
    fn minimax_m3_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">"
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_streaming_ignores_text_after_tool_block() {
        let text = format!(
            "{} ignored",
            build_tool_block(&[("get_weather", element("city", "Seattle"))])
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m3_finish_fails_incomplete_tool_call() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">"
            ))
            .unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m3_finish_recovers_after_bare_tool_block_start() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        parser.parse_chunk(TOOL_CALL_START).unwrap();

        let output = parser.finish().unwrap();
        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m3_finish_recovers_completed_invoke_with_whitespace_tail() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_complete(&format!(
                "{}\n{}\n  \n",
                TOOL_CALL_START,
                invoke("get_weather", &element("city", "Seattle"))
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_finish_fails_partial_outer_end_marker() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        parser
            .parse_chunk(&format!(
                "{}\n{}\n{}",
                TOOL_CALL_START,
                invoke("get_weather", &element("city", "Seattle")),
                &TOOL_CALL_END[..3]
            ))
            .unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m3_malformed_tool_call_fails_fast() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let error = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{ELEMENT_START}bad>{TOOL_CALL_END}"
            ))
            .unwrap_err();

        expect!["tool parser parsing failed: "].assert_eq(&error.to_report_string());
    }

    #[test]
    fn minimax_m3_mixed_content_is_preserved_as_text_field() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let body = element(
            "payload",
            &format!("text before {} text after", element("child", "value")),
        );
        let output = parser.parse_complete(&build_tool_block(&[("convert", body)])).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "payload": {
                    "child": "value",
                    "$text": "text before  text after"
                }
            })
        );
    }

    #[test]
    fn minimax_m3_mixed_text_field_avoids_child_name_collision() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let body = element(
            "payload",
            &format!(
                "text{}{}",
                element("$text", "child text"),
                element("child", "value")
            ),
        );
        let output = parser.parse_complete(&build_tool_block(&[("convert", body)])).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "payload": {
                    "$text": "child text",
                    "$$text": "text",
                    "child": "value"
                }
            })
        );
    }
}
