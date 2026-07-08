use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, peek, preceded, seq};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::{ParamElement, ParamInput, ToolSchemas};
use super::utils::{parse_buffered_event, safe_text_len};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MinimaxM3Mode {
    Text,
    ToolBlock,
    InsideInvoke,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM3Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    InvokeStart {
        name: String,
    },
    Param {
        name: String,
        value: ParamInput,
    },
    InvokeEnd,
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
/// tag.
///
/// Arguments are streamed incrementally, one JSON fragment per top-level
/// `<parameter>` element: the function name is emitted first (with empty
/// arguments), then each parameter is emitted as its own `{"name":value` /
/// `,"name":value` fragment, and finally a closing `}` fragment. Coalescing
/// all fragments for one tool index reconstructs the exact same JSON object as
/// a non-streaming parse. Nested object/array parameters are buffered and
/// emitted as one fragment covering their whole subtree.
pub struct MinimaxM3ToolParser {
    buffer: String,
    mode: MinimaxM3Mode,
    emitted_tool_count: usize,
    /// Function name of the invoke currently being streamed, used for
    /// per-parameter schema-aware conversion.
    current_tool_name: Option<String>,
    /// Whether the current invoke has already emitted its opening `{` and at
    /// least one parameter fragment (controls comma placement and the closing
    /// fragment).
    param_emitted: bool,
    tool_parameters: ToolSchemas,
}

impl MinimaxM3ToolParser {
    /// Create a MiniMax M3 tool parser.
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM3Mode::Text,
            emitted_tool_count: 0,
            current_tool_name: None,
            param_emitted: false,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Push a fragment that best-effort closes the JSON of the currently open
    /// invoke, so a truncated stream still yields a parseable object.
    fn push_invoke_close(&mut self, output: &mut ToolParserOutput) {
        let closing = if self.param_emitted { "}" } else { "{}" };
        output.push_call(ToolCallDelta {
            tool_index: self.emitted_tool_count,
            name: None,
            arguments: closing.to_string(),
        });
    }

    /// Apply one parsed MiniMax M3 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MinimaxM3Event::ToolBlockStart => self.mode = MinimaxM3Mode::ToolBlock,
            MinimaxM3Event::InvokeStart { name } => {
                self.current_tool_name = Some(name.clone());
                self.param_emitted = false;
                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments: String::new(),
                });
                self.mode = MinimaxM3Mode::InsideInvoke;
            }
            MinimaxM3Event::Param { name, value } => {
                let tool_name = self.current_tool_name.as_deref().unwrap_or("");
                let value = self.tool_parameters.convert_param_with_schema(tool_name, &name, value);
                let key = serde_json::to_string(&name).map_err(|error| {
                    parsing_failed!("failed to serialize parameter name: {}", error)
                })?;
                let value = serde_json::to_string(&value)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;
                let fragment =
                    format!("{}{}:{}", if self.param_emitted { "," } else { "{" }, key, value);
                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: None,
                    arguments: fragment,
                });
                self.param_emitted = true;
            }
            MinimaxM3Event::InvokeEnd => {
                self.push_invoke_close(output);
                self.emitted_tool_count += 1;
                self.current_tool_name = None;
                self.param_emitted = false;
                self.mode = MinimaxM3Mode::ToolBlock;
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
            parse_next_minimax_m3_event(input, self.mode)
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
            MinimaxM3Mode::InsideInvoke => {
                // The invoke header was emitted but the closing marker never
                // arrived (e.g. truncated generation). Degrade gracefully by
                // best-effort closing the JSON instead of raising, so the
                // caller still gets a parseable partial tool call.
                self.push_invoke_close(&mut output);
            }
            MinimaxM3Mode::ToolBlock => {
                // A tool block was opened but never closed. Any already-emitted
                // invokes are self-closed; leftover buffer is incomplete
                // structural markup and is dropped as best-effort rather than
                // raising a hard error.
            }
            MinimaxM3Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM3Mode::Text;
        self.emitted_tool_count = 0;
        self.current_tool_name = None;
        self.param_emitted = false;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a MiniMax M3 event for the current parser mode.
fn parse_next_minimax_m3_event(
    input: &mut MinimaxM3Input<'_>,
    mode: MinimaxM3Mode,
) -> ModalResult<MinimaxM3Event> {
    match mode {
        MinimaxM3Mode::Text => parse_text_event(input),
        MinimaxM3Mode::ToolBlock => parse_tool_block_event(input),
        MinimaxM3Mode::InsideInvoke => parse_inside_invoke_event(input),
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

/// Parse one event inside a MiniMax M3 tool block (between invokes).
fn parse_tool_block_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((tool_block_end_event, invoke_start_event)).parse_next(input)
}

/// Parse a MiniMax M3 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM3Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse only the header of a MiniMax M3 invoke block: `<invoke name="X">`.
///
/// The body (parameters) and the closing `</invoke>` are parsed incrementally
/// afterwards while in `InsideInvoke` mode.
fn invoke_start_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    let name = seq!(
        _: ws0,
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
    )
    .parse_next(input)?;

    Ok(MinimaxM3Event::InvokeStart {
        name: name.0.trim().to_string(),
    })
}

/// Parse one event inside a MiniMax M3 invoke block.
fn parse_inside_invoke_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    alt((invoke_end_event, param_event, invoke_body_junk_event)).parse_next(input)
}

/// Parse a MiniMax M3 invoke end marker.
fn invoke_end_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (ws0, literal(INVOKE_END))
        .value(MinimaxM3Event::InvokeEnd)
        .parse_next(input)
}

/// Parse one top-level parameter element inside an invoke block.
fn param_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    let element = preceded(ws0, parameter_element).parse_next(input)?;
    Ok(MinimaxM3Event::Param {
        name: element.name,
        value: element.value,
    })
}

/// Tolerantly drop the remainder of an invoke body up to and including the
/// closing marker.
///
/// Mirrors the non-streaming behavior of keeping already-emitted parameters
/// and discarding ordinary text (or otherwise unexpected content) that appears
/// where a parameter or `</invoke>` was expected. This branch is only reached
/// after `invoke_end_event` and `param_event` have both backtracked, i.e. the
/// content is definitively neither the invoke end nor a (partial) parameter
/// element.
fn invoke_body_junk_event(input: &mut MinimaxM3Input<'_>) -> ModalResult<MinimaxM3Event> {
    (take_until(0.., INVOKE_END), literal(INVOKE_END))
        .value(MinimaxM3Event::InvokeEnd)
        .parse_next(input)
}

/// Parse a MiniMax M3 parameter element (streaming, Partial-aware).
fn parameter_element(input: &mut MinimaxM3Input<'_>) -> ModalResult<ParamElement> {
    let name = open_element_tag(input)?.to_string();
    let value = element_body(input, &name)?;
    close_element_tag(input, &name)?;
    Ok(ParamElement { name, value })
}

/// Parse a MiniMax M3 opening element tag.
fn open_element_tag<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
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
fn close_element_tag(input: &mut MinimaxM3Input<'_>, name: &str) -> ModalResult<()> {
    literal(ELEMENT_END_START).void().parse_next(input)?;
    literal(name).void().parse_next(input)?;
    literal(">").void().parse_next(input)
}

/// Parse the body of one MiniMax M3 element (streaming, Partial-aware).
///
/// Returns `Incomplete` (via `take_until` / `literal` on the `Partial` stream)
/// whenever the element — including any nested object/array subtree — has not
/// fully arrived yet, so callers hold the buffer until the whole element is
/// available.
fn element_body(input: &mut MinimaxM3Input<'_>, closing_name: &str) -> ModalResult<ParamInput> {
    let close_tag = format!("{ELEMENT_END_START}{closing_name}>");
    let mut text = String::new();
    let mut elements = Vec::new();

    loop {
        text.push_str(text_until_namespace(input)?);

        // Input now begins with the namespace marker. Decide whether this is
        // the close tag for this element (peek, do not consume — the caller's
        // `close_element_tag` consumes it) or a nested child element. A partial
        // marker yields `Incomplete` and holds the buffer.
        let child = alt((
            peek(literal(close_tag.as_str())).value(None::<ParamElement>),
            parameter_element.map(Some),
        ))
        .parse_next(input)?;

        match child {
            None => break,
            Some(child) => elements.push(child),
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
fn text_until_namespace<'i>(input: &mut MinimaxM3Input<'i>) -> ModalResult<&'i str> {
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
    use crate::tool::{Tool, ToolCallDelta, ToolParserTestExt as _};

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

    /// Collect raw (non-coalesced) deltas by feeding chunks one at a time,
    /// including the `finish()` flush.
    fn collect_raw_deltas(parser: &mut MinimaxM3ToolParser, chunks: &[&str]) -> Vec<ToolCallDelta> {
        let mut deltas = Vec::new();
        for chunk in chunks {
            let output = parser.parse_chunk(chunk).unwrap();
            deltas.extend(output.calls().into_iter().cloned());
        }
        deltas.extend(parser.finish().unwrap().calls().into_iter().cloned());
        deltas
    }

    /// The argument-bearing deltas (those without a name) for one tool index.
    fn arg_fragments(deltas: &[ToolCallDelta]) -> Vec<String> {
        deltas
            .iter()
            .filter(|delta| delta.name.is_none())
            .map(|delta| delta.arguments.clone())
            .collect()
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
    fn minimax_m3_streaming_emits_tool_call_header_before_arguments() {
        // Incremental contract: the function name is emitted (with empty
        // arguments) as soon as the invoke header parses, before any argument
        // fragment. This mirrors Kimi K2's proven streaming behavior.
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">"
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].tool_index, 0);
        // No argument fragment yet: the name delta carries empty arguments.
        assert_eq!(output.calls()[0].arguments, "");
    }

    #[test]
    fn minimax_m3_streaming_emits_multiple_argument_fragments() {
        // A multi-parameter tool call must stream as multiple argument
        // fragments (one per parameter, plus the closing brace), never a
        // single blob.
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let deltas = collect_raw_deltas(
            &mut parser,
            &[
                TOOL_CALL_START,
                &invoke(
                    "get_weather",
                    &format!("{}{}", element("city", "Seattle"), element("days", "5")),
                ),
                TOOL_CALL_END,
            ],
        );

        let fragments = arg_fragments(&deltas);
        // city fragment, days fragment, closing brace => at least 2 (here 3).
        assert!(
            fragments.len() > 1,
            "expected multiple argument fragments, got {fragments:?}"
        );
        // No single fragment contains the whole arguments object.
        let complete = r#"{"city":"Seattle","days":5}"#;
        assert!(
            fragments.iter().all(|fragment| fragment != complete
                && !(fragment.contains("city") && fragment.contains("days"))),
            "no single fragment should contain the whole arguments: {fragments:?}"
        );
        // Coalescing the fragments reconstructs the exact complete JSON.
        assert_eq!(fragments.concat(), complete);
    }

    #[test]
    fn minimax_m3_streaming_coalesces_to_exact_complete_json() {
        // Byte-exact backward compatibility with the non-streaming path
        // (serde_json is built with `preserve_order`, so key order is
        // document order in both paths).
        let text = build_tool_block(&[(
            "get_weather",
            format!("{}{}", element("city", "Seattle"), element("days", "5")),
        )]);
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let output = collect_stream(&mut parser, &split_by_chars(&text, 1));

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Seattle","days":5}"#);
    }

    #[test]
    fn minimax_m3_streaming_is_chunk_split_invariant() {
        // Feeding the same input split at every possible boundary must yield
        // identical coalesced output.
        let text = build_tool_block(&[("create_order", order_arguments())]);
        let expected = {
            let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
            parser.parse_complete(&text).unwrap()
        };

        for size in [1usize, 2, 3, 5, 7, 13, 29] {
            let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
            let output = collect_stream(&mut parser, &split_by_chars(&text, size));
            assert_eq!(
                output, expected,
                "chunk size {size} produced different output"
            );
        }
    }

    #[test]
    fn minimax_m3_streaming_buffers_nested_param_as_single_fragment() {
        // A nested object parameter is emitted as ONE fragment covering its
        // whole subtree, even when streamed one character at a time.
        let text = build_tool_block(&[(
            "create_order",
            element(
                "shipping",
                &format!("{}{}", element("city", "Singapore"), element("zip", "018956")),
            ),
        )]);
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let deltas = collect_raw_deltas(&mut parser, &split_by_chars(&text, 1));

        let fragments = arg_fragments(&deltas);
        let shipping_fragments: Vec<&String> =
            fragments.iter().filter(|fragment| fragment.contains("shipping")).collect();
        assert_eq!(
            shipping_fragments.len(),
            1,
            "nested object must arrive in exactly one fragment, got {fragments:?}"
        );
        // That single fragment carries the whole nested subtree.
        assert_eq!(
            shipping_fragments[0],
            &r#"{"shipping":{"city":"Singapore","zip":18956}"#.to_string()
        );
        assert_eq!(
            serde_json::from_str::<Value>(&fragments.concat()).unwrap(),
            json!({ "shipping": { "city": "Singapore", "zip": 18956 } })
        );
    }

    #[test]
    fn minimax_m3_streaming_nested_array_of_objects_coalesces_to_valid_json() {
        // Real production failing shape: an array of objects, e.g.
        // `comments: [{category, content}, ...]`. Each array element is itself a
        // nested object. The whole `comments` parameter must buffer into ONE
        // argument fragment (covering the entire subtree) and, streamed at every
        // char boundary, must coalesce to the exact same valid JSON as a
        // non-streaming parse.
        let tools = vec![Tool {
            name: "submit_review".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": { "type": "string" },
                                "content": { "type": "string" }
                            }
                        }
                    }
                }
            }),
            strict: None,
        }];

        let comment_a = element(
            "comment",
            &format!(
                "{}{}",
                element("category", "bug"),
                element("content", "crashes on start")
            ),
        );
        let comment_b = element(
            "comment",
            &format!(
                "{}{}",
                element("category", "ux"),
                element("content", "confusing layout")
            ),
        );
        let body = format!(
            "{}{}",
            element("title", "Beta feedback"),
            element("comments", &format!("{comment_a}{comment_b}")),
        );
        let text = build_tool_block(&[("submit_review", body)]);

        let expected = json!({
            "title": "Beta feedback",
            "comments": [
                { "category": "bug", "content": "crashes on start" },
                { "category": "ux", "content": "confusing layout" }
            ]
        });

        // Non-streaming reference: the array-of-objects converts correctly.
        {
            let mut parser = MinimaxM3ToolParser::new(&tools);
            let output = parser.parse_complete(&text).unwrap();
            assert_eq!(output.calls().len(), 1);
            assert_eq!(output.calls()[0].name.as_deref(), Some("submit_review"));
            assert_eq!(
                serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
                expected
            );
        }

        // Streaming one character at a time: the `comments` array (with its
        // nested objects) must arrive as exactly one fragment, and coalescing
        // every fragment reconstructs valid JSON identical to the reference.
        let mut parser = MinimaxM3ToolParser::new(&tools);
        let deltas = collect_raw_deltas(&mut parser, &split_by_chars(&text, 1));
        let fragments = arg_fragments(&deltas);
        let comments_fragments: Vec<&String> =
            fragments.iter().filter(|fragment| fragment.contains("comments")).collect();
        assert_eq!(
            comments_fragments.len(),
            1,
            "nested array-of-objects must arrive in one fragment, got {fragments:?}"
        );
        assert_eq!(
            serde_json::from_str::<Value>(&fragments.concat()).unwrap(),
            expected
        );
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
    fn minimax_m3_finish_degrades_incomplete_tool_call() {
        // Truncated mid-invoke: finish() must NOT error. It best-effort closes
        // the JSON so the caller still gets a parseable partial tool call.
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let mut output = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">"
            ))
            .unwrap();
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        // No parameters arrived, so the best-effort close is an empty object.
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({})
        );
    }

    #[test]
    fn minimax_m3_finish_degrades_truncated_after_partial_params() {
        // Truncated after some parameters streamed: best-effort close yields a
        // valid object containing the parameters received so far.
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let mut output = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{INVOKE_START} name=\"get_weather\">{}",
                element("city", "Seattle"),
            ))
            .unwrap();
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
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
    fn minimax_m3_finish_degrades_partial_outer_end_marker() {
        // A completed invoke followed by a partial tool-block end marker must
        // degrade gracefully (the invoke is already emitted), not error.
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let mut output = parser
            .parse_chunk(&format!(
                "{}\n{}\n{}",
                TOOL_CALL_START,
                invoke("get_weather", &element("city", "Seattle")),
                &TOOL_CALL_END[..3]
            ))
            .unwrap();
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m3_malformed_tool_call_fails_fast() {
        let mut parser = MinimaxM3ToolParser::new(&m3_test_tools());
        let error = parser
            .parse_chunk(&format!(
                "{TOOL_CALL_START}{ELEMENT_START}bad>{TOOL_CALL_END}"
            ))
            .unwrap_err();

        expect![[r#"tool parser parsing failed: near "]<]minimax[>[<bad>]<]minimax[>[</tool_call>": "#]].assert_eq(&error.to_report_string());
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
