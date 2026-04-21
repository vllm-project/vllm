//! Default output processing pipeline.

mod reasoning;
mod structured;
mod tool;

use futures::Stream;

use self::reasoning::reasoning_event_stream;
use self::structured::structured_chat_event_stream;
use self::tool::tool_event_stream;
use crate::error::Result;
use crate::output::{AssistantEvent, ChatEventStream, ContentEvent, DecodedTextEventStream};
use crate::parser::reasoning::ReasoningParser;
use crate::parser::tool::ToolParser;

trait ContentEventStream = Stream<Item = Result<ContentEvent>> + Send + 'static;
trait AssistantEventStream = Stream<Item = Result<AssistantEvent>> + Send + 'static;

/// Request-scoped processors that adapt decoded text into structured chat events.
pub(crate) struct OutputProcessors {
    pub(crate) reasoning_parser: Option<Box<dyn ReasoningParser>>,
    pub(crate) tool_parser: Option<Box<dyn ToolParser>>,
}

/// Transforms a raw generate-output token stream into structured chat events
/// through three sequential stages once text decoding has already happened:
///
/// 1. [`reasoning_event_stream`] — reasoning/content separation
/// 2. [`tool_event_stream`] — tool-call parsing
/// 3. [`structured_chat_event_stream`] — final block assembly
pub(crate) fn output_stream(
    intermediate: bool,
    decoded: impl DecodedTextEventStream,
    OutputProcessors {
        reasoning_parser,
        tool_parser,
    }: OutputProcessors,
) -> Result<impl ChatEventStream> {
    let reasoning = reasoning_event_stream(decoded, reasoning_parser);
    let tool = tool_event_stream(reasoning, intermediate, tool_parser);
    let structured = structured_chat_event_stream(tool);
    Ok(structured)
}
