// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use vllm_parser::tool::{
    Result, StructuralTagBuilder, Tool, ToolParser, ToolParserError, ToolParserOutput,
};
use vllm_parser::unified::{
    UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput,
};
use vllm_tokenizer::Tokenizer;

/// Tokenizer stub used by unified-parser benchmarks.
struct BenchTokenizer;

impl Tokenizer for BenchTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        Ok(text.chars().map(|_| u32::MAX).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_tokenizer::Result<String> {
        Ok("\u{FFFD}".repeat(token_ids.len()))
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        Some(u32::MAX)
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        Some("\u{FFFD}".to_string())
    }
}

/// Bench-only adapter that exposes a unified parser through the tool-parser
/// benchmark harness.
///
/// Returns error if the unified parser produces reasoning events.
pub struct UnifiedToolParserAdapter<T> {
    inner: Box<dyn UnifiedParser>,
    _marker: std::marker::PhantomData<T>,
}

fn map_unified_error(error: UnifiedParserError) -> ToolParserError {
    ToolParserError::ParsingFailed {
        message: format!("unified parser failed: {error}"),
    }
}

fn append_unified_output(
    output: UnifiedParserOutput,
    tool_output: &mut ToolParserOutput,
) -> Result<()> {
    for event in output.events {
        match event {
            UnifiedParserEvent::Text(text) => tool_output.push_text(text),
            UnifiedParserEvent::ToolCall(call) => tool_output.push_call(call),
            UnifiedParserEvent::Reasoning(_) => {
                return Err(ToolParserError::ParsingFailed {
                    message: "unified parser emitted reasoning in tool-parser adapter".to_string(),
                });
            }
        }
    }
    Ok(())
}

impl<T: UnifiedParser> ToolParser for UnifiedToolParserAdapter<T> {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        let inner = T::create(tools, Arc::new(BenchTokenizer)).map_err(map_unified_error)?;
        Ok(Box::new(Self {
            inner,
            _marker: std::marker::PhantomData,
        }))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        self.inner.structural_tag_builder()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.inner.tool_call_id(tool_index)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        let mut unified_output = UnifiedParserOutput::default();
        let result = self.inner.parse_into(chunk, &mut unified_output).map_err(map_unified_error);
        append_unified_output(unified_output, output)?;
        result
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let unified_output = self.inner.finish().map_err(map_unified_error)?;
        let mut output = ToolParserOutput::default();
        append_unified_output(unified_output, &mut output)?;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}
