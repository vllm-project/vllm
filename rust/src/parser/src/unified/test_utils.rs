// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared test scaffolding for unified parser modules.

use vllm_tokenizer::DecodedText;

use super::{Result, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};
use crate::tool::ToolCallDelta;

pub(crate) trait UnifiedParserTestExt {
    fn parse_chunk(&mut self, chunk: &str) -> Result<UnifiedParserOutput>;
    fn parse_complete(&mut self, text: &str) -> Result<UnifiedParserOutput>;
}

impl<T: UnifiedParser + ?Sized> UnifiedParserTestExt for T {
    fn parse_chunk(&mut self, chunk: &str) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        self.parse_into(DecodedText::unattributed(chunk), &mut output)?;
        Ok(output)
    }

    fn parse_complete(&mut self, text: &str) -> Result<UnifiedParserOutput> {
        let mut output = self.parse_chunk(text)?;
        output.append(self.finish()?);
        Ok(output)
    }
}

pub(crate) trait UnifiedOutputTestExt {
    fn normal_text(&self) -> String;
    fn reasoning_text(&self) -> String;
    fn calls(&self) -> Vec<ToolCallDelta>;
}

impl UnifiedOutputTestExt for UnifiedParserOutput {
    fn normal_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn calls(&self) -> Vec<ToolCallDelta> {
        self.events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }
}

pub(crate) fn collect_stream<P: UnifiedParser + ?Sized>(
    parser: &mut P,
    chunks: &[&str],
) -> UnifiedParserOutput {
    let mut output = UnifiedParserOutput::default();
    for chunk in chunks {
        output.append(parser.parse_chunk(chunk).unwrap());
    }
    output.append(parser.finish().unwrap());
    output
}

pub(crate) fn first_call(output: &UnifiedParserOutput) -> ToolCallDelta {
    output.calls().first().expect("expected one tool call").clone()
}
