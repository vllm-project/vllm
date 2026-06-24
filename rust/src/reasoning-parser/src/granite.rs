use vllm_tokenizer::DynTokenizer;

use super::{ReasoningDelta, ReasoningParser, Result};

/// Markers that open the thought section (reasoning phase).
const THINK_STARTS: [&str; 2] = ["Here's my thought process:", "Here is my thought process:"];
/// Markers that open the response (visible content) section.
const RESPONSE_STARTS: [&str; 2] = ["Here's my response:", "Here is my response:"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    BeforeThink,
    InReasoning,
    InContent,
}

impl Phase {
    fn next(self) -> Self {
        match self {
            Phase::BeforeThink => Phase::InReasoning,
            Phase::InReasoning | Phase::InContent => Phase::InContent,
        }
    }
}

/// Reasoning parser for IBM Granite thinking models (3.x and 4.x).
///
/// Granite frames reasoning with plain-text phrase markers rather than special
/// tokens: `(Here's|Here is) my thought process:` opens the thought section and
/// `(Here's|Here is) my response:` opens the visible content. Because the
/// markers are multi-token phrases (not single vocab tokens), this parser
/// cannot reuse `DelimitedReasoningParser`; it runs its own incremental state
/// machine, holding back any trailing text that could still complete a marker
/// across streaming chunk boundaries.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/granite_reasoning_parser.py>
pub struct GraniteReasoningParser {
    phase: Phase,
    buffer: String,
}

impl GraniteReasoningParser {
    /// Create a Granite reasoning parser.
    fn new() -> Self {
        Self {
            phase: Phase::BeforeThink,
            buffer: String::new(),
        }
    }

    /// Markers that end the current phase, or `None` once in the content phase.
    fn phase_markers(&self) -> Option<&'static [&'static str]> {
        match self.phase {
            Phase::BeforeThink => Some(&THINK_STARTS),
            Phase::InReasoning => Some(&RESPONSE_STARTS),
            Phase::InContent => None,
        }
    }

    /// Drain the buffer through the phase machine, returning a reasoning delta.
    ///
    /// Complete markers are consumed (advancing the phase) before any trailing
    /// partial-marker fragment is held back, so a single chunk that crosses a
    /// phase boundary is split correctly and the held fragment is matched
    /// against the phase the stream actually ends in.
    fn drain_buffer(&mut self) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();
        loop {
            let Some(markers) = self.phase_markers() else {
                delta.push_content(&self.buffer);
                self.buffer.clear();
                break;
            };

            if let Some((idx, marker_len)) = first_complete_marker(&self.buffer, markers) {
                push_for_phase(&mut delta, self.phase, &self.buffer[..idx]);
                self.buffer.drain(..idx + marker_len);
                self.phase = self.phase.next();
                continue;
            }

            // No complete marker: hold back any partial marker, emit the rest.
            let partial = partial_suffix_len(&self.buffer, markers);
            let stable_len = self.buffer.len() - partial;
            push_for_phase(&mut delta, self.phase, &self.buffer[..stable_len]);
            self.buffer.drain(..stable_len);
            break;
        }
        delta
    }
}

impl ReasoningParser for GraniteReasoningParser {
    fn create(_tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()))
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        self.buffer.push_str(delta);
        Ok(self.drain_buffer())
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        // Any held-back fragment never completed a marker; flush it as text.
        let mut delta = ReasoningDelta::default();
        push_for_phase(&mut delta, self.phase, &self.buffer);
        self.buffer.clear();
        Ok(delta)
    }
}

/// Append `text` to the field that matches the current phase.
fn push_for_phase(delta: &mut ReasoningDelta, phase: Phase, text: &str) {
    match phase {
        Phase::InReasoning => delta.push_reasoning(text),
        Phase::BeforeThink | Phase::InContent => delta.push_content(text),
    }
}

/// Find the earliest complete marker in `text`, returning `(start, len)`.
fn first_complete_marker(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|idx| (idx, marker.len())))
        .min_by_key(|(idx, _)| *idx)
}

/// Return the length of the longest trailing fragment of `text` that is a
/// proper prefix of any marker (and so could still complete on the next chunk).
fn partial_suffix_len(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for idx in text.char_indices().map(|(idx, _)| idx) {
        let suffix = &text[idx..];
        for marker in markers {
            if marker.len() > suffix.len() && marker.starts_with(suffix) {
                best = best.max(suffix.len());
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::GraniteReasoningParser;
    use crate::ReasoningParser;

    /// Feed `chunks` through `push`/`finish` and return the coalesced split.
    fn run_streaming(chunks: &[&str]) -> (Option<String>, Option<String>) {
        let mut parser = GraniteReasoningParser::new();
        let mut reasoning = String::new();
        let mut content = String::new();

        for delta in chunks {
            let result = parser.push(delta).unwrap();
            if let Some(next) = result.reasoning {
                reasoning.push_str(&next);
            }
            if let Some(next) = result.content {
                content.push_str(&next);
            }
        }

        let final_delta = parser.finish().unwrap();
        if let Some(next) = final_delta.reasoning {
            reasoning.push_str(&next);
        }
        if let Some(next) = final_delta.content {
            content.push_str(&next);
        }

        (
            (!reasoning.is_empty()).then_some(reasoning),
            (!content.is_empty()).then_some(content),
        )
    }

    /// Split a string into one-character chunks to stress marker reassembly.
    fn char_chunks(text: &str) -> Vec<String> {
        text.chars().map(|c| c.to_string()).collect()
    }

    #[test]
    fn granite_reasoning_streaming_splits_thought_and_response() {
        let cases = [
            (
                "no_reasoning",
                vec!["This is content"],
                None,
                Some("This is content"),
            ),
            (
                "reasoning_and_content",
                vec![
                    "Here is my thought process:This is a reasoning section\
                     Here is my response:This is the rest",
                ],
                Some("This is a reasoning section"),
                Some("This is the rest"),
            ),
            (
                "complete_empty_response",
                vec!["Here is my thought process:This is a reasoning sectionHere is my response:"],
                Some("This is a reasoning section"),
                None,
            ),
            (
                "multiline",
                vec![
                    "Here is my thought process:This\nThatHere is my response:This is the rest\nThat",
                ],
                Some("This\nThat"),
                Some("This is the rest\nThat"),
            ),
            (
                "apostrophe_spelling",
                vec!["Here's my thought process:Some thinkingHere's my response:Some response"],
                Some("Some thinking"),
                Some("Some response"),
            ),
            (
                "thought_only",
                vec!["Here is my thought process:reasoning with no response marker"],
                Some("reasoning with no response marker"),
                None,
            ),
            (
                "one_chunk_spanning_both_markers",
                vec!["Here is my thought process: foo Here is my response: bar"],
                Some(" foo "),
                Some(" bar"),
            ),
            ("empty", vec![""], None, None),
        ];

        for (name, output, expected_reasoning, expected_content) in cases {
            let (reasoning, content) = run_streaming(&output);
            assert_eq!(reasoning.as_deref(), expected_reasoning, "{name}");
            assert_eq!(content.as_deref(), expected_content, "{name}");
        }
    }

    #[test]
    fn granite_reasoning_holds_markers_split_across_chunks() {
        let input = "Here is my thought process:This is a reasoning section\
                     Here is my response:This is the rest";
        let chunks = char_chunks(input);
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

        let (reasoning, content) = run_streaming(&chunk_refs);

        assert_eq!(reasoning.as_deref(), Some("This is a reasoning section"));
        assert_eq!(content.as_deref(), Some("This is the rest"));
    }

    #[test]
    fn granite_reasoning_recovers_broken_response_marker() {
        // A "Here is " inside the reasoning briefly looks like the response
        // marker, then breaks; it must be emitted as reasoning, not dropped.
        let input = "Here is my thought process:thinking Here is not a marker";
        let chunks = char_chunks(input);
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

        let (reasoning, content) = run_streaming(&chunk_refs);

        assert_eq!(reasoning.as_deref(), Some("thinking Here is not a marker"));
        assert_eq!(content, None);
    }

    #[test]
    fn granite_reasoning_keeps_text_before_thought_marker_as_content() {
        let (reasoning, content) = run_streaming(&[
            "Preamble Here is my thought process:the reasoningHere is my response:",
        ]);

        assert_eq!(reasoning.as_deref(), Some("the reasoning"));
        assert_eq!(content.as_deref(), Some("Preamble "));
    }
}
